import asyncio
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import time

import pytest
import pytest_asyncio

from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.model import PluginRuntime
from agent.plugin_composition.workload_slots import (
    Workload,
    WorkloadData,
    WorkloadHealth,
    WorkloadLimits,
    WorkloadPort,
    _WorkloadDeclarations,
)
from agent.plugins.workload_generation_host import WorkloadGenerationHost, _http_health
from agent.workloads.model import (
    WorkloadEndpoint,
    WorkloadLease,
    WorkloadStartReceipt,
    WorkloadStopReceipt,
)


class Controller:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.effects = []

    async def start(self, request):
        self.started.append(request)
        self.effects.append(("start", request.generation_id))
        lease = WorkloadLease(
            request.workspace_id,
            request.plugin_id,
            request.workload,
            request.mode,
            request.transaction_id,
            request.generation_id,
            "container-A",
            request.spec_digest,
        )
        return WorkloadStartReceipt(
            lease, (WorkloadEndpoint("gateway", "http://container-A:8080"),), None
        )

    async def stop(self, lease):
        self.stopped.append(lease)
        self.effects.append(("stop", lease.generation_id))
        return WorkloadStopReceipt(lease, True, True)

    async def cleanup_candidates(self, workspace_id: str) -> tuple[WorkloadStopReceipt, ...]:
        _ = workspace_id
        return ()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 503])
async def test_health_probe_does_not_mistake_busy_loop_for_unhealthy_workload(status):
    """主循环停顿时，按健康接口的真实结果判定。"""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    # 故意让主循环停顿超过探测预算，重现启动和同步插件代码造成的延迟。
    pause = asyncio.get_running_loop().call_later(0.001, time.sleep, 0.2)
    try:
        result = await _http_health(f"http://localhost:{server.server_port}/health", 0.05)
        assert result == (status == 200, f"HTTP {status}")
    finally:
        pause.cancel()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        worker.join()


@pytest_asyncio.fixture(loop_scope="session")
async def workload(tmp_path):
    root = CompositionRoot("test")
    declarations = _WorkloadDeclarations()

    async def apply(ctx):
        await declarations.register(
            ctx,
            Workload(
                name="desktop",
                image="example/desktop@sha256:" + "a" * 64,
                command=("/start",),
                ports=(WorkloadPort("gateway", 8080),),
                data=(WorkloadData("profile", "/data"),),
                health=WorkloadHealth("gateway"),
                limits=WorkloadLimits(0, 0, 0),
            ),
        )

    await root.mount(
        apply,
        name="desktop",
        runtime=PluginRuntime(
            "desktop",
            "formal-A",
            tmp_path,
            tmp_path,
            tmp_path,
            {},
        ),
    )
    registry = declarations.freeze(root.instance_token)
    binding = next(iter(registry.values()))
    controller = Controller()

    async def healthy(url, timeout):
        return True, "ready"

    host = WorkloadGenerationHost(
        controller, workspace_id="workspace", health_probe=healthy
    )
    await host.start_generation(
        "formal-A", "desktop", {"desktop": binding}, mode="formal"
    )
    try:
        yield host, controller, binding
    finally:
        await host.stop_generation("formal-A")
        await root.dispose()


@pytest.mark.asyncio
async def test_borrow_never_starts_or_stops_a_matching_desktop(workload):
    host, controller, binding = workload
    descriptor = binding.descriptor
    async with host.borrow(descriptor) as endpoints:
        assert endpoints["gateway"] == "http://container-A:8080"
        with pytest.raises(TypeError):
            endpoints["gateway"] = "http://new:8080"
    assert len(controller.started) == 1
    assert controller.stopped == []
    for changed in (
        replace(descriptor, image="example/desktop@sha256:" + "b" * 64),
        replace(descriptor, data=(WorkloadData("other-profile", "/data"),)),
        replace(descriptor, owner="another-plugin"),
    ):
        with pytest.raises(RuntimeError, match="相同 Workload"):
            async with host.borrow(changed):
                pytest.fail("changed resource was borrowed")
    assert len(controller.started) == 1
    assert controller.stopped == []


@pytest.mark.asyncio
async def test_stop_waits_for_borrow_and_cancellation_does_not_lose_stop_owner(
    workload,
):
    host, controller, binding = workload
    descriptor = binding.descriptor
    entered = asyncio.Event()
    cleanup = host._cleanup

    async def observed_cleanup(generation):
        entered.set()
        await cleanup(generation)

    host._cleanup = observed_cleanup
    async with host.borrow(descriptor):
        stop = asyncio.create_task(host.stop_generation("formal-A"))
        await entered.wait()
        assert not controller.stopped
        stop.cancel()
        with pytest.raises(RuntimeError, match="相同 Workload"):
            async with host.borrow(descriptor):
                pytest.fail("stopping resource was borrowed")
    with pytest.raises(asyncio.CancelledError):
        await stop
    assert len(controller.stopped) == 1
    assert host.get("formal-A") is None
    with pytest.raises(RuntimeError, match="相同 Workload"):
        async with host.borrow(descriptor):
            pytest.fail("removed resource was borrowed")


@pytest.mark.asyncio
async def test_replacement_cannot_reuse_endpoint_until_borrow_is_closed(workload):
    host, controller, binding = workload
    stopping = asyncio.Event()
    replacing = asyncio.Event()
    cleanup = host._cleanup

    async def observed_cleanup(generation):
        stopping.set()
        await cleanup(generation)

    host._cleanup = observed_cleanup
    new_binding = replace(
        binding,
        descriptor=replace(
            binding.descriptor, image="example/desktop@sha256:" + "b" * 64
        ),
    )

    async def start_new():
        replacing.set()
        return await host.start_generation(
            "formal-B", "desktop", {"desktop": new_binding}, mode="formal"
        )

    async with host.borrow(binding.descriptor):
        stop = asyncio.create_task(host.stop_generation("formal-A"))
        await stopping.wait()
        start = asyncio.create_task(start_new())
        await replacing.wait()
        assert controller.effects == [("start", "formal-A")]
    try:
        await stop
        await start
        assert controller.effects == [
            ("start", "formal-A"),
            ("stop", "formal-A"),
            ("start", "formal-B"),
        ]
    finally:
        await host.stop_generation("formal-B")

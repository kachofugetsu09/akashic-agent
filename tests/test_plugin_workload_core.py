from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent.plugin_composition import (
    MCP_SERVERS,
    WORKLOADS,
    CompositionRoot,
    McpServerDefinition,
    PluginRuntime,
    Workload,
    WorkloadData,
    WorkloadEnv,
    WorkloadHealth,
    WorkloadLimits,
    WorkloadPort,
)
from agent.plugin_composition.mcp_slots import PluginMcpServers
from agent.plugin_composition.workload_slots import (
    PluginWorkloads,
    _freeze_plugin_workloads,
)
from agent.plugin_composition.model import CompositionError
from agent.plugins.composition_generation_host import CompositionGenerationHost
from agent.plugins.generation import GateResult, PluginContributions, PluginGeneration
from agent.plugins.scope import PluginScope
from agent.plugins.snapshot import RuntimeSnapshotCompiler
from agent.plugins.manager import PluginManager
from agent.plugins.workload_generation_host import WorkloadGenerationHost
from bus.event_bus import EventBus
from agent.tools.registry import ToolRegistry
from agent.workloads.model import (
    WorkloadEndpoint,
    WorkloadLease,
    WorkloadStartRequest,
    WorkloadStartReceipt,
    WorkloadStopReceipt,
)
from agent.workloads.client import WorkloadEffectUnknown

_IMAGE = "example.invalid/worker@sha256:" + "a" * 64


class _ReadyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


class _FakeController:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.starts: list[WorkloadStartRequest] = []
        self.stops: list[WorkloadLease] = []

    async def start(self, request: WorkloadStartRequest) -> WorkloadStartReceipt:
        self.starts.append(request)
        lease = WorkloadLease(
            workspace_id=request.workspace_id,
            plugin_id=request.plugin_id,
            workload=request.workload,
            mode=request.mode,
            transaction_id=request.transaction_id,
            generation_id=request.generation_id,
            container_id=f"container-{len(self.starts)}",
            spec_digest=request.spec_digest,
        )
        return WorkloadStartReceipt(
            lease,
            (WorkloadEndpoint("gateway", self.endpoint),),
            None,
        )

    async def stop(self, lease: WorkloadLease) -> WorkloadStopReceipt:
        self.stops.append(lease)
        return WorkloadStopReceipt(lease, True, True)

    async def cleanup_candidates(
        self, workspace_id: str
    ) -> tuple[WorkloadStopReceipt, ...]:
        _ = workspace_id
        return ()


class _StopOnceController(_FakeController):
    def __init__(self, endpoint: str, fail_name: str) -> None:
        super().__init__(endpoint)
        self.fail_name = fail_name
        self.failed = False

    async def stop(self, lease: WorkloadLease) -> WorkloadStopReceipt:
        self.stops.append(lease)
        if lease.workload == self.fail_name and not self.failed:
            self.failed = True
            raise RuntimeError("temporary stop failure")
        return WorkloadStopReceipt(lease, True, True)


class _LostStartResponseController(_FakeController):
    def __init__(self, endpoint: str) -> None:
        super().__init__(endpoint)
        self.receipt: WorkloadStartReceipt | None = None

    async def start(self, request: WorkloadStartRequest) -> WorkloadStartReceipt:
        if self.receipt is None:
            self.receipt = await super().start(request)
            raise WorkloadEffectUnknown("response lost")
        self.starts.append(request)
        return self.receipt


def _workload() -> Workload:
    return Workload(
        name="worker",
        image=_IMAGE,
        command=("serve",),
        ports=(WorkloadPort("gateway", 8080),),
        data=(WorkloadData("state", "/data"),),
        health=WorkloadHealth("gateway", "/health", 5.0),
        limits=WorkloadLimits(128, 1.0, 64),
    )


def _named_workload(name: str) -> Workload:
    workload = _workload()
    return Workload(
        name=name,
        image=workload.image,
        command=workload.command,
        ports=workload.ports,
        data=(WorkloadData(name, f"/{name}"),),
        health=workload.health,
        limits=workload.limits,
    )


def _runtime(plugin_dir: Path, data_dir: Path, workspace: Path) -> PluginRuntime:
    return PluginRuntime(
        plugin_id="fixture",
        generation_id="fixture:test",
        plugin_dir=plugin_dir,
        data_dir=data_dir,
        workspace=workspace,
        config=None,
    )


@pytest.mark.asyncio
async def test_workload_registry_is_owner_scoped_and_fiber_owned(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("workload-root")
    declarations = PluginWorkloads(root.instance_token)
    await root.context.provide(WORKLOADS, declarations)
    plugin_dir = tmp_path / "fixture"
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    plugin_dir.mkdir()
    data_dir.mkdir()
    workspace.mkdir()

    async def apply(ctx) -> None:
        await ctx.require(WORKLOADS).register(ctx, _workload())

    fiber = await root.mount(
        apply,
        name="fixture",
        inject=(WORKLOADS,),
        runtime=_runtime(plugin_dir, data_dir, workspace),
    )
    registry = _freeze_plugin_workloads(declarations, root.instance_token)
    binding = registry.owned("fixture", "worker")
    assert binding is not None
    assert binding.descriptor.image == _IMAGE
    assert binding.descriptor.user_namespaces is False

    await fiber.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_one_plugin_cannot_give_two_workloads_the_same_writable_data(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("workload-writer-owner")
    workloads = PluginWorkloads(root.instance_token)
    await root.context.provide(WORKLOADS, workloads)
    plugin_dir = tmp_path / "fixture"
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    plugin_dir.mkdir()
    data_dir.mkdir()
    workspace.mkdir()

    async def apply(ctx) -> None:
        await ctx.require(WORKLOADS).register(ctx, _workload())
        await ctx.require(WORKLOADS).register(
            ctx,
            Workload(
                name="other",
                image=_IMAGE,
                command=("serve",),
                ports=(WorkloadPort("gateway", 8081),),
                data=(WorkloadData("state", "/other"),),
                health=WorkloadHealth("gateway", "/health", 5.0),
                limits=WorkloadLimits(128, 1.0, 64),
            ),
        )

    await root.mount(
        apply,
        name="fixture",
        inject=(WORKLOADS,),
        runtime=_runtime(plugin_dir, data_dir, workspace),
    )
    with pytest.raises(CompositionError, match="多个 writer"):
        _freeze_plugin_workloads(workloads, root.instance_token)
    await root.dispose()


@pytest.mark.asyncio
async def test_composition_host_injects_same_owner_workload_url_into_mcp(
    tmp_path: Path,
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReadyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    controller = _FakeController(endpoint)

    plugin_dir = tmp_path / "fixture"
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    plugin_dir.mkdir()
    data_dir.mkdir()
    workspace.mkdir()
    mcp_script = plugin_dir / "mcp.py"
    mcp_script.write_text(
        "import json, os, sys\n"
        "for raw in sys.stdin:\n"
        " msg=json.loads(raw); method=msg.get('method')\n"
        " if method=='initialize': result={'protocolVersion':'2025-11-25'}\n"
        " elif method=='tools/list': result={'tools':[{'name':'where','inputSchema':{'type':'object'}}]}\n"
        " elif method=='tools/call': result={'content':[{'type':'text','text':os.environ['WORKER_URL']}]}\n"
        " else: continue\n"
        " print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}),flush=True)\n",
        encoding="utf-8",
    )
    root = CompositionRoot("workload-composition")
    workloads = PluginWorkloads(root.instance_token)
    mcp = PluginMcpServers(root.instance_token)
    await root.context.provide(WORKLOADS, workloads)
    await root.context.provide(MCP_SERVERS, mcp)

    async def apply(ctx) -> None:
        await ctx.require(WORKLOADS).register(ctx, _workload())
        await ctx.require(MCP_SERVERS).register(
            ctx,
            McpServerDefinition(
                name="fixture",
                command=("python", "mcp.py"),
                required_tools=("where",),
                candidate_read_only_tools=("where",),
                workload_env=(WorkloadEnv("WORKER_URL", "worker", "gateway"),),
            ),
        )

    await root.mount(
        apply,
        name="fixture",
        inject=(WORKLOADS, MCP_SERVERS),
        runtime=_runtime(plugin_dir, data_dir, workspace),
    )
    generation = PluginGeneration(
        plugin_id="fixture",
        generation_id="fixture:test",
        module_path="plugins.fixture",
        source_revision="source",
        config_revision="config",
        plugin_dir=plugin_dir,
        data_dir=data_dir,
        config=None,
        instance=object(),
        scope=PluginScope("fixture"),
        contributions=PluginContributions(manifest={}),
        gate_result=GateResult(
            gate_id="gate",
            plugin_id="fixture",
            candidate_revision="source",
            status="passed",
            checks=(),
        ),
        static_runtime_commands=(("mcp:fixture", (sys.executable, str(mcp_script))),),
    )
    snapshot = RuntimeSnapshotCompiler().compile(
        {"fixture": generation},
        composition_root=root,
    )
    snapshot.tool_registry = ToolRegistry(follow_runtime_snapshot=False)
    host = CompositionGenerationHost(
        workload_controller=controller,
        workspace_id="workspace-test",
    )
    try:
        runtime = await host.start(generation, snapshot, mode="candidate")
        assert runtime is not None and runtime.workloads is not None
        registry = host.attach_tools(snapshot.tool_registry, runtime)
        assert registry is not None
        tool = registry.get_tool("mcp_fixture__where")
        assert tool is not None
        assert await tool.execute() == endpoint
        assert controller.starts[0].data == (("state", "/data", True),)
    finally:
        await host.stop(generation.generation_id)
        await root.dispose()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert len(controller.stops) == 1


@pytest.mark.asyncio
async def test_cleanup_retry_only_stops_entries_that_are_still_owned(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("workload-cleanup-retry")
    workloads = PluginWorkloads(root.instance_token)
    await root.context.provide(WORKLOADS, workloads)
    plugin_dir = tmp_path / "fixture"
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    plugin_dir.mkdir()
    data_dir.mkdir()
    workspace.mkdir()

    async def apply(ctx) -> None:
        await ctx.require(WORKLOADS).register(ctx, _named_workload("first"))
        await ctx.require(WORKLOADS).register(ctx, _named_workload("second"))

    await root.mount(
        apply,
        name="fixture",
        inject=(WORKLOADS,),
        runtime=_runtime(plugin_dir, data_dir, workspace),
    )
    registry = _freeze_plugin_workloads(workloads, root.instance_token)
    controller = _StopOnceController("http://127.0.0.1:1", "first")
    host = WorkloadGenerationHost(
        controller,
        workspace_id="workspace-test",
        health_probe=lambda _url, _timeout: _ready(),
    )
    bindings = {
        binding.descriptor.name: binding
        for binding in registry.values()
        if binding.descriptor.owner == "fixture"
    }
    await host.start_generation("fixture:test", "fixture", bindings, mode="formal")

    with pytest.raises(BaseExceptionGroup):
        await host.stop_generation("fixture:test")
    assert [lease.workload for lease in controller.stops] == ["second", "first"]

    await host.retry_generation_cleanup("fixture:test")
    assert [lease.workload for lease in controller.stops] == [
        "second",
        "first",
        "first",
    ]
    await root.dispose()


@pytest.mark.asyncio
async def test_lost_start_response_is_recovered_and_stopped(tmp_path: Path) -> None:
    root = CompositionRoot("workload-lost-response")
    workloads = PluginWorkloads(root.instance_token)
    await root.context.provide(WORKLOADS, workloads)
    plugin_dir = tmp_path / "fixture"
    data_dir = tmp_path / "data"
    workspace = tmp_path / "workspace"
    plugin_dir.mkdir()
    data_dir.mkdir()
    workspace.mkdir()

    async def apply(ctx) -> None:
        await ctx.require(WORKLOADS).register(ctx, _workload())

    await root.mount(
        apply,
        name="fixture",
        inject=(WORKLOADS,),
        runtime=_runtime(plugin_dir, data_dir, workspace),
    )
    registry = _freeze_plugin_workloads(workloads, root.instance_token)
    binding = registry.owned("fixture", "worker")
    assert binding is not None
    controller = _LostStartResponseController("http://127.0.0.1:1")
    host = WorkloadGenerationHost(
        controller,
        workspace_id="workspace-test",
        health_probe=lambda _url, _timeout: _ready(),
    )

    with pytest.raises(WorkloadEffectUnknown):
        await host.start_generation(
            "fixture:test", "fixture", {"worker": binding}, mode="formal"
        )

    assert len(controller.starts) == 2
    assert len(controller.stops) == 1
    assert host.get("fixture:test") is None
    await root.dispose()


async def _ready() -> tuple[bool, str]:
    return True, "ready"


def _write_external_fixture(plugin_dir: Path, version: str) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import (WORKLOADS, Workload, WorkloadData, "
        "WorkloadHealth, WorkloadLimits, WorkloadPort)\n"
        "api_version = 3\n"
        "name = 'outside-box'\n"
        f"version = {version!r}\n"
        "inject = (WORKLOADS,)\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(WORKLOADS).register(ctx, Workload(\n"
        "        name='worker',\n"
        f"        image={_IMAGE!r},\n"
        "        command=('serve',),\n"
        "        ports=(WorkloadPort('gateway', 8080),),\n"
        "        data=(WorkloadData('state', '/data'),),\n"
        "        health=WorkloadHealth('gateway', '/health', 5.0),\n"
        "        limits=WorkloadLimits(128, 1.0, 64),\n"
        "    ))\n",
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'outside-box'\n"
        f"version = {version!r}\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        "[[workloads]]\n"
        "name = 'worker'\n"
        f"image = {_IMAGE!r}\n"
        "command = ['serve']\n\n"
        "[[workloads.ports]]\n"
        "name = 'gateway'\n"
        "number = 8080\n\n"
        "[[workloads.data]]\n"
        "name = 'state'\n"
        "target = '/data'\n"
        "writable = true\n\n"
        "[workloads.health]\n"
        "port = 'gateway'\n"
        "path = '/health'\n"
        "timeout_seconds = 5.0\n\n"
        "[workloads.limits]\n"
        "memory_mb = 128\n"
        "cpu_count = 1.0\n"
        "pids = 64\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_external_fixture_updates_and_stops_through_public_workload_api(
    tmp_path: Path,
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReadyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    controller = _FakeController(f"http://127.0.0.1:{server.server_port}")
    plugin_dir = tmp_path / "plugins" / "outside-box"
    _write_external_fixture(plugin_dir, "1.0.0")
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "plugin-home" / "cache",
        workload_controller=controller,
    )
    try:
        await manager.load_all()
        stable = manager.current_snapshot
        assert stable is not None and stable.workload_registry is not None
        assert stable.workload_registry.owned("outside-box", "worker") is not None

        _write_external_fixture(plugin_dir, "1.0.1")
        candidate = await manager.prepare_candidate("outside-box")
        assert candidate is not None and candidate.runtime_snapshot is not None
        result = await manager.publish_prepared("outside-box")
        assert result["publication_state"] == "committed"
        assert any(request.mode == "candidate" for request in controller.starts)
        assert manager.current_snapshot is not stable
    finally:
        await manager.terminate_all()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert controller.stops
    assert len(controller.stops) == len(controller.starts)

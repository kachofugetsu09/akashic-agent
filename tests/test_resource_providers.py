"""普通资源 provider 的授权与取得责任；测试只由明确验证命令执行。"""
import asyncio
from dataclasses import replace
from pathlib import Path
import sys

import pytest

from agent.host_bridge.plugin_execution import CodeOwner, ControllerAccess, ExecutionAccess, spawn_process
from agent.plugin_composition import CompositionRoot, PluginRuntime, ServiceKey
from agent.plugin_composition.execution import EXECUTION, WORKLOAD_CONTROLLER
from agent.plugin_composition.mcp_slots import MCP_SERVERS, McpServerDefinition, WorkloadEnv
from agent.plugin_composition.workload_slots import WORKLOADS, Workload, WorkloadPort, WorkloadData, WorkloadHealth, WorkloadLimits
from agent.workloads.client import WorkloadEffectUnknown
from plugins.workloads import plugin as workloads_plugin
from plugins.mcp import plugin as mcp_plugin
from tests.test_workload_borrow import Controller


def definition():
    return Workload("desktop", "example/desktop@sha256:" + "a" * 64,
        ("/start",), (WorkloadPort("gateway", 8080),), (WorkloadData("state", "/data"),),
        WorkloadHealth("gateway"), WorkloadLimits(0, 0, 0))


async def root_with_grants(tmp_path, controller, *, candidate=True):
    root = CompositionRoot("resource-test")
    execution = ExecutionAccess(root.instance_token, {
        owner: CodeOwner(owner + "-code", tmp_path,
            lambda command, cwd: (sys.executable, str(tmp_path / command[0]), *command[1:]))
        for owner in ("consumer", "other")
    }, candidate=candidate)
    await root.context.provide(EXECUTION, execution)
    await root.context.provide(WORKLOAD_CONTROLLER, ControllerAccess(execution, controller, "workspace"))
    # Provider 名称不参与 Core dispatch，依赖仅由窄 service key 解析。
    await root.mount(workloads_plugin.apply, name="independent-controller-provider", inject=workloads_plugin.inject)
    await root.mount(mcp_plugin.apply, name="independent-protocol-provider", inject=mcp_plugin.inject)
    return root


def runtime(tmp_path, owner="consumer"):
    data = tmp_path / owner
    data.mkdir(exist_ok=True)
    return PluginRuntime(owner, owner + "-code", tmp_path, data, tmp_path, {})


@pytest.mark.asyncio
async def test_apply_owns_workload_before_controller_await_and_uses_real_reference(tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    contexts, handles = [], []
    class DelayedController(Controller):
        async def start(self, request):
            assert any(effect.label == "workload:desktop" for effect in contexts[0]._fiber.effects)
            entered.set()
            await release.wait()
            return await super().start(request)
    async def healthy(*args):
        return True, "ready"
    monkeypatch.setattr("plugins.workloads.host._http_health", healthy)
    (tmp_path / "server.py").write_text("pass\n")
    controller = DelayedController()
    root = await root_with_grants(tmp_path, controller)
    async def apply(ctx):
        contexts.append(ctx)
        handle = await ctx.require(WORKLOADS).register(ctx, definition())
        handles.append(handle)
        await ctx.require(MCP_SERVERS).register(ctx, McpServerDefinition(
            "desktop", ("server.py",), workload_env=(WorkloadEnv("DESKTOP_URL", handle, "gateway"),)))
        await ctx.provide(ServiceKey("external.unlisted.capability"), lambda: handle.url(ctx, "gateway"))
    task = asyncio.create_task(root.mount(apply, name="consumer", inject=(WORKLOADS, MCP_SERVERS), runtime=runtime(tmp_path)))
    await entered.wait()
    assert not task.done()
    release.set()
    try:
        await task
        assert root.receipt().ready
        assert root.context.require(ServiceKey("external.unlisted.capability"))() == "http://container-A:8080"
        assert controller.started[0].mode == "candidate"
        entry = root.context.require(MCP_SERVERS)._entries["desktop"]
        assert entry.definition.workload_env[0].workload is handles[0]
        async def wrong_owner(ctx):
            with pytest.raises(PermissionError):
                handles[0].url(ctx, "gateway")
            grant = ctx.require(WORKLOAD_CONTROLLER).bind(ctx)
            with pytest.raises(PermissionError):
                await grant.start(replace(controller.started[0], mode="formal"))
        await root.mount(wrong_owner, name="other", inject=(WORKLOADS,), runtime=runtime(tmp_path, "other"))
    finally:
        await root.dispose()
    assert len(controller.started) == len(controller.stopped) == 1


@pytest.mark.asyncio
async def test_unknown_workload_result_remains_in_provider_and_scope(tmp_path):
    class UnknownController(Controller):
        async def start(self, request):
            await super().start(request)
            raise WorkloadEffectUnknown("lost receipt")
    controller = UnknownController()
    root = await root_with_grants(tmp_path, controller)
    contexts = []
    async def apply(ctx):
        contexts.append(ctx)
        await ctx.require(WORKLOADS).register(ctx, definition())
    with pytest.raises(BaseExceptionGroup):
        await root.mount(apply, name="consumer", inject=(WORKLOADS,), runtime=runtime(tmp_path))
    service = contexts[0].require(WORKLOADS)
    handle = service._entries[("consumer", "desktop")]
    assert any(effect.label == "workload:desktop" for effect in contexts[0]._fiber.effects)
    request = controller.started[0]
    for _ in range(2):
        with pytest.raises(BaseExceptionGroup):
            await root.dispose()
        assert handle._host._generations[handle._grant.identity].pending["desktop"][1] is request
        assert controller.started == [request]
        assert controller.stopped == []


@pytest.mark.asyncio
async def test_candidate_execution_never_inherits_formal_environment_or_other_root(tmp_path, monkeypatch):
    monkeypatch.setenv("UNRELATED_HOST_SECRET", "formal-secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/formal/agent.sock")
    root = await root_with_grants(tmp_path, Controller())
    other = CompositionRoot("other")
    grants = []
    async def apply(ctx):
        grant = ctx.require(EXECUTION).bind(ctx)
        grants.append(grant)
        env = grant.environment({"FORMAL_TOKEN": "formal"}, {"VALIDATION_TOKEN": "isolated"})
        assert "UNRELATED_HOST_SECRET" not in env
        assert "SSH_AUTH_SOCK" not in env
        assert "FORMAL_TOKEN" not in env
        assert env["VALIDATION_TOKEN"] == "isolated"
        assert env["HOME"] == str(ctx.runtime.data_dir)
        await other.context.provide(EXECUTION, ctx.require(EXECUTION))
    await root.mount(apply, name="consumer", runtime=runtime(tmp_path))
    async def wrong_root(ctx):
        with pytest.raises(PermissionError, match="Root"):
            ctx.require(EXECUTION).bind(ctx)
    try:
        await other.mount(wrong_root, name="consumer", runtime=runtime(tmp_path))
    finally:
        await other.dispose()
        await root.dispose()


@pytest.mark.asyncio
async def test_spawn_cancellation_delivers_the_owned_process_receipt(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    process = object()
    async def spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        return process
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(spawn_process("test"))
    await entered.wait()
    task.cancel()
    release.set()
    receipt, cancelled = await task
    assert receipt is process and cancelled


@pytest.mark.asyncio
async def test_managed_process_provider_waits_for_borrow_before_stop(tmp_path, monkeypatch):
    from plugins.managed_processes import plugin as process_plugin
    from agent.plugin_composition.process_slots import MANAGED_PROCESSES, ManagedProcessDefinition
    (tmp_path / "server.py").write_text('''
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass
HTTPServer(("127.0.0.1", int(os.environ["PORT"])), Handler).serve_forever()
''')
    root = await root_with_grants(tmp_path, Controller())
    await root.mount(process_plugin.apply, name="independent-process-provider", inject=process_plugin.inject)
    handles, contexts = [], []
    async def apply(ctx):
        contexts.append(ctx)
        handles.append(await ctx.require(MANAGED_PROCESSES).register(ctx,
            ManagedProcessDefinition("server", ("server.py",))))
    try:
        await root.mount(apply, name="consumer", inject=(MANAGED_PROCESSES,), runtime=runtime(tmp_path))
        handle, ctx = handles[0], contexts[0]
        current = handle._host._generations[handle._id].entries["server"].process
        assert current.returncode is None
        effect = next(item for item in ctx._fiber.effects if item.label == "process:server")
        waiting = asyncio.Event()
        wait_for_borrowers = handle._drained.wait
        async def observe_drain():
            waiting.set()
            await wait_for_borrowers()
        monkeypatch.setattr(handle._drained, "wait", observe_drain)
        async with handle.borrow(ctx) as port:
            assert port > 0
            closing = asyncio.create_task(effect.aclose())
            await waiting.wait()
            assert not closing.done()
            assert current.returncode is None
            with pytest.raises(PermissionError):
                async with handle.borrow(ctx):
                    pytest.fail("关闭中的 owner 不得接纳新借用")
        await closing
        assert effect not in ctx._fiber.effects
        assert current.returncode is not None
        with pytest.raises(PermissionError):
            handle.port(ctx)
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_missing_resource_provider_is_an_ordinary_missing_dependency():
    root = CompositionRoot("no-hidden-provider")
    called = False
    async def consumer(ctx):
        nonlocal called
        called = True
    try:
        await root.mount(consumer, name="consumer", inject=(WORKLOADS,))
        assert not root.receipt().ready
        assert not called
        assert root.context.get(WORKLOADS) is None
    finally:
        await root.dispose()

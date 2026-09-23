import asyncio
import ast
import json
import shutil
import sys
from pathlib import Path

from agent.plugin_composition.mcp_slots import MCP_SERVERS

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.model import CompositionError, FiberState, ServiceKey
from session.log import MessageLog
from tests.test_plugin_bindings import manager

SERVICE = ServiceKey("test.bound.mcp")


def _write_source(path, source):
    """Parse and compile fixture source in memory before writing it."""
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    path.write_text(source)


def write_plugin(path):
    path.mkdir(parents=True)
    (path / "first").mkdir()
    (path / "first" / "requirements.txt").write_text("")
    (path / "second").mkdir()
    (path / "second" / "requirements.txt").write_text("")
    _write_source(path / "plugin.py", """
from agent.plugin_composition import MCP_SERVERS, McpServerDefinition, ServiceKey
api_version = 3
name = "probe"
version = "1.0.0"
inject = (MCP_SERVERS,)
async def apply(ctx):
    service = ctx.require(MCP_SERVERS)
    for name in ("first", "second"):
        await service.register(ctx, McpServerDefinition(
            name=name, command=("python", "first/server.py" if name == "first" else "second/server.py"), env={"SERVER": name}, candidate_env={"SERVER": name},
            required_tools=("ping",), candidate_read_only_tools=("ping",),
        ))
    await ctx.provide(ServiceKey("test.bound.mcp"), lambda: service.open(ctx, "first"))
    await ctx.provide(ServiceKey("test.bound.text"), "fixed text")
""")
    _write_source(path / "server.py", """
import json, os, sys
from pathlib import Path
count = Path(os.environ["AKA_PLUGIN_DATA_DIR"]) / (os.environ["SERVER"] + ".count")
count.write_text(str(int(count.read_text()) + 1 if count.exists() else 1))
count.with_suffix(".boot").write_text(json.dumps({key: os.environ.get(key) for key in ("AKASHIC_BOOT_ID", "AKASHIC_SUPERVISED")}))
for raw in sys.stdin:
    request = json.loads(raw)
    method = request.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": "probe", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "ping", "description": "fixed A", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "fixed A"}]}
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
""")
    shutil.copy2(path / "server.py", path / "second" / "server.py")
    shutil.copy2(path / "server.py", path / "first" / "server.py")


def select_mcp_provider(folder):
    """安装 fixture 显式选择普通 provider；Manager 不补隐藏依赖。"""
    target = folder / "mcp"
    shutil.copytree(Path(__file__).parents[1] / "plugins/mcp", target,
                   ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


@pytest.mark.asyncio
async def test_mcp_is_opened_per_call_and_route_expires(tmp_path):
    plugins = tmp_path / "plugins"
    write_plugin(plugins / "probe")
    initialize_plugin_workspace(tmp_path / "workspace")
    select_mcp_provider(plugins)
    owner = manager(tmp_path, [plugins])
    try:
        await owner.load_all()
        root = owner.live_root
        assert root is not None
        data = root.plugin_runtime("probe").data_dir
        assert not (data / "first.count").exists()
        identities = []
        for _ in range(2):
            open_server = root.service_value(SERVICE)
            assert open_server is not None
            async with open_server() as server:
                identities.append(server.generation_id)
                route = server.route()
                assert (await route.call("ping", {})).output == "fixed A"
            with pytest.raises(RuntimeError):
                await route.call("ping", {})
        assert identities[0] != identities[1]
        assert (data / "first.count").read_text() == "2"
        assert not (data / "second.count").exists()
        assert root.context.require(MCP_SERVERS).failures() == ()
        assert root.receipt().ready
    finally:
        await owner.terminate_all()


@pytest.mark.asyncio
async def test_missing_environment_does_not_create_a_session_or_lose_effect(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    write_plugin(plugins / "probe")
    select_mcp_provider(plugins)
    initialize_plugin_workspace(tmp_path / "workspace")
    owner = manager(tmp_path, [plugins])
    def missing(*args):
        raise FileNotFoundError("fixed environment missing")
    monkeypatch.setattr(owner, "_resolve_runtime_command", missing)
    try:
        await owner.load_all()
        root = owner.live_root
        assert root is not None
        service = root.context.require(MCP_SERVERS)
        open_server = root.service_value(SERVICE)
        assert open_server is not None
        with pytest.raises(FileNotFoundError, match="fixed environment"):
            async with open_server():
                pytest.fail("missing artifact started")
        assert service._sessions == {}
        assert service.failures() == ()
    finally:
        await owner.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["retry", "shutdown", "concurrent"])
async def test_scoped_cleanup_failure_retains_real_owner(tmp_path, monkeypatch, cleanup):
    plugins = tmp_path / "plugins"
    write_plugin(plugins / "probe")
    initialize_plugin_workspace(tmp_path / "workspace")
    select_mcp_provider(plugins)
    owner = manager(tmp_path, [plugins])
    original = None
    failed = False
    process = None
    async def fail_once(host, entry):
        nonlocal failed, process
        if not failed:
            failed = True
            process = entry.client._process
            raise OSError("injected disconnect failure")
        await original(host, entry)
    try:
        await owner.load_all()
        root = owner.live_root
        assert root is not None
        service = root.context.require(MCP_SERVERS)
        data = root.plugin_runtime("probe").data_dir
        host_class = sys.modules[type(service).__module__].McpGenerationHost
        original = host_class._cleanup_entry
        monkeypatch.setattr(host_class, "_cleanup_entry", fail_once)
        with pytest.raises(RuntimeError, match="injected disconnect failure"):
            open_server = root.service_value(SERVICE)
            assert open_server is not None
            async with open_server() as server:
                identity = server.generation_id
                route = server.route()
                assert (await route.call("ping", {})).output == "fixed A"
        assert service.failures()[0].identity == identity
        assert process.returncode is None
        ctx = service._entries["first"].ctx
        retained = service._sessions[identity]
        effect = retained._effect
        assert effect in ctx._fiber.effects
        starts = (data / "first.count").read_text()
        if cleanup == "retry":
            await service.retry_cleanup(ctx, identity)
        elif cleanup == "shutdown":
            await owner.terminate_all()
        else:
            await asyncio.gather(service.retry_cleanup(ctx, identity), owner.terminate_all())
        assert process.returncode is not None
        assert (data / "first.count").read_text() == starts
        assert effect not in ctx._fiber.effects
        assert service.failures() == ()
        assert service._sessions == {}
    finally:
        await owner.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [False, True], ids=["manager", "local"])
async def test_shutdown_waits_for_an_owned_start(tmp_path, monkeypatch, local):
    """Real MCP connect drain plus the narrow local contribution branch."""
    from agent.plugin_composition import RUNTIME_STARTED, RUNTIME_STOPPING
    from agent.mcp.client import McpClient

    connect_entered, connect_release = asyncio.Event(), asyncio.Event()
    open_started, open_release = asyncio.Event(), asyncio.Event()
    hard_consumer_cleanup_started = asyncio.Event()
    hard_consumer_release = asyncio.Event()
    original_connect = McpClient.connect

    async def gated_connect(client):
        connect_entered.set()
        await connect_release.wait()
        return await original_connect(client)

    monkeypatch.setattr(McpClient, "connect", gated_connect)
    task = dispose_task = shutdown = rejected = None
    root = None
    owner = None
    service = None
    probe_context = None
    peer_context = None
    peer_fiber = None
    hard_consumer = None
    peer_service = ServiceKey("test.mcp.unrelated.peer")
    peer_state = {"started": 0, "stopping": 0, "effect": 0, "cleanup": 0}
    try:
        plugins = tmp_path / "plugins"
        write_plugin(plugins / "probe")
        initialize_plugin_workspace(tmp_path / "workspace")
        select_mcp_provider(plugins)
        owner = manager(tmp_path, [plugins])
        await owner.load_all()
        root = owner.live_root
        assert root is not None
        service = root.context.require(MCP_SERVERS)
        probe_fiber = next(
            fiber for fiber in root.root_fiber.children
            if fiber.runtime is not None and fiber.runtime.plugin_id == "probe"
        )
        mcp_fiber = next(
            fiber for fiber in root.root_fiber.children
            if fiber.runtime is not None and fiber.runtime.plugin_id == "mcp"
        )
        probe_context, _ = root._service_provider(SERVICE)
        open_server = root.service_value(SERVICE)
        assert open_server is not None

        async def peer_apply(ctx):
            nonlocal peer_context
            peer_context = ctx

            def setup():
                peer_state["effect"] += 1

                def cleanup():
                    peer_state["cleanup"] += 1

                return cleanup

            async def started(_event):
                peer_state["started"] += 1

            async def stopping(_event):
                peer_state["stopping"] += 1

            await ctx.effect(setup, label="mcp-unrelated-peer")
            await ctx.on(RUNTIME_STARTED, started)
            await ctx.on(RUNTIME_STOPPING, stopping)
            await ctx.provide(peer_service, "peer-service")

        async def hard_consumer_apply(ctx):
            _ = ctx.require(SERVICE)

            def setup():
                async def cleanup():
                    hard_consumer_cleanup_started.set()
                    await hard_consumer_release.wait()

                return cleanup

            await ctx.effect(setup, label="mcp-hard-consumer")

        peer_fiber = await root.mount(peer_apply, name="unrelated-peer")
        hard_consumer = await root.mount(
            hard_consumer_apply,
            name="mcp-hard-consumer",
            inject=(SERVICE,),
        )
        assert peer_context is not None
        assert peer_fiber is not None
        peer_raw_fiber = peer_context._fiber
        assert peer_fiber.context is peer_context
        peer_activation = peer_context.fiber.activation_token
        peer_effects = tuple(peer_context._fiber.effects)
        peer_counts = peer_state.copy()

        async def use():
            async with open_server() as _server:
                open_started.set()
                await open_release.wait()

        async with asyncio.timeout(30):
            task = asyncio.create_task(use())
            await connect_entered.wait()
            assert probe_context is not None and peer_fiber is not None and hard_consumer is not None
            assert len(service._sessions) == 1
            retained = next(iter(service._sessions.values()))
            session_effect = retained._effect
            session_count = len(service._sessions)
            assert session_effect in probe_context._fiber.effects
            if local:
                dispose_task = asyncio.create_task(probe_fiber.dispose())
            else:
                shutdown = asyncio.create_task(owner.terminate_all())
            await hard_consumer_cleanup_started.wait()
            assert probe_fiber.state is FiberState.UNLOADING
            assert hard_consumer.state is FiberState.UNLOADING
            assert not connect_release.is_set()
            assert not open_started.is_set()
            assert not (dispose_task if local else shutdown).done()
            assert session_effect in probe_context._fiber.effects
            assert probe_context._fiber._in_flight_calls

            if local:
                async with peer_context.runtime_scope():
                    assert peer_context.require(peer_service) == "peer-service"
                assert not peer_raw_fiber._in_flight_calls
                assert peer_fiber.context is peer_context
                assert peer_context._fiber is peer_raw_fiber
                assert peer_fiber.state is FiberState.ACTIVE
                assert peer_context.fiber.activation_token is peer_activation
                assert tuple(peer_context._fiber.effects) == peer_effects
                assert peer_state == peer_counts

            body_executed = asyncio.Event()

            async def rejected_open():
                with pytest.raises(CompositionError) as error:
                    async with open_server() as _server:
                        body_executed.set()
                assert error.value.code == "OWNER_UNAVAILABLE"

            rejected = asyncio.create_task(rejected_open())
            await rejected
            assert not body_executed.is_set()
            assert len(service._sessions) == session_count

            connect_release.set()
            await open_started.wait()
            assert not task.done()
            open_release.set()
            await task
            assert service._sessions == {}
            assert session_effect not in probe_context._fiber.effects
            assert not probe_context._fiber._in_flight_calls
            hard_consumer_release.set()
            if local:
                await dispose_task
                assert owner.live_root is root
                assert probe_fiber.state is FiberState.DISPOSED
                assert hard_consumer.state is FiberState.PENDING
                async with peer_context.runtime_scope():
                    assert peer_context.require(peer_service) == "peer-service"
                assert not peer_raw_fiber._in_flight_calls
                assert root.context.require(MCP_SERVERS) is service
                assert mcp_fiber.state is FiberState.ACTIVE
                assert peer_fiber.context is peer_context
                assert peer_context._fiber is peer_raw_fiber
                assert peer_fiber.state is FiberState.ACTIVE
                assert peer_context.fiber.activation_token is peer_activation
                assert tuple(peer_context._fiber.effects) == peer_effects
                assert peer_state == peer_counts
            else:
                await shutdown
    finally:
        connect_release.set()
        open_release.set()
        hard_consumer_release.set()
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        for cleanup_task in (task, rejected):
            if cleanup_task is not None and not cleanup_task.done():
                cleanup_task.cancel()
        for cleanup_task in (task, rejected, dispose_task, shutdown):
            if cleanup_task is not None:
                try:
                    await cleanup_task
                except asyncio.CancelledError:
                    pass
                except BaseException as error:
                    cleanup_errors.append(error)
        if owner is not None:
            try:
                await owner.terminate_all()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            errors = cleanup_errors if primary_error is None else [primary_error, *cleanup_errors]
            raise BaseExceptionGroup("MCP drain test cleanup failed", errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["AKASHIC_BOOT_ID", "AKASHIC_SUPERVISED"])
async def test_mcp_rejects_supervisor_identity_override(tmp_path, key):
    from agent.plugin_composition import CompositionRoot, McpServerDefinition, PluginRuntime
    from plugins.mcp.plugin import McpServers
    root = CompositionRoot("reserved-environment")
    service = McpServers(root.context)
    await root.context.provide(MCP_SERVERS, service)
    async def apply(ctx):
        with pytest.raises(ValueError, match=key):
            await service.register(ctx, McpServerDefinition(name="invalid", command=("python",), candidate_env={key: "fake"}))
    try:
        await root.mount(apply, name="probe", inject=(MCP_SERVERS,), runtime=PluginRuntime("probe", "test", tmp_path, tmp_path, tmp_path, {}))
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_isolated_host_grant_and_allowlist_are_host_bound(tmp_path, monkeypatch):
    from agent.host_bridge.plugin_execution import CodeOwner, ExecutionAccess
    from agent.plugin_composition import CompositionRoot, McpServerDefinition, PluginRuntime
    from agent.plugin_composition.execution import EXECUTION
    from plugins.mcp import plugin as mcp_plugin

    source = tmp_path / "source" / "probe"
    first = source / "first"
    first.mkdir(parents=True)
    data = tmp_path / "isolated-data"
    workspace = tmp_path / "isolated-workspace"
    data.mkdir()
    workspace.mkdir()
    _write_source(first / "server.py", """
import json, os, sys
from pathlib import Path
count = Path(os.environ["AKA_PLUGIN_DATA_DIR"]) / "first.count"
count.write_text(str(int(count.read_text()) + 1 if count.exists() else 1))
for raw in sys.stdin:
    request = json.loads(raw)
    method = request.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": "probe", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": name, "description": "probe", "inputSchema": {"type": "object"}} for name in ("ping", "mutate")]}
    elif method == "tools/call":
        if request["params"]["name"] == "mutate":
            count.with_suffix(".mutated").write_text("changed")
        environment = {key: os.environ.get(key) for key in ("VALIDATION_MARK", "FORMAL_TOKEN", "UNRELATED_HOST_SECRET", "HOME", "AKA_PLUGIN_DATA_DIR", "AKASHIC_WORKSPACE")}
        result = {"content": [{"type": "text", "text": json.dumps(environment)}]}
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
""")
    generation_id = "isolated-probe"
    root = CompositionRoot("isolated-host-grant")

    def resolve(command, cwd):
        assert command == ("first/server.py",)
        assert cwd == "."
        return (sys.executable, str(source / command[0]))

    async def apply(ctx):
        service = ctx.require(MCP_SERVERS)
        await service.register(ctx, McpServerDefinition(
            name="first",
            command=("first/server.py",),
            env={"SERVER": "first", "FORMAL_TOKEN": "formal-secret"},
            candidate_env={"SERVER": "first", "VALIDATION_MARK": "candidate"},
            required_tools=("ping",),
            candidate_read_only_tools=("ping",),
        ))
        await ctx.provide(SERVICE, lambda: service.open(ctx, "first"))

    monkeypatch.setenv("UNRELATED_HOST_SECRET", "must-not-inherit")
    try:
        execution = ExecutionAccess(
            root.instance_token,
            {("probe", generation_id): CodeOwner(generation_id, source, resolve)},
            candidate=True,
        )
        await root.context.provide(EXECUTION, execution)
        await root.mount(mcp_plugin.apply, name="mcp", inject=mcp_plugin.inject)
        probe = await root.mount(
            apply,
            name="probe",
            inject=(MCP_SERVERS,),
            runtime=PluginRuntime("probe", generation_id, source, data, workspace, {}),
        )
        assert probe.state is FiberState.ACTIVE
        open_server = root.service_value(SERVICE)
        assert open_server is not None
        async with open_server() as server:
            assert set(server.tools) == {"ping"}
            async with server.route() as route:
                environment = json.loads((await route.call("ping", {})).output)
                assert environment == {
                    "VALIDATION_MARK": "candidate", "FORMAL_TOKEN": None,
                    "UNRELATED_HOST_SECRET": None, "HOME": str(data),
                    "AKA_PLUGIN_DATA_DIR": str(data),
                    "AKASHIC_WORKSPACE": str(workspace),
                }
                with pytest.raises(PermissionError, match="allowlist"):
                    await route.call("mutate", {})
        assert not (data / "first.mutated").exists()
        assert not (workspace / "first.count").exists()
        assert not (tmp_path / "workspace" / "plugin-data" / "probe" / "first.count").exists()
    finally:
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_scoped_mcp_waits_for_eof_grace_and_process_group_cleanup(tmp_path, monkeypatch, cancel):
    """成功调用后，忽略 EOF 的真实进程仍完成 TERM 回收，取消不遗留资源。"""
    import plugins.mcp.client as client_module
    from utils.process_group import process_group_exists

    plugins = tmp_path / "plugins"
    write_plugin(plugins / "probe")
    select_mcp_provider(plugins)
    script = plugins / "probe/first/server.py"
    _write_source(script, script.read_text().replace(
        "for raw in sys.stdin:", "own_count = int(count.read_text())\nfor raw in sys.stdin:"
    ) + '''
if own_count > 0:
    import signal
    count.with_suffix(".eof-pid").write_text(str(os.getpid()))
    signal.pause()
''')
    initialize_plugin_workspace(tmp_path / "workspace")
    waiting_for_exit = asyncio.Event()
    original_wait = client_module._wait_for_leader_exit
    log = None
    owner = None
    root = None
    task = None

    async def wait_for_exit(process):
        waiting_for_exit.set()
        return await original_wait(process)

    monkeypatch.setattr(client_module, "_wait_for_leader_exit", wait_for_exit)
    try:
        log = MessageLog(tmp_path / "messages.db")
        owner = manager(tmp_path, [plugins], message_log=log)
        # 1. 不预启动 MCP；本次调用的进程在 EOF 后继续存活。
        await owner.load_all()
        root = owner.live_root
        assert root is not None
        data = root.plugin_runtime("probe").data_dir
        bindings = root.context.require(BINDINGS)
        provider_context, _ = root._service_provider(SERVICE)
        async with provider_context.runtime_scope():
            identity = bindings.bind(SERVICE, {})

        async def call():
            async with bindings.open(identity, SERVICE) as (open_server, metadata):
                assert metadata == {}
                async with open_server() as server:
                    async with server.route() as route:
                        assert (await route.call("ping", {})).output == "fixed A"

        # 2. 在真实 EOF 清理阶段取消；正常和取消路径都必须等进程组消失。
        task = asyncio.create_task(call())
        await asyncio.wait_for(waiting_for_exit.wait(), 10)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert not process_group_exists(int((data / "first.eof-pid").read_text()))
        assert root.context.require(MCP_SERVERS).failures() == ()
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        try:
            if task is not None:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except BaseException as error:
                    cleanup_errors.append(error)
            if owner is not None:
                try:
                    await owner.terminate_all()
                except BaseException as error:
                    cleanup_errors.append(error)
        finally:
            if log is not None:
                try:
                    log.close()
                except BaseException as error:
                    cleanup_errors.append(error)
        if cleanup_errors:
            errors = cleanup_errors if primary_error is None else [primary_error, *cleanup_errors]
            raise BaseExceptionGroup("MCP EOF test cleanup failed", errors)

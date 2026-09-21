import asyncio
import json
import shutil
import sys
from pathlib import Path

from agent.plugin_composition.mcp_slots import MCP_SERVERS
from agent.plugin_composition.context import RuntimeScope

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.model import ServiceKey
from agent.plugins.snapshot import get_current_runtime_lease, lease_runtime_snapshot
from session.log import MessageLog
from tests.test_plugin_bindings import manager

SERVICE = ServiceKey("test.bound.mcp")


def write_plugin(path):
    path.mkdir(parents=True)
    (path / "first").mkdir()
    (path / "first" / "requirements.txt").write_text("")
    (path / "second").mkdir()
    (path / "second" / "requirements.txt").write_text("")
    (path / "plugin.py").write_text("""
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
    (path / "server.py").write_text("""
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
        snapshot = owner.current_snapshot
        root = snapshot.composition_root
        data = snapshot.generations["probe"].data_dir
        assert not (data / "first.count").exists()
        identities = []
        for _ in range(2):
            async with lease_runtime_snapshot(owner.snapshot_store):
                async with root.service_value(SERVICE)() as server:
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
        root = owner.current_snapshot.composition_root
        service = root.context.require(MCP_SERVERS)
        async with lease_runtime_snapshot(owner.snapshot_store):
            with pytest.raises(FileNotFoundError, match="fixed environment"):
                async with root.service_value(SERVICE)():
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
        snapshot = owner.current_snapshot
        root = snapshot.composition_root
        service = root.context.require(MCP_SERVERS)
        host_class = sys.modules[type(service).__module__].McpGenerationHost
        original = host_class._cleanup_entry
        monkeypatch.setattr(host_class, "_cleanup_entry", fail_once)
        with pytest.raises(RuntimeError, match="injected disconnect failure"):
            async with lease_runtime_snapshot(owner.snapshot_store):
                async with root.service_value(SERVICE)() as server:
                    identity = server.generation_id
                    route = server.route()
                    assert (await route.call("ping", {})).output == "fixed A"
        assert service.failures()[0].identity == identity
        assert process.returncode is None
        ctx = service._entries["first"].ctx
        retained = service._sessions[identity]
        effect = retained._effect
        assert effect in ctx._fiber.effects
        starts = (snapshot.generations["probe"].data_dir / "first.count").read_text()
        if cleanup == "retry":
            await service.retry_cleanup(ctx, identity)
        elif cleanup == "shutdown":
            await owner.terminate_all()
        else:
            await asyncio.gather(service.retry_cleanup(ctx, identity), owner.terminate_all())
        assert process.returncode is not None
        assert (snapshot.generations["probe"].data_dir / "first.count").read_text() == starts
        assert effect not in ctx._fiber.effects
        assert service.failures() == ()
        assert service._sessions == {}
    finally:
        await owner.terminate_all()


@pytest.mark.asyncio
async def test_shutdown_waits_for_an_owned_start(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    write_plugin(plugins / "probe")
    initialize_plugin_workspace(tmp_path / "workspace")
    select_mcp_provider(plugins)
    owner = manager(tmp_path, [plugins])
    entered, release = asyncio.Event(), asyncio.Event()
    original = None
    async def delayed(host, *args, **kwargs):
        entered.set()
        await release.wait()
        return await original(host, *args, **kwargs)
    task = shutdown = None
    try:
        await owner.load_all()
        root = owner.current_snapshot.composition_root
        service = root.context.require(MCP_SERVERS)
        host_class = sys.modules[type(service).__module__].McpGenerationHost
        original = host_class.start_generation
        monkeypatch.setattr(host_class, "start_generation", delayed)
        async def use():
            async with root.service_value(SERVICE)():
                pass
        task = asyncio.create_task(use())
        await entered.wait()
        assert len(service._sessions) == 1
        retained = next(iter(service._sessions.values()))
        assert retained._effect in retained._entry.ctx._fiber.effects
        shutdown = asyncio.create_task(owner.terminate_all())
        release.set()
        await task
        await shutdown
        assert service._sessions == {}
    finally:
        release.set()
        await asyncio.gather(*(item for item in (task, shutdown) if item is not None), return_exceptions=True)
        await owner.terminate_all()


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
async def test_candidate_environment_and_allowlist_are_host_bound(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    source = plugins / "probe"
    write_plugin(source)
    select_mcp_provider(plugins)
    path = source / "plugin.py"
    path.write_text(path.read_text().replace(
        'env={"SERVER": name}, candidate_env={"SERVER": name}',
        'env={"SERVER": name, "FORMAL_TOKEN": "formal-secret"}, candidate_env={"SERVER": name, "VALIDATION_MARK": "candidate"}',
    ))
    for name in ("first", "second"):
        path = source / name / "server.py"
        path.write_text(path.read_text().replace(
            '[{"name": "ping", "description": "fixed A", "inputSchema": {"type": "object"}}]',
            '[{"name": name, "description": "probe", "inputSchema": {"type": "object"}} for name in ("ping", "mutate")]',
        ).replace(
            'result = {"content": [{"type": "text", "text": "fixed A"}]}',
            'if request["params"]["name"] == "mutate":\n'
            '            count.with_suffix(".mutated").write_text("changed")\n'
            '        result = {"content": [{"type": "text", "text": json.dumps({key: os.environ.get(key) for key in '
            '("VALIDATION_MARK", "FORMAL_TOKEN", "UNRELATED_HOST_SECRET", "HOME", "AKA_PLUGIN_DATA_DIR", "AKASHIC_WORKSPACE")})}]}',
        ))
    initialize_plugin_workspace(tmp_path / "workspace")
    owner = manager(tmp_path, [plugins])
    monkeypatch.setenv("UNRELATED_HOST_SECRET", "must-not-inherit")
    try:
        await owner.load_all()
        candidate = await owner.prepare_candidate("probe")
        snapshot = candidate.runtime_snapshot
        transaction = owner._begin_snapshot_publication(snapshot)
        await owner.snapshot_store.commit_latest(transaction)
        root = snapshot.composition_root
        lease = owner.snapshot_store.lease(selector="latest")
        async with RuntimeScope(lease):
            assert lease.snapshot is snapshot
            async with root.service_value(SERVICE)() as server:
                current = get_current_runtime_lease()
                assert current is not lease and current.snapshot is snapshot
                assert current.active
                assert set(server.tools) == {"ping"}
                async with server.route() as route:
                    environment = json.loads((await route.call("ping", {})).output)
                    assert environment == {
                        "VALIDATION_MARK": "candidate", "FORMAL_TOKEN": None,
                        "UNRELATED_HOST_SECRET": None, "HOME": str(candidate.data_dir),
                        "AKA_PLUGIN_DATA_DIR": str(candidate.data_dir),
                        "AKASHIC_WORKSPACE": str(candidate.validation_workspace),
                    }
                    with pytest.raises(PermissionError, match="allowlist"):
                        await route.call("mutate", {})
            assert not current.active
        assert not (candidate.data_dir / "first.mutated").exists()
        assert not (tmp_path / "workspace" / "plugin-data" / "probe" / "first.count").exists()
    finally:
        await owner.terminate_all()


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
    script.write_text(script.read_text().replace(
        "for raw in sys.stdin:", "own_count = int(count.read_text())\nfor raw in sys.stdin:"
    ) + '''
if own_count > 0:
    import signal
    count.with_suffix(".eof-pid").write_text(str(os.getpid()))
    signal.pause()
''')
    initialize_plugin_workspace(tmp_path / "workspace")
    owner = manager(tmp_path, [plugins])
    log = MessageLog(tmp_path / "messages.db")
    waiting_for_exit = asyncio.Event()
    original_wait = client_module._wait_for_leader_exit

    async def wait_for_exit(process):
        waiting_for_exit.set()
        return await original_wait(process)

    monkeypatch.setattr(client_module, "_wait_for_leader_exit", wait_for_exit)
    try:
        # 1. 不预启动 MCP；本次调用的进程在 EOF 后继续存活。
        await owner.load_all()
        snapshot = owner.current_snapshot
        data = snapshot.generations["probe"].data_dir
        assert snapshot.composition_root is not None
        bindings = Bindings(log, owner._archive, snapshot.composition_root)
        async with lease_runtime_snapshot(owner.snapshot_store):
            identity = bindings.bind(SERVICE, {})

        async def call():
            async with bindings.open(identity, SERVICE) as (open_server, _):
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
        assert snapshot.composition_root.context.require(MCP_SERVERS).failures() == ()
        assert owner.current_snapshot is snapshot
    finally:
        await owner.terminate_all()
        log.close()

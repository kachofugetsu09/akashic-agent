import asyncio
import shutil

import pytest

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.model import ServiceKey
from agent.plugins.snapshot import lease_runtime_snapshot
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
async def apply(ctx, config):
    service = ctx.require(MCP_SERVERS)
    for name in ("first", "second"):
        await service.register(ctx, McpServerDefinition(
            name=name, command=("python", "first/server.py" if name == "first" else "second/server.py"), env={"SERVER": name},
            required_tools=("ping",), candidate_read_only_tools=("ping",),
        ))
    await ctx.provide(ServiceKey("test.bound.mcp"), lambda: service.open(ctx, "first"))
    await ctx.provide(ServiceKey("test.bound.text"), "fixed text")
""")
    (path / "akashic.plugin.toml").write_text("""schema_version = 1
name = "probe"
version = "1.0.0"
api_version = 3
entrypoint = "plugin.py"
[[python]]
requirements = "first/requirements.txt"
[[python]]
requirements = "second/requirements.txt"
[[mcp]]
name = "first"
command = ["python", "first/server.py"]
env = {SERVER = "first"}
required_tools = ["ping"]
candidate_read_only_tools = ["ping"]
[[mcp]]
name = "second"
command = ["python", "second/server.py"]
env = {SERVER = "second"}
required_tools = ["ping"]
candidate_read_only_tools = ["ping"]
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


@pytest.mark.asyncio
async def test_archive_opens_only_selected_mcp_after_source_and_cache_removal(tmp_path, monkeypatch):
    monkeypatch.setenv("AKASHIC_BOOT_ID", "scoped-test-boot")
    monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    plugins = tmp_path / "plugins"
    write_plugin(plugins / "probe")
    other = plugins / "other"
    other.mkdir()
    (other / "plugin.py").write_text("""
from agent.plugin_composition import MCP_SERVERS, ServiceKey
api_version = 3
name = "other"
version = "1.0.0"
inject = (MCP_SERVERS, ServiceKey("test.bound.mcp"))
async def apply(ctx, config):
    await ctx.provide(ServiceKey("test.foreign.mcp"), lambda: ctx.require(MCP_SERVERS).open(ctx, "first"))
    await ctx.provide(ServiceKey("test.delegated.mcp"), ctx.require(ServiceKey("test.bound.mcp")))
""")
    first = manager(tmp_path, [plugins])
    log = MessageLog(tmp_path / "messages.db")
    try:
        await first.load_all()
        bindings = Bindings(log, first._archive, first.open_binding)
        async with lease_runtime_snapshot(first.snapshot_store):
            identity = bindings.bind(SERVICE, {})
            foreign = first.current_snapshot.composition_root.service_value(
                ServiceKey("test.foreign.mcp")
            )
            with pytest.raises(PermissionError, match="owner"):
                async with foreign():
                    pytest.fail("foreign Context opened a private target")
            delegated = first.current_snapshot.composition_root.service_value(
                ServiceKey("test.delegated.mcp")
            )
            async with delegated() as server:
                async with server.route() as route:
                    assert (await route.call("ping", {})).output == "fixed A"
            text_identity = bindings.bind(ServiceKey("test.bound.text"), {})
            generation = first.snapshot_store.current.generations["probe"]
            data = generation.data_dir
        assert (data / "first.count").read_text() == "2"
        assert (data / "second.count").read_text() == "1"
        record = first._archive.read_descriptor(generation.archive_ref)
        old_component = first._archive.save_descriptor({**record, "version": 1})
        with pytest.raises(RuntimeError, match="运行合同不兼容"):
            async with first.open_binding((old_component,)):
                pytest.fail("old component descriptor opened")
        env_refs = record["python_environments"]
        unused_env = first._archive.read_descriptor(env_refs["second"])
        environment_root = tmp_path / "workspace/runtime/plugin-python-environments"
        await first.terminate_all()
        shutil.rmtree(environment_root / unused_env["location"])
        shutil.rmtree(plugins)
        second = manager(tmp_path, [])
        try:
            recovered = Bindings(log, second._archive, second.open_binding)
            async with recovered.open(identity, SERVICE) as (open_server, _):
                async with open_server() as server:
                    async with server.route() as route:
                        result = await route.call("ping", {})
                        assert result.output == "fixed A"
            assert (data / "first.count").read_text() == "3"
            import json
            assert json.loads((data / "first.boot").read_text()) == {
                "AKASHIC_BOOT_ID": "scoped-test-boot", "AKASHIC_SUPERVISED": "1",
            }
            assert (data / "second.count").read_text() == "1"
            with pytest.raises(RuntimeError):
                server.route()
            shutil.rmtree(environment_root)
            async with recovered.open(text_identity, ServiceKey("test.bound.text")) as (
                text,
                _,
            ):
                assert text == "fixed text"
            async with recovered.open(identity, SERVICE) as (open_server, _):
                with pytest.raises(FileNotFoundError, match="环境根"):
                    async with open_server():
                        pytest.fail("missing environment was rebuilt")
            assert not environment_root.exists()
        finally:
            await second.terminate_all()
    finally:
        await first.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_runtime_command_failure_releases_root_before_scope_disposal(tmp_path):
    """环境无法打开时不保留 Root，也不启动额外的外部进程。"""
    from agent.plugins.composition_generation_host import CompositionGenerationHost

    plugins = tmp_path / "plugins"
    write_plugin(plugins / "probe")
    owner = manager(tmp_path, [plugins])

    def missing_environment(generation, kind, name):
        raise FileNotFoundError("fixed environment missing")

    failed = CompositionGenerationHost(command_resolver=missing_environment)
    try:
        await owner.load_all()
        snapshot = owner.current_snapshot
        generation = snapshot.generations["probe"]
        with pytest.raises(FileNotFoundError, match="fixed environment"):
            await failed.start(generation, snapshot, mode="formal")
        assert failed.get(generation.generation_id) is None
        assert failed._bridges == {}
        assert (generation.data_dir / "first.count").read_text() == "1"
        assert (generation.data_dir / "second.count").read_text() == "1"
    finally:
        await owner.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["retry", "shutdown", "concurrent"])
async def test_scoped_cleanup_failure_retains_resources_without_plugin_reload(
    tmp_path, monkeypatch, cleanup
):
    """真实 MCP 清理失败由调用资源 owner 重试，不能进入正式插件发布。"""
    plugins = tmp_path / "plugins"
    write_plugin(plugins / "probe")
    owner = manager(tmp_path, [plugins])
    log = MessageLog(tmp_path / "messages.db")
    returned = []
    original = owner._composition_generation_host._mcp_host._cleanup_entry
    failed = False
    selected = None

    async def fail_once(entry):
        nonlocal failed
        if entry.generation_id == selected and not failed:
            failed = True
            raise OSError("injected disconnect failure")
        await original(entry)

    monkeypatch.setattr(owner._composition_generation_host._mcp_host, "_cleanup_entry", fail_once)
    try:
        await owner.load_all()
        snapshot = owner.current_snapshot
        bindings = Bindings(log, owner._archive, owner.open_binding)
        async with lease_runtime_snapshot(owner.snapshot_store):
            identity = bindings.bind(SERVICE, {})
        status = owner.candidate_status()
        with pytest.raises(RuntimeError, match="injected disconnect failure"):
            async with bindings.open(identity, SERVICE) as (open_server, _):
                async with open_server() as server:
                    selected = server.generation_id
                    retained = owner._composition_generation_host._owners[selected]
                    retained.borrowed.callback(lambda: returned.append(selected))
                    async with server.route() as route:
                        assert (await route.call("ping", {})).output == "fixed A"
        assert failed
        assert returned == []
        failures = owner.resource_failures()
        assert len(failures) == 1 and failures[0].generation_id == selected
        assert "injected disconnect failure" in failures[0].error
        assert owner.current_snapshot is snapshot
        assert owner.candidate_status() == status
        if cleanup == "retry":
            await owner.retry_resource_cleanup(selected)
            assert owner.current_snapshot is snapshot
            assert owner.candidate_status() == status
        elif cleanup == "shutdown":
            await owner.terminate_all()
        else:
            await asyncio.gather(owner.retry_resource_cleanup(selected), owner.terminate_all())
        assert returned == [selected]
        assert owner.resource_failures() == ()
        assert owner._composition_generation_host.get(selected) is None
        assert selected not in owner._composition_generation_host._bridges
    finally:
        await owner.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_shutdown_waits_for_admitted_mcp_start_and_closes_new_admission(tmp_path, monkeypatch):
    """在实际 MCP 启动前暂停，关闭必须等完整启动后回收同一 owner。"""
    plugins = tmp_path / "plugins"
    write_plugin(plugins / "probe")
    owner = manager(tmp_path, [plugins])
    log = MessageLog(tmp_path / "messages.db")
    host = owner._composition_generation_host
    entered, release, closing, leave = (asyncio.Event() for _ in range(4))
    selected = None
    original_start = host._mcp_host.start_generation
    original_close = host.close_scoped

    async def delayed_start(scope_id, *args, **kwargs):
        nonlocal selected
        if host.scoped(scope_id):
            selected = scope_id
            entered.set()
            await release.wait()
        return await original_start(scope_id, *args, **kwargs)

    async def observe_close():
        closing.set()
        await original_close()

    task = shutdown = None
    try:
        await owner.load_all()
        snapshot = owner.current_snapshot
        bindings = Bindings(log, owner._archive, owner.open_binding)
        async with lease_runtime_snapshot(owner.snapshot_store):
            identity = bindings.bind(SERVICE, {})
        monkeypatch.setattr(host._mcp_host, "start_generation", delayed_start)
        monkeypatch.setattr(host, "close_scoped", observe_close)

        async def use():
            async with bindings.open(identity, SERVICE) as (open_server, _):
                async with open_server():
                    await leave.wait()

        task = asyncio.create_task(use())
        await entered.wait()
        shutdown = asyncio.create_task(owner.terminate_all())
        await closing.wait()
        assert not shutdown.done()
        with pytest.raises(RuntimeError, match="停止接纳"):
            async with host.open_mcp(snapshot, "first"):
                pytest.fail("shutdown admitted another MCP")
        release.set()
        await shutdown
        assert host.get(selected) is None
        assert selected not in host._bridges
        assert host._mcp_host.get(selected) is None
        assert host._process_host.get(selected) is None
        leave.set()
        await task
    finally:
        release.set()
        leave.set()
        await asyncio.gather(*(item for item in (task, shutdown) if item is not None), return_exceptions=True)
        await owner.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_manager_restart_reopens_scoped_resources_after_old_owners_drain(tmp_path):
    """同一 Manager 可以在完整停止后重新打开新 generation 的调用资源。"""
    plugins = tmp_path / "plugins"
    write_plugin(plugins / "probe")
    owner = manager(tmp_path, [plugins])
    try:
        for _ in range(2):
            await owner.load_all()
            async with lease_runtime_snapshot(owner.snapshot_store):
                open_server = owner.current_snapshot.composition_root.service_value(SERVICE)
                async with open_server() as server:
                    with pytest.raises(RuntimeError, match="尚未清空"):
                        owner._composition_generation_host.start_scoped()
                    async with server.route() as route:
                        assert (await route.call("ping", {})).output == "fixed A"
            await owner.terminate_all()
    finally:
        await owner.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["AKASHIC_BOOT_ID", "AKASHIC_SUPERVISED"])
async def test_mcp_registration_rejects_supervisor_identity_override(tmp_path, key):
    """注册边界在启动或发布前拒绝插件配置 Supervisor 的进程身份。"""
    from agent.plugin_composition import CompositionRoot, MCP_SERVERS, McpServerDefinition
    from agent.plugin_composition.mcp_slots import PluginMcpServers
    from agent.plugin_composition.model import PluginRuntime

    root = CompositionRoot("reserved-environment")
    service = PluginMcpServers(root.instance_token)
    await root.context.provide(MCP_SERVERS, service)

    async def apply(ctx):
        with pytest.raises(ValueError, match=key):
            await service.register(ctx, McpServerDefinition(
                name="invalid", command=("python",), candidate_env={key: "fake"},
            ))

    try:
        await root.mount(apply, name="probe", inject=(MCP_SERVERS,), runtime=PluginRuntime(
            "probe", "test", tmp_path, tmp_path, tmp_path, {},
        ))
    finally:
        await root.dispose()

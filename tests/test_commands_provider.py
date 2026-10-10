"""普通 Commands provider 的显式选择、贡献归属与恢复闭包。"""
import ast
from pathlib import Path
import shutil
import pytest
from plugins.ledger.contract import BINDINGS
from plugins.commands.contract import COMMANDS, CommandResult
from agent.plugins.manager import PluginManager
from plugins.ledger.log import MessageLog
from tests.fixtures.plugin_workspace import initialize_plugin_workspace

def _write_source(path, source):
    """在 fixture 写盘前静态解析并内存编译动态 Python 源码。"""
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    path.write_text(source)

@pytest.mark.asyncio
async def test_core_only_manager_does_not_supply_commands(tmp_path):
    initialize_plugin_workspace(tmp_path / "workspace")
    host = PluginManager([], workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home")
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        assert root.context.get(COMMANDS) is None
    finally:
        await host.terminate_all()

@pytest.mark.asyncio
async def test_binding_keeps_selected_provider_handler_and_dependency_closure(tmp_path):
    """异名 provider 按服务选择；命令绑定保留实际贡献闭包且不吸收无关插件。"""
    source = tmp_path / "plugins"
    provider = source / "human_actions"
    shutil.copytree(Path(__file__).parents[1] / "plugins/commands", provider,
                    ignore=shutil.ignore_patterns("__pycache__"))
    entry = provider / "plugin.py"
    _write_source(entry, entry.read_text().replace('name = "commands"', 'name = "human_actions"'))
    for name in ("dependency", "unused"):
        directory = source / name
        directory.mkdir()
        _write_source(directory / "plugin.py", f'''
from agent.plugin_composition import ServiceKey
api_version = 3
name = "{name}"
version = "1.0.0"
async def apply(ctx):
    await ctx.provide(ServiceKey("fixture.{name}"), "{name}")
''')
    owner = source / "command_owner"
    owner.mkdir()
    _write_source(owner / "plugin.py", '''
from agent.plugin_composition import ServiceKey
from plugins.commands.contract import COMMANDS, CommandDefinition, CommandResult
api_version = 3
name = "command_owner"
version = "1.0.0"
DEPENDENCY = ServiceKey("fixture.dependency")
inject = (COMMANDS, DEPENDENCY)
async def apply(ctx):
    value = ctx.require(DEPENDENCY)
    await ctx.require(COMMANDS).register(ctx, CommandDefinition(
        "probe", "show dependency", lambda call: CommandResult("success", value), read_only=True))
''')
    shutil.copytree(Path(__file__).parents[1] / "plugins/ledger", source / "ledger",
                    ignore=shutil.ignore_patterns("__pycache__"))
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    log = MessageLog(workspace / "sessions.db")
    host = PluginManager([source], workspace=workspace,
                         installed_cache_root=tmp_path / "home")
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        assert root.plugin_service_owners()[COMMANDS] == "human_actions"
        commands_context = root._service_provider(COMMANDS)[0]
        bindings = root.context.require(BINDINGS)
        async with commands_context.runtime_scope():
            registry = commands_context.require(COMMANDS).freeze()
            identity = registry.bind(bindings, "/probe")
        assert identity is not None
        descriptor = log.read_binding(identity)
        expected = {}
        for name in ("human_actions", "command_owner", "dependency"):
            generation = host.generation(name)
            assert generation is not None
            expected[name] = generation.generation_id
        assert descriptor["origins"] == expected
        async with commands_context.runtime_scope(), bindings.open(identity, COMMANDS) as (selected, metadata):
            assert selected is root.context.require(COMMANDS)
            assert metadata == {"name": "probe"}
            result = await selected.freeze().execute(
                "/probe", session_key="chat", channel="test", chat_id="room", sender="user", recover=True,
            )
            assert result is not None
            assert result.result == CommandResult("success", "dependency")
    finally:
        cleanup_errors = []
        try:
            await host.terminate_all()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            log.close()
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup("Commands binding test cleanup failed", cleanup_errors)

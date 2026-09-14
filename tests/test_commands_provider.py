"""普通 Commands provider 的显式选择、贡献归属与恢复闭包。"""
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition import (
    CompositionError, CompositionRoot, PluginRuntime, SNAPSHOT_SEALING, SnapshotSealing,
)
from agent.plugin_composition.archive import PluginArchive
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.commands import COMMANDS, CommandDefinition, CommandResult
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.commands import plugin as commands_plugin
from plugins.commands.registry import PluginCommands
from session.log import MessageLog


@pytest.mark.asyncio
async def test_core_only_manager_does_not_supply_commands(tmp_path):
    host = PluginManager([], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home")
    try:
        await host.load_all()
        snapshot = host.current_snapshot
        if snapshot is not None and snapshot.composition_root is not None:
            assert snapshot.composition_root.context.get(COMMANDS) is None
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_provider_rejects_foreign_context_and_registration_after_sealing(tmp_path):
    root, foreign = CompositionRoot("selected"), CompositionRoot("foreign")
    contexts = []
    definition = CommandDefinition("echo", "echo input", lambda call: CommandResult("success", call.raw_input))

    async def contribute(ctx):
        contexts.append(ctx)
        await ctx.require(COMMANDS).register(ctx, definition)

    try:
        await root.mount(commands_plugin.apply, name="human-actions")
        await root.mount(contribute, name="echo-owner", inject=(COMMANDS,), runtime=PluginRuntime(
            plugin_id="echo-owner", generation_id="echo-generation", plugin_dir=tmp_path,
            data_dir=tmp_path / "data", workspace=tmp_path, config={},
        ))
        commands = root.context.require(COMMANDS)
        with pytest.raises(ValueError, match="同一 Root"):
            await commands.register(foreign.context, definition)
        borrowed = PluginCommands(contexts[0])
        with pytest.raises(ValueError, match="实际选中"):
            await borrowed.register(contexts[0], definition)
        await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
        root.freeze()
        with pytest.raises(CompositionError, match="已封存"):
            await commands.register(contexts[0], definition)
        result = await commands.freeze().execute(
            "/ECHO@bot  original", session_key="chat", channel="test", chat_id="room", sender="user",
        )
        assert result is not None
        assert result.result == CommandResult("success", "  original")
        assert commands.freeze().descriptors[0].owner == "echo-owner"
    finally:
        await foreign.dispose()
        await root.dispose()


@pytest.mark.asyncio
async def test_binding_keeps_selected_provider_handler_and_dependency_closure(tmp_path):
    """异名 provider 按服务选择；命令绑定保留实际贡献闭包且不吸收无关插件。"""
    source = tmp_path / "plugins"
    provider = source / "human_actions"
    shutil.copytree(Path(__file__).parents[1] / "plugins/commands", provider,
                    ignore=shutil.ignore_patterns("__pycache__"))
    entry = provider / "plugin.py"
    entry.write_text(entry.read_text().replace('name = "commands"', 'name = "human_actions"'))
    for name in ("dependency", "unused"):
        directory = source / name
        directory.mkdir()
        (directory / "plugin.py").write_text(f'''
from agent.plugin_composition import ServiceKey
api_version = 3
name = "{name}"
version = "1.0.0"
async def apply(ctx):
    await ctx.provide(ServiceKey("fixture.{name}"), "{name}")
''')
    owner = source / "command_owner"
    owner.mkdir()
    (owner / "plugin.py").write_text('''
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.commands import COMMANDS, CommandDefinition, CommandResult
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
    log = MessageLog(tmp_path / "sessions.db")
    workspace = tmp_path / "workspace"
    host = PluginManager([source], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root
            assert root is not None
            assert root.plugin_service_owners()[COMMANDS] == "human_actions"
            registry = root.context.require(COMMANDS).freeze()
            bindings = root.context.require(BINDINGS)
            identity = registry.bind(bindings, "/probe")
            assert identity is not None
            archive = PluginArchive(workspace / "runtime/plugin-archives", create=False)
            root_ref = log.read_binding(identity)["root_ref"]
            assert isinstance(root_ref, str)
            descriptor = archive.read_descriptor(root_ref)
            assert set(descriptor["components"]) == {
                snapshot.generations[name].archive_ref
                for name in ("human_actions", "command_owner", "dependency")
            }
            async with bindings.open(identity, COMMANDS) as (selected, metadata):
                assert selected is root.context.require(COMMANDS)
                assert metadata == {"name": "probe"}
                result = await selected.freeze().execute(
                    "/probe", session_key="chat", channel="test", chat_id="room", sender="user", recover=True,
                )
                assert result is not None
                assert result.result == CommandResult("success", "dependency")
    finally:
        await host.terminate_all()
        log.close()

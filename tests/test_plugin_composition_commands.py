from __future__ import annotations

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.passive_turn import (
    ContextStore,
    PassiveTurnDeps,
    PassiveTurnPipeline,
    Reasoner,
)
from agent.context import ContextBuilder
from agent.lifecycle.types import BeforeTurnCtx
from agent.looping.ports import SessionServices
from agent.plugin_composition import (
    COMMANDS,
    CommandDefinition,
    CommandDescriptor,
    CommandInvocation,
    CommandRegistry,
    CommandResult,
    CompositionRoot,
    PluginCommands,
    PluginRuntime,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manager import PluginManager
from agent.plugins.registry import plugin_registry
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from agent.tools.registry import ToolRegistry
from agent.turns.outbound import OutboundPort
from bus.event_bus import EventBus
from bus.events import InboundMessage, TurnDisposition


@pytest.fixture(autouse=True)
def _clean_registry():
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


class _LeakedPluginCommands(PluginCommands):
    def _register(
        self,
        plugin_id: str,
        definition: CommandDefinition,
    ) -> Callable[[], None]:
        _ = super()._register(plugin_id, definition)
        return lambda: None


def _write_plugin(root: Path, source: str) -> None:
    plugin_dir = root / "command_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")


def _manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


@pytest.mark.asyncio
async def test_command_registry_preserves_alias_and_argument_compatibility(
    tmp_path: Path,
) -> None:
    seen: list[CommandInvocation] = []
    root = CompositionRoot("commands-unit")
    commands = PluginCommands()
    _ = await root.context.provide(COMMANDS, commands)

    async def plugin(ctx) -> None:
        async def handle(invocation: CommandInvocation) -> CommandResult:
            seen.append(invocation)
            return CommandResult("success", "ready")

        await commands.register(
            ctx,
            CommandDefinition(
                name="chatid",
                description="查看 chat_id",
                aliases=("myid",),
                input_hint="[ignored]",
                handler=handle,
            ),
        )

    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    _ = await root.mount(
        plugin,
        name="command_probe",
        inject=(COMMANDS,),
        runtime=PluginRuntime(
            plugin_id="command_probe",
            plugin_dir=plugin_dir,
            data_dir=plugin_dir / "data",
            workspace=plugin_dir / "workspace",
            config=object(),
        ),
    )
    registry = commands.freeze()

    assert registry.descriptors == (
        CommandDescriptor(
            name="chatid",
            description="查看 chat_id",
            input_hint="[ignored]",
        ),
    )
    execution = await registry.execute(
        "  /MYID@AkashicBot extra  ",
        session_key="telegram:42",
        channel="telegram",
        chat_id="42",
        sender="hua",
    )
    assert execution is not None
    assert execution.name == "chatid"
    assert execution.result == CommandResult("success", "ready")
    assert seen == [
        CommandInvocation(
            name="chatid",
            raw_input=" extra",
            session_key="telegram:42",
            channel="telegram",
            chat_id="42",
            sender="hua",
        )
    ]
    assert (
        await registry.execute(
            "/unknown",
            session_key="telegram:42",
            channel="telegram",
            chat_id="42",
            sender="hua",
        )
        is None
    )
    assert len(seen) == 1
    await root.dispose()


@pytest.mark.asyncio
async def test_command_registry_accepts_sync_handler_and_rejects_empty_result() -> None:
    def handle(invocation: CommandInvocation) -> CommandResult:
        return CommandResult("success", invocation.raw_input.strip())

    registry = CommandRegistry(
        {
            "echo": CommandDefinition(
                name="echo",
                description="echo",
                handler=handle,
            )
        },
        (),
    )
    execution = await registry.execute(
        "/echo ready",
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        sender="hua",
    )
    assert execution is not None
    assert execution.result == CommandResult("success", "ready")

    with pytest.raises(ValueError, match="result text 不能为空"):
        _ = await registry.execute(
            "/echo",
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            sender="hua",
        )


@pytest.mark.asyncio
async def test_command_registration_rolls_back_duplicate_and_kills_mutant(
    tmp_path: Path,
) -> None:
    correct, correct_errors = await _duplicate_registration_fixture(
        tmp_path / "correct",
        PluginCommands,
    )
    mutant, mutant_errors = await _duplicate_registration_fixture(
        tmp_path / "mutant",
        _LeakedPluginCommands,
    )

    assert any("插件 Command 名称重复" in error for error in correct_errors)
    assert any("插件 Command 名称重复" in error for error in mutant_errors)
    assert correct == ()
    assert tuple(item.name for item in mutant) == ("first",)


async def _duplicate_registration_fixture(
    root_dir: Path,
    commands_type: type[PluginCommands],
) -> tuple[tuple[object, ...], tuple[str, ...]]:
    """Run an alias conflict through real Fiber rollback."""

    # 1. Mount identical behavior against the production and mutant collectors.
    root = CompositionRoot(f"command-duplicate:{commands_type.__name__}")
    commands = commands_type()
    _ = await root.context.provide(COMMANDS, commands)

    async def handle(invocation: CommandInvocation) -> CommandResult:
        del invocation
        return CommandResult("success", "ready")

    async def plugin(ctx) -> None:
        await commands.register(
            ctx,
            CommandDefinition(
                name="first",
                description="first",
                aliases=("shared",),
                handler=handle,
            ),
        )
        await commands.register(
            ctx,
            CommandDefinition(
                name="second",
                description="second",
                aliases=("shared",),
                handler=handle,
            ),
        )

    plugin_dir = root_dir / "plugin"
    plugin_dir.mkdir(parents=True)
    _ = await root.mount(
        plugin,
        name="command_probe",
        inject=(COMMANDS,),
        runtime=PluginRuntime(
            plugin_id="command_probe",
            plugin_dir=plugin_dir,
            data_dir=plugin_dir / "data",
            workspace=plugin_dir / "workspace",
            config=object(),
        ),
    )

    # 2. Freeze before Root disposal so cleanup remains observable.
    descriptors = cast(tuple[object, ...], commands.freeze().descriptors)
    errors = root.receipt().errors
    await root.dispose()
    return descriptors, errors


@pytest.mark.asyncio
async def test_v3_command_snapshot_short_circuits_before_session_and_model(
    tmp_path: Path,
) -> None:
    _write_plugin(tmp_path / "plugins", _command_plugin_source())
    manager = _manager(tmp_path)
    await manager.load_all()

    generation = manager.generation("command_probe")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert isinstance(generation.instance, ComposablePlugin)
    assert snapshot.command_registry is not None
    assert tuple(item.name for item in snapshot.command_registry.descriptors) == (
        "chatid",
    )
    assert manager.telegram_bot_commands == [("chatid", "查看 chat_id")]
    assert manager.mobile_bot_commands == []
    assert snapshot.composition_topology is not None
    assert "core.commands" in snapshot.composition_topology.services

    session_manager = SimpleNamespace(
        get_or_create=MagicMock(),
        peek_next_message_id=MagicMock(),
        append_messages=AsyncMock(),
    )
    context_store = SimpleNamespace(prepare=AsyncMock())
    reasoner = SimpleNamespace(run_turn=AsyncMock())
    outbound_port = SimpleNamespace(dispatch=AsyncMock())
    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(session_manager=session_manager, presence=None),
            ),
            context_store=cast(ContextStore, context_store),
            context=cast(ContextBuilder, SimpleNamespace()),
            tools=cast(ToolRegistry, SimpleNamespace()),
            reasoner=cast(Reasoner, reasoner),
            outbound_port=cast(OutboundPort, outbound_port),
        )
    )

    lease = manager.snapshot_store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        outbound = await pipeline.run(
            InboundMessage(
                channel="telegram",
                sender="hua",
                chat_id="42",
                content="/MYID@AkashicBot ignored",
            ),
            "telegram:42",
        )
    finally:
        reset_runtime_snapshot(token)
        await lease.release()

    assert outbound.content == "telegram:42"
    assert outbound.turn_disposition is TurnDisposition.SHORT_CIRCUITED
    assert generation.instance.module.seen == [
        ("chatid", " ignored", "telegram:42", "telegram", "42", "hua")
    ]
    session_manager.get_or_create.assert_not_called()
    session_manager.peek_next_message_id.assert_not_called()
    session_manager.append_messages.assert_not_awaited()
    context_store.prepare.assert_not_awaited()
    reasoner.run_turn.assert_not_awaited()
    outbound_port.dispatch.assert_awaited_once()

    root = snapshot.composition_root
    assert root is not None
    assert "command_probe:command:chatid" in root.receipt().effects
    await manager.terminate_all()
    assert root.receipt().effects == ()
    assert root.receipt().services == ()


@pytest.mark.asyncio
async def test_unknown_command_continues_into_existing_before_turn_path(
    tmp_path: Path,
) -> None:
    _write_plugin(tmp_path / "plugins", _command_plugin_source())
    manager = _manager(tmp_path)
    await manager.load_all()
    snapshot = manager.current_snapshot
    assert snapshot is not None

    before_turn = SimpleNamespace(
        run=AsyncMock(
            return_value=BeforeTurnCtx(
                session_key="telegram:42",
                channel="telegram",
                chat_id="42",
                content="/unknown",
                timestamp=InboundMessage(
                    channel="telegram",
                    sender="hua",
                    chat_id="42",
                    content="/unknown",
                ).timestamp,
                retrieved_memory_block="",
                retrieval_trace_raw=None,
                history_messages=(),
                abort=True,
                abort_reply="legacy path",
            )
        )
    )
    pipeline = PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(
                    session_manager=SimpleNamespace(),
                    presence=None,
                ),
            ),
            context_store=cast(ContextStore, SimpleNamespace()),
            context=cast(ContextBuilder, SimpleNamespace()),
            tools=cast(ToolRegistry, SimpleNamespace()),
            reasoner=cast(Reasoner, SimpleNamespace()),
        )
    )
    phases = SimpleNamespace(before_turn=before_turn)
    cast(Any, pipeline)._runtime_phases = lambda: phases

    lease = manager.snapshot_store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        outbound = await pipeline.run(
            InboundMessage(
                channel="telegram",
                sender="hua",
                chat_id="42",
                content="/unknown",
            ),
            "telegram:42",
            dispatch_outbound=False,
        )
    finally:
        reset_runtime_snapshot(token)
        await lease.release()

    assert outbound.content == "legacy path"
    before_turn.run.assert_awaited_once()
    await manager.terminate_all()


def _command_plugin_source() -> str:
    return (
        "from agent.plugin_composition import (\n"
        "    COMMANDS, CommandDefinition, CommandResult,\n"
        ")\n"
        "api_version = 3\n"
        "name = 'command_probe'\n"
        "version = '1.0.0'\n"
        "inject = (COMMANDS,)\n"
        "seen = []\n"
        "async def handle(invocation):\n"
        "    seen.append((\n"
        "        invocation.name, invocation.raw_input, invocation.session_key,\n"
        "        invocation.channel, invocation.chat_id, invocation.sender,\n"
        "    ))\n"
        "    return CommandResult('success', f'{invocation.channel}:{invocation.chat_id}')\n"
        "async def apply(ctx, config):\n"
        "    del config\n"
        "    await ctx.require(COMMANDS).register(\n"
        "        ctx,\n"
        "        CommandDefinition(\n"
        "            name='chatid', description='查看 chat_id',\n"
        "            aliases=('myid',), handler=handle,\n"
        "        ),\n"
        "    )\n"
    )

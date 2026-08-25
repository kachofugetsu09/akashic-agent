from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.context import ContextBuilder
from agent.core.passive_turn import (
    ContextStore,
    PassiveTurnDeps,
    PassiveTurnPipeline,
    Reasoner,
)
from agent.core.runtime_support import SessionLike
from agent.looping.ports import SessionServices
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps
from agent.context import ContextBuilder
from agent.plugin_composition import (
    COMMANDS,
    CommandDefinition,
    CommandDescriptor,
    CommandRegistry,
    CommandResult,
    CompositionRoot,
    PluginCommands,
    PluginRuntime,
)
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.lifecycle.types import BeforeTurnCtx
from agent.tools.registry import ToolRegistry
from agent.turns.outbound import OutboundPort
from bus.event_bus import EventBus
from bus.events import InboundMessage, TurnDisposition
from bus.queue import MessageBus
from tests.memory_fakes import FakeMemoryEngine
from tests.provider_fakes import ProviderContextBudgetStub


def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir


def _manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


def _command_plugin(description: str, reply: str) -> str:
    return (
        "from agent.plugin_composition import COMMANDS, CommandDefinition, CommandResult\n"
        "api_version = 3\n"
        "name = 'commands_v3'\n"
        "version = '1.0.0'\n"
        "inject = (COMMANDS,)\n"
        "async def apply(ctx, config):\n"
        "    async def handler(invocation):\n"
        f"        return CommandResult('success', {reply!r} + ':' + invocation.raw_input)\n"
        "    await ctx.require(COMMANDS).register(ctx, CommandDefinition(\n"
        f"        name='hello', description={description!r}, handler=handler,\n"
        "        aliases=('hi',), input_hint='name'))\n"
    )


@pytest.mark.asyncio
async def test_plugin_commands_execute_alias_and_cleanup(tmp_path: Path) -> None:
    root = CompositionRoot("commands")
    commands = PluginCommands()
    _ = await root.context.provide(COMMANDS, commands)
    runtime = PluginRuntime(
        plugin_id="command-probe",
        plugin_dir=tmp_path / "plugin",
        data_dir=tmp_path / "data",
        workspace=tmp_path,
        config=None,
    )

    async def plugin(ctx) -> None:
        async def handler(invocation):
            return CommandResult("success", invocation.raw_input or "empty")

        await ctx.require(COMMANDS).register(
            ctx,
            CommandDefinition(
                name="hello",
                description="say hello",
                handler=handler,
                aliases=("hi",),
                input_hint="name",
            ),
        )

    _ = await root.mount(plugin, name="command-probe", runtime=runtime)
    registry = commands.freeze()

    execution = await registry.execute(
        "/HI@akashic  花月",
        session_key="web:1",
        channel="web",
        chat_id="1",
        sender="hua",
    )

    assert execution is not None
    assert execution.name == "hello"
    assert execution.result == CommandResult("success", "  花月")
    assert registry.claims == {"hello": "command-probe", "hi": "command-probe"}
    assert registry.descriptors[0].input_hint == "name"
    await root.dispose()
    assert root.receipt().effects == ()
    assert root.receipt().services == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ("bad-name", "a" * 33))
async def test_plugin_commands_reject_names_outside_channel_contract(
    tmp_path: Path,
    name: str,
) -> None:
    root = CompositionRoot("commands")
    commands = PluginCommands()
    _ = await root.context.provide(COMMANDS, commands)
    runtime = PluginRuntime(
        plugin_id="command-probe",
        plugin_dir=tmp_path / "plugin",
        data_dir=tmp_path / "data",
        workspace=tmp_path,
        config=None,
    )

    async def plugin(ctx) -> None:
        await ctx.require(COMMANDS).register(
            ctx,
            CommandDefinition(
                name=name,
                description="invalid",
                handler=lambda _invocation: CommandResult("success", "ok"),
            ),
        )

    fiber = await root.mount(plugin, name="command-probe", runtime=runtime)
    assert fiber.state.value == "failed"
    assert "Command name 无效" in (root.receipt().fibers[0].error or "")
    assert not root.receipt().ready
    await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "aliases", "match"),
    (("stop", (), "Core 保留"), ("hello", ("stop",), "别名由 Core 保留")),
)
async def test_plugin_commands_reject_core_reserved_names(
    tmp_path: Path,
    name: str,
    aliases: tuple[str, ...],
    match: str,
) -> None:
    root = CompositionRoot("commands")
    commands = PluginCommands()
    _ = await root.context.provide(COMMANDS, commands)
    runtime = PluginRuntime(
        plugin_id="command-probe",
        plugin_dir=tmp_path / "plugin",
        data_dir=tmp_path / "data",
        workspace=tmp_path,
        config=None,
    )

    async def plugin(ctx) -> None:
        await ctx.require(COMMANDS).register(
            ctx,
            CommandDefinition(
                name=name,
                description="reserved",
                handler=lambda _invocation: CommandResult("success", "ok"),
                aliases=aliases,
            ),
        )

    fiber = await root.mount(plugin, name="command-probe", runtime=runtime)
    assert fiber.state.value == "failed"
    assert match in (root.receipt().fibers[0].error or "")
    assert not root.receipt().ready
    await root.dispose()


@pytest.mark.asyncio
async def test_plugin_commands_reject_channel_description_over_256_chars(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("commands")
    commands = PluginCommands()
    _ = await root.context.provide(COMMANDS, commands)
    runtime = PluginRuntime(
        plugin_id="command-probe",
        plugin_dir=tmp_path / "plugin",
        data_dir=tmp_path / "data",
        workspace=tmp_path,
        config=None,
    )

    async def plugin(ctx) -> None:
        await ctx.require(COMMANDS).register(
            ctx,
            CommandDefinition(
                name="hello",
                description="x" * 257,
                handler=lambda _invocation: CommandResult("success", "ok"),
            ),
        )

    fiber = await root.mount(plugin, name="command-probe", runtime=runtime)
    assert fiber.state.value == "failed"
    assert "超过 256 字符" in (root.receipt().fibers[0].error or "")
    assert not root.receipt().ready
    await root.dispose()


def test_command_digest_covers_every_descriptor_field() -> None:
    async def handler(_invocation):
        return CommandResult("success", "ok")

    base = CommandDefinition("hello", "description", handler, ("hi",), "name")

    def digest(
        definition: CommandDefinition,
        *,
        owner: str = "plugin-a",
    ) -> str:
        commands = {definition.name: definition}
        owners = {definition.name: owner}
        descriptor = (
            CommandDescriptor(
                definition.name,
                definition.description,
                definition.aliases,
                definition.input_hint,
                owner,
            ),
        )
        return CommandRegistry(commands, owners, descriptor).catalog_digest

    variants = (
        CommandDefinition("other", "description", handler, ("hi",), "name"),
        CommandDefinition("hello", "changed", handler, ("hi",), "name"),
        CommandDefinition("hello", "description", handler, ("hey",), "name"),
        CommandDefinition("hello", "description", handler, ("hi",), "target"),
    )

    baseline = digest(base)
    assert all(digest(item) != baseline for item in variants)
    assert digest(base, owner="plugin-b") != baseline


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "description", "aliases", "input_hint", "owner"),
    (
        ("other", "description", ("hi",), "name", "plugin-a"),
        ("hello", "changed", ("hi",), "name", "plugin-a"),
        ("hello", "description", ("hey",), "name", "plugin-a"),
        ("hello", "description", ("hi",), "target", "plugin-a"),
        ("hello", "description", ("hi",), "name", "plugin-b"),
    ),
)
async def test_command_descriptor_fields_change_snapshot_identity(
    tmp_path: Path,
    name: str,
    description: str,
    aliases: tuple[str, ...],
    input_hint: str,
    owner: str,
) -> None:
    async def build(
        *,
        command_name: str,
        command_description: str,
        command_aliases: tuple[str, ...],
        command_input_hint: str,
        plugin_id: str,
    ) -> CompositionRoot:
        root = CompositionRoot("commands")
        commands = PluginCommands()
        _ = await root.context.provide(COMMANDS, commands)
        runtime = PluginRuntime(
            plugin_id=plugin_id,
            plugin_dir=tmp_path / "plugin",
            data_dir=tmp_path / "data",
            workspace=tmp_path,
            config=None,
        )

        async def plugin(ctx) -> None:
            await ctx.require(COMMANDS).register(
                ctx,
                CommandDefinition(
                    command_name,
                    command_description,
                    lambda _invocation: CommandResult("success", "ok"),
                    command_aliases,
                    command_input_hint,
                ),
            )

        _ = await root.mount(plugin, name="plugin", runtime=runtime)
        return root

    baseline_root = await build(
        command_name="hello",
        command_description="description",
        command_aliases=("hi",),
        command_input_hint="name",
        plugin_id="plugin-a",
    )
    variant_root = await build(
        command_name=name,
        command_description=description,
        command_aliases=aliases,
        command_input_hint=input_hint,
        plugin_id=owner,
    )
    compiler = RuntimeSnapshotCompiler()

    baseline = compiler.compile({}, composition_root=baseline_root)
    variant = compiler.compile({}, composition_root=variant_root)

    assert baseline.composition_topology is not None
    assert variant.composition_topology is not None
    assert (
        baseline.composition_topology.identity == variant.composition_topology.identity
    )
    assert baseline.snapshot_id != variant.snapshot_id
    await baseline_root.dispose()
    await variant_root.dispose()


@pytest.mark.asyncio
async def test_manager_keeps_candidate_commands_private_until_promotion(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "commands_v3",
        _command_plugin("old description", "old"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    assert old_snapshot is not None and old_snapshot.command_registry is not None
    assert manager.stable_telegram_command_catalog() == (("hello", "old description"),)
    assert manager.stable_mobile_command_catalog() == (("hello", "old description"),)
    endpoint_calls: list[tuple[tuple[str, str], ...]] = []

    async def endpoint_switcher(
        _old_commands,
        new_commands,
    ) -> None:
        provisional = manager.latest_snapshot
        assert provisional is not None and provisional is not old_snapshot
        assert manager.current_snapshot is old_snapshot
        assert provisional.accepting_leases is False
        assert old_snapshot.state == "committed"
        assert old_snapshot.accepting_leases is False
        assert manager.stable_telegram_command_catalog() == (
            ("hello", "old description"),
        )
        assert manager.stable_mobile_command_catalog() == (
            ("hello", "old description"),
        )
        with pytest.raises(RuntimeError, match="暂停接收"):
            manager.snapshot_store.lease()
        endpoint_calls.append(new_commands)

    quiesce = AsyncMock()
    resume = AsyncMock()
    manager.bind_endpoint_switcher(endpoint_switcher)
    manager.bind_endpoint_admission(quiesce=quiesce, resume=resume)

    (plugin_dir / "plugin.py").write_text(
        _command_plugin("new description", "new"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("commands_v3")
    assert candidate is not None and candidate.runtime_snapshot is not None
    assert candidate.runtime_snapshot.snapshot_id != old_snapshot.snapshot_id
    assert manager.stable_telegram_command_catalog() == (("hello", "old description"),)

    result = await manager.publish_prepared("commands_v3")

    assert result["publication_state"] == "committed"
    assert manager.stable_telegram_command_catalog() == (("hello", "new description"),)
    assert manager.stable_mobile_command_catalog() == (("hello", "new description"),)
    assert all(
        name != "hi" for name, _description in manager.stable_telegram_command_catalog()
    )
    assert endpoint_calls == [(("hello", "new description"),)]
    quiesce.assert_not_awaited()
    resume.assert_not_awaited()
    root = manager.current_snapshot.composition_root
    assert root is not None
    await manager.terminate_all()
    assert root.receipt().effects == ()
    assert root.receipt().services == ()


@pytest.mark.asyncio
async def test_command_catalog_failure_restores_old_stable_and_generation(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "commands_v3",
        _command_plugin("old description", "old"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    old_generation = manager.generation("commands_v3")
    assert old_snapshot is not None and old_generation is not None
    published: list[tuple[tuple[str, str], ...]] = []

    async def endpoint_switcher(
        _old_commands,
        new_commands,
    ) -> None:
        published.append(new_commands)
        if new_commands == (("hello", "new description"),):
            raise RuntimeError("telegram publication failed")

    manager.bind_endpoint_switcher(endpoint_switcher)
    (plugin_dir / "plugin.py").write_text(
        _command_plugin("new description", "new"),
        encoding="utf-8",
    )
    assert await manager.prepare_candidate("commands_v3") is not None

    with pytest.raises(RuntimeError, match="telegram publication failed"):
        await manager.publish_prepared("commands_v3")

    assert published == [
        (("hello", "new description"),),
        (("hello", "old description"),),
    ]
    assert manager.current_snapshot is old_snapshot
    assert manager.generation("commands_v3") is old_generation
    assert manager.stable_telegram_command_catalog() == (("hello", "old description"),)
    lease = manager.snapshot_store.lease()
    assert lease.snapshot is old_snapshot
    await lease.release()
    await manager.terminate_all()


class _CommandProvider(ProviderContextBudgetStub):
    async def chat(self, **_kwargs):
        raise AssertionError("known command must not call the model")


@pytest.mark.asyncio
async def test_agent_loop_command_precedes_model_session_and_turn_started(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "commands_v3",
        _command_plugin("description", "handled"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    session_manager = MagicMock()
    tools = ToolRegistry()
    provider = _CommandProvider()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=cast(Any, provider),
            tools=tools,
            session_manager=session_manager,
            workspace=tmp_path / "loop-workspace",
                context=ContextBuilder(tmp_path, FakeMemoryEngine(tmp_path)),
        ),
        AgentLoopConfig(),
    )
    loop.bind_runtime_snapshot_store(manager.snapshot_store)
    loop._resolve_model_selection = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("known command must not resolve a model")
    )
    loop._observe_turn_started = AsyncMock()  # type: ignore[method-assign]

    result = await loop._process_with_runtime_admission(
        InboundMessage("web", "hua", "1", "/hi Akashic"),
        dispatch_outbound=False,
    )

    assert result.content == "handled: Akashic"
    session_manager.get_or_create.assert_not_called()
    loop._resolve_model_selection.assert_not_awaited()
    loop._observe_turn_started.assert_not_awaited()
    await manager.terminate_all()


def _passive_pipeline(
    *,
    session_manager: object,
    reasoner: object,
    outbound_port: object,
) -> PassiveTurnPipeline:
    return PassiveTurnPipeline(
        PassiveTurnDeps(
            session=cast(
                SessionServices,
                SimpleNamespace(session_manager=session_manager, presence=None),
            ),
            context_store=cast(
                ContextStore,
                SimpleNamespace(prepare=AsyncMock()),
            ),
            context=cast(
                ContextBuilder,
                SimpleNamespace(render=MagicMock()),
            ),
            tools=cast(ToolRegistry, SimpleNamespace(set_context=MagicMock())),
            reasoner=cast(Reasoner, reasoner),
            event_bus=EventBus(),
            outbound_port=cast(OutboundPort, outbound_port),
        )
    )


@pytest.mark.asyncio
async def test_stable_command_short_circuits_before_session_and_model(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "commands_v3",
        _command_plugin("description", "handled"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    lease = manager.snapshot_store.lease()
    token = bind_runtime_snapshot(lease)
    session_manager = SimpleNamespace(get_or_create=MagicMock())
    reasoner = SimpleNamespace(run_turn=AsyncMock())
    outbound = SimpleNamespace(dispatch=AsyncMock())
    pipeline = _passive_pipeline(
        session_manager=session_manager,
        reasoner=reasoner,
        outbound_port=outbound,
    )
    try:
        result = await pipeline.run(
            InboundMessage("web", "hua", "1", "/hi Akashic"),
            "web:1",
        )
    finally:
        reset_runtime_snapshot(token)
        await lease.release()

    assert result.content == "handled: Akashic"
    assert result.turn_disposition is TurnDisposition.SHORT_CIRCUITED
    session_manager.get_or_create.assert_not_called()
    reasoner.run_turn.assert_not_awaited()
    outbound.dispatch.assert_awaited_once()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_unknown_command_continues_before_turn_path(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "commands_v3",
        _command_plugin("description", "handled"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    lease = manager.snapshot_store.lease()
    token = bind_runtime_snapshot(lease)
    session = cast(SessionLike, SimpleNamespace(key="web:1", messages=[], metadata={}))
    session_manager = SimpleNamespace(get_or_create=MagicMock(return_value=session))
    reasoner = SimpleNamespace(run_turn=AsyncMock())
    outbound = SimpleNamespace(dispatch=AsyncMock())
    pipeline = _passive_pipeline(
        session_manager=session_manager,
        reasoner=reasoner,
        outbound_port=outbound,
    )

    async def abort(ctx):
        ctx.abort = True
        ctx.abort_reply = "legacy path"
        return ctx

    pipeline._bus.on(BeforeTurnCtx, abort)  # pyright: ignore[reportPrivateUsage]
    try:
        result = await pipeline.run(
            InboundMessage("web", "hua", "1", "/unknown"),
            "web:1",
        )
    finally:
        reset_runtime_snapshot(token)
        await lease.release()

    assert result.content == "legacy path"
    session_manager.get_or_create.assert_called_once_with("web:1")
    reasoner.run_turn.assert_not_awaited()
    await manager.terminate_all()

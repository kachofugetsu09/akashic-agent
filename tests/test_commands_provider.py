"""普通 Commands provider 的显式选择、贡献归属与恢复闭包。"""
import ast
import asyncio
from pathlib import Path
import shutil
from typing import cast

import pytest

from agent.plugin_composition import CompositionError, CompositionRoot, Context, FiberState, PluginRuntime, ServiceKey
from agent.plugin_composition import RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.archive import PluginArchive
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.commands import (
    COMMANDS, CommandDefinition, CommandRecoveryRequired, CommandResult,
)
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from plugins.commands import plugin as commands_plugin
from plugins.commands.registry import CommandRegistry, PluginCommands
from session.log import MessageLog
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


def _write_source(path, source):
    """在 fixture 写盘前静态解析并内存编译动态 Python 源码。"""
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    path.write_text(source)


@pytest.mark.asyncio
async def test_core_only_manager_does_not_supply_commands(tmp_path):
    initialize_plugin_workspace(tmp_path / "workspace")
    host = PluginManager([], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home")
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        assert root.context.get(COMMANDS) is None
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_provider_rejects_foreign_context_and_keeps_views_independent(tmp_path):
    root, foreign = CompositionRoot("selected"), CompositionRoot("foreign")
    contexts = []
    definition = CommandDefinition("echo", "echo input", lambda call: CommandResult("success", call.raw_input))

    async def contribute(ctx):
        contexts.append(ctx)
        await ctx.require(COMMANDS).register(ctx, definition)

    try:
        provider_fiber = await root.mount(commands_plugin.apply, name="human-actions")
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
        first = commands.freeze()
        extra = CommandDefinition(
            "fresh", "fresh input", lambda call: CommandResult("success", call.raw_input),
            aliases=("f",),
        )
        await commands.register(contexts[0], extra)
        second = commands.freeze()
        assert [descriptor.name for descriptor in first.descriptors] == ["echo"]
        assert [descriptor.name for descriptor in second.descriptors] == ["echo", "fresh"]
        assert second.descriptors[1].aliases == ("f",)
        assert await first.execute(
            "/fresh", session_key="chat", channel="test", chat_id="room", sender="user",
        ) is None
        result = await first.execute(
            "/ECHO@bot  original", session_key="chat", channel="test", chat_id="room", sender="user",
        )
        assert result is not None
        assert result.result == CommandResult("success", "  original")
        assert second.descriptors[0].owner == "echo-owner"

        assert await second.execute(
            "/f  fresh", session_key="chat", channel="test", chat_id="room", sender="user",
        ) is not None
        fresh_effect = next(
            effect for effect in contexts[0]._fiber.effects if effect.label == "command:fresh"
        )
        await fresh_effect.aclose()
        third = commands.freeze()
        assert [descriptor.name for descriptor in third.descriptors] == ["echo"]
        assert await second.execute(
            "/f  retained", session_key="chat", channel="test", chat_id="room", sender="user",
        ) is not None
        assert await third.execute(
            "/f  removed", session_key="chat", channel="test", chat_id="room", sender="user",
        ) is None
        replacement = CommandDefinition(
            "fresh", "fresh replacement", lambda call: CommandResult("success", call.raw_input),
            aliases=("f",),
        )
        await commands.register(contexts[0], replacement)
        fourth = commands.freeze()
        assert [descriptor.name for descriptor in fourth.descriptors] == ["echo", "fresh"]
        rebound = await fourth.execute(
            "/f  rebound", session_key="chat", channel="test", chat_id="room", sender="user",
        )
        assert rebound is not None
        assert rebound.result == CommandResult("success", "  rebound")

        await provider_fiber.dispose()
        with pytest.raises(CompositionError) as error:
            commands.freeze()
        assert error.value.code == "INACTIVE_SERVICE"
        with pytest.raises(CompositionError) as error:
            await first.execute(
                "/echo original", session_key="chat", channel="test",
                chat_id="room", sender="user",
            )
        assert error.value.code == "OWNER_UNAVAILABLE"
    finally:
        cleanup_errors = []
        for composition in (foreign, root):
            try:
                await composition.dispose()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup("Commands view test cleanup failed", cleanup_errors)


@pytest.mark.asyncio
async def test_provider_view_rejects_stale_context_after_same_fiber_dependency_reactivation(tmp_path):
    dependency = ServiceKey("fixture.commands.provider_dependency")
    root = CompositionRoot("commands-stale-provider")

    def dependency_apply(value):
        async def apply(ctx):
            await ctx.provide(dependency, value)
        return apply

    async def provider_apply(ctx):
        assert ctx.require(dependency) in {"a", "b"}
        await commands_plugin.apply(ctx)

    async def contribute(ctx):
        await ctx.require(COMMANDS).register(
            ctx,
            CommandDefinition("echo", "echo input", lambda call: CommandResult("success", call.raw_input)),
        )

    try:
        dependency_fiber = await root.mount(dependency_apply("a"), name="commands-dependency")
        provider_fiber = await root.mount(
            provider_apply, name="human-actions", inject=(dependency,),
            runtime=PluginRuntime(
                plugin_id="human-actions", generation_id="human-actions-generation",
                plugin_dir=tmp_path, data_dir=tmp_path / "data",
                workspace=tmp_path, config={},
            ),
        )
        await root.mount(
            contribute, name="echo-owner", inject=(COMMANDS,),
            runtime=PluginRuntime(
                plugin_id="echo-owner", generation_id="echo-generation",
                plugin_dir=tmp_path, data_dir=tmp_path / "owner-data",
                workspace=tmp_path, config={},
            ),
        )
        old_context = provider_fiber.context
        old_token = old_context.fiber.activation_token
        old_commands = root.context.require(COMMANDS)
        old_view = old_commands.freeze()

        await dependency_fiber.dispose()
        assert provider_fiber.state is FiberState.PENDING
        await root.mount(dependency_apply("b"), name="commands-dependency")
        assert provider_fiber.state is FiberState.ACTIVE
        new_context = provider_fiber.context
        assert new_context is not old_context
        assert new_context.fiber.activation_token is not old_token

        with pytest.raises(CompositionError) as error:
            old_commands.freeze()
        assert error.value.code == "STALE_ACTIVATION"
        with pytest.raises(CompositionError) as error:
            await old_view.execute(
                "/echo input", session_key="chat", channel="test",
                chat_id="room", sender="user",
            )
        assert error.value.code == "STALE_ACTIVATION"

        new_view = root.context.require(COMMANDS).freeze()
        result = await new_view.execute(
            "/echo input", session_key="chat", channel="test",
            chat_id="room", sender="user",
        )
        assert result is not None
        assert result.result == CommandResult("success", " input")
    finally:
        await root.dispose()


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
    initialize_plugin_workspace(workspace)
    host = PluginManager([source], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=tmp_path / "home", message_log=log)
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
        archive = PluginArchive(workspace / "runtime/plugin-archives", create=False)
        root_ref = log.read_binding(identity)["root_ref"]
        assert isinstance(root_ref, str)
        descriptor = archive.read_descriptor(root_ref)
        archive_refs = []
        for name in ("human_actions", "command_owner", "dependency"):
            generation = host.generation(name)
            assert generation is not None
            archive_refs.append(generation.archive_ref)
        components = descriptor["components"]
        assert isinstance(components, tuple)
        assert set(components) == set(archive_refs)
        async with bindings.open(identity, COMMANDS) as (selected, metadata):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["provider", "contributor"])
async def test_execute_scopes_both_owners_and_rejects_new_known_calls_during_drain(tmp_path, owner):
    """已接纳命令持有两层 owner；排空只拒绝新命令，不抹成未知。"""
    root = CompositionRoot(f"commands-drain-{owner}")
    provider_fiber = contributor_fiber = peer_fiber = None
    contributor_context: Context | None = None
    peer_context: Context | None = None
    call_task = dispose_task = rejected_task = None
    commands = None
    release = asyncio.Event()
    child_gate = asyncio.Event()
    entered = asyncio.Event()
    child_observed = asyncio.Event()
    cleanup_started = asyncio.Event()
    provider_cleanup_started = asyncio.Event()
    peer_service = ServiceKey("fixture.unrelated.command.peer")
    peer_state = {"started": 0, "stopping": 0, "effect": 0, "cleanup": 0}
    calls = []
    captured_scopes = {}
    cleanup_counts = {"provider": 0, "contributor": 0}
    view_ref: CommandRegistry | None = None

    async def contribute(ctx):
        nonlocal contributor_context
        contributor_context = ctx

        async def handler(invocation):
            assert contributor_context is not None
            calls.append(invocation.message_id)
            assert provider_context._fiber._call_owned_by_current_task() is not None
            assert contributor_context._fiber._call_owned_by_current_task() is not None
            captured_scopes["provider"] = provider_context.capture_runtime_scope()
            captured_scopes["contributor"] = contributor_context.capture_runtime_scope()
            entered.set()
            await child_gate.wait()

            async def child_call():
                assert view_ref is not None
                with pytest.raises(CompositionError) as error:
                    await view_ref.execute(
                        "/p child", session_key="chat", channel="test",
                        chat_id="room", sender="user", message_id="child",
                    )
                assert error.value.code == "OWNER_UNAVAILABLE"
                child_observed.set()

            async with provider_context.runtime_scope():
                async with contributor_context.runtime_scope():
                    assert contributor_context.require(COMMANDS) is commands
                    assert contributor_context.require(COMMANDS) is provider_context.require(COMMANDS)
            await asyncio.create_task(child_call())
            await release.wait()
            return CommandResult("success", invocation.raw_input)

        await ctx.require(COMMANDS).register(
            ctx,
            CommandDefinition("probe", "probe command", handler, aliases=("p",)),
        )

        def cleanup():
            cleanup_counts["contributor"] += 1
            cleanup_started.set()

        await ctx.effect(lambda: cleanup, label="probe-hard-effect")

    async def provide_commands(ctx):
        await commands_plugin.apply(ctx)

        def cleanup():
            cleanup_counts["provider"] += 1
            provider_cleanup_started.set()

        await ctx.effect(lambda: cleanup, label="commands-hard-effect")

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

        await ctx.effect(setup, label="unrelated-command-peer")
        await ctx.on(RUNTIME_STARTED, started)
        await ctx.on(RUNTIME_STOPPING, stopping)
        await ctx.provide(peer_service, "peer")

    try:
        provider_fiber = await root.mount(provide_commands, name="commands")
        provider_context, commands = root._service_provider(COMMANDS)
        contributor_fiber = await root.mount(
            contribute,
            name="probe-owner",
            inject=(COMMANDS,),
            runtime=PluginRuntime(
                plugin_id="probe-owner", generation_id="probe-generation",
                plugin_dir=tmp_path, data_dir=tmp_path / "data",
                workspace=tmp_path, config={},
            ),
        )
        peer_fiber = await root.mount(peer_apply, name="unrelated-peer")
        assert contributor_context is not None and peer_context is not None
        assert provider_fiber is not None and contributor_fiber is not None and peer_fiber is not None
        view = commands.freeze()
        assert [descriptor.name for descriptor in view.descriptors] == ["probe"]
        view_ref = view
        peer_raw_fiber = peer_context._fiber
        peer_activation = peer_context.fiber.activation_token
        peer_effects = tuple(peer_raw_fiber.effects)
        peer_counts = peer_state.copy()

        async with asyncio.timeout(10):
            call_task = asyncio.create_task(view.execute(
                "/probe first", session_key="chat", channel="test",
                chat_id="room", sender="user", message_id="first",
            ))
            await entered.wait()
            target = provider_fiber if owner == "provider" else contributor_fiber
            dispose_task = asyncio.create_task(target.dispose())
            await captured_scopes[owner].wait_admission_closed()
            assert target.state is FiberState.UNLOADING
            for scope in captured_scopes.values():
                await scope.close()
            assert cleanup_counts[owner] == 0

            if owner == "contributor":
                draining_view = commands.freeze()
                assert [descriptor.name for descriptor in draining_view.descriptors] == ["probe"]
                view_ref = draining_view
            else:
                draining_view = view
            child_gate.set()
            await child_observed.wait()

            async with peer_context.runtime_scope():
                assert peer_context.require(peer_service) == "peer"
            assert not peer_raw_fiber._in_flight_calls
            assert peer_fiber.context is peer_context
            assert peer_context._fiber is peer_raw_fiber
            assert peer_context.fiber.activation_token is peer_activation
            assert tuple(peer_raw_fiber.effects) == peer_effects
            assert peer_state == peer_counts

            async def reject_new_call():
                with pytest.raises(CompositionError) as error:
                    await draining_view.execute(
                        "/p new", session_key="chat", channel="test",
                        chat_id="room", sender="user", message_id="new",
                    )
                assert error.value.code == "OWNER_UNAVAILABLE"

            rejected_task = asyncio.create_task(reject_new_call())
            await rejected_task
            assert calls == ["first"]
            assert not cleanup_started.is_set()
            assert not provider_cleanup_started.is_set()

            release.set()
            result = await call_task
            assert result is not None
            assert result.result == CommandResult("success", " first")
            for scope in captured_scopes.values():
                await scope.close()
            await dispose_task
            assert target.state is FiberState.DISPOSED
            assert cleanup_counts[owner] == 1
            assert (cleanup_started if owner == "contributor" else provider_cleanup_started).is_set()
            assert not provider_context._fiber._in_flight_calls
            assert not contributor_context._fiber._in_flight_calls

            if owner == "contributor":
                assert commands.freeze().descriptors == ()
                await root.mount(
                    contribute,
                    name="probe-owner-reloaded",
                    inject=(COMMANDS,),
                    runtime=PluginRuntime(
                        plugin_id="probe-owner-reloaded", generation_id="probe-generation-2",
                        plugin_dir=tmp_path, data_dir=tmp_path / "data-2",
                        workspace=tmp_path, config={},
                    ),
                )
                assert [descriptor.name for descriptor in commands.freeze().descriptors] == ["probe"]
            else:
                with pytest.raises(CompositionError) as error:
                    commands.freeze()
                assert error.value.code == "INACTIVE_SERVICE"
    finally:
        cleanup_errors = []
        release.set()
        child_gate.set()
        for task in (call_task, rejected_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (call_task, rejected_task):
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except BaseException as error:
                    cleanup_errors.append(error)
        for scope in captured_scopes.values():
            try:
                await scope.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if dispose_task is not None:
            try:
                await dispose_task
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            await root.dispose()
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup("Commands provider test cleanup failed", cleanup_errors)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", [
        "normal", "handler_error", "result_error", "cancel", "recover",
        "recover_error", "no_recover",
    ],
)
async def test_execute_releases_both_owner_scopes_for_success_error_cancel_and_recover(
    tmp_path, outcome,
):
    root = CompositionRoot(f"commands-outcome-{outcome}")
    provider_context = contributor_context = None
    command_task = None
    entered = asyncio.Event()
    release = asyncio.Event()
    handler_finished = asyncio.Event()
    handler_calls = []
    recover_calls = []
    health_calls = []

    async def contribute(ctx):
        nonlocal contributor_context
        contributor_context = ctx

        def check_owner():
            assert provider_context is not None
            assert provider_context._fiber._call_owned_by_current_task() is not None
            assert contributor_context._fiber._call_owned_by_current_task() is not None

        async def cancelled_handler(invocation):
            entered.set()
            try:
                await release.wait()
            finally:
                handler_finished.set()
            return CommandResult("success", invocation.raw_input)

        def handler(invocation):
            handler_calls.append(invocation.message_id)
            check_owner()
            if outcome == "normal":
                return CommandResult("success", invocation.raw_input)
            if outcome == "handler_error":
                raise OSError("handler failure")
            if outcome == "result_error":
                return cast(CommandResult, object())  # Inject an invalid runtime result.
            if outcome == "cancel":
                return cancelled_handler(invocation)
            raise AssertionError("recover path must not invoke the original handler")

        def recover(invocation):
            recover_calls.append(invocation.message_id)
            check_owner()
            if outcome == "recover_error":
                raise RuntimeError("recover failure")
            return CommandResult("success", " recovered")

        await ctx.require(COMMANDS).register(
            ctx,
            CommandDefinition(
                "probe", "probe command", handler,
                read_only=outcome == "normal",
                recover=recover if outcome in {"recover", "recover_error"} else None,
            ),
        )

        def health(invocation):
            health_calls.append(invocation.message_id)
            check_owner()
            return CommandResult("success", invocation.raw_input)

        await ctx.require(COMMANDS).register(
            ctx,
            CommandDefinition("health", "health command", health, read_only=True),
        )

    try:
        await root.mount(commands_plugin.apply, name="commands")
        provider_context, commands = root._service_provider(COMMANDS)
        await root.mount(
            contribute,
            name="probe-owner",
            inject=(COMMANDS,),
            runtime=PluginRuntime(
                plugin_id="probe-owner", generation_id="probe-generation",
                plugin_dir=tmp_path, data_dir=tmp_path / "data",
                workspace=tmp_path, config={},
            ),
        )
        view = commands.freeze()
        if outcome == "normal":
            result = await view.execute(
                "/probe input", session_key="chat", channel="test",
                chat_id="room", sender="user", message_id="normal",
            )
            assert result is not None
            assert result.result == CommandResult("success", " input")
        elif outcome == "handler_error":
            with pytest.raises(OSError, match="handler failure"):
                await view.execute(
                    "/probe input", session_key="chat", channel="test",
                    chat_id="room", sender="user", message_id="error",
                )
        elif outcome == "result_error":
            with pytest.raises(TypeError, match="Command .* handler 必须返回"):
                await view.execute(
                    "/probe input", session_key="chat", channel="test",
                    chat_id="room", sender="user", message_id="bad-result",
                )
        elif outcome == "cancel":
            async with asyncio.timeout(10):
                command_task = asyncio.create_task(view.execute(
                    "/probe input", session_key="chat", channel="test",
                    chat_id="room", sender="user", message_id="cancel",
                ))
                await entered.wait()
                command_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await command_task
            release.set()
            assert handler_finished.is_set()
        elif outcome == "recover":
            result = await view.execute(
                "/probe input", session_key="chat", channel="test",
                chat_id="room", sender="user", message_id="recover", recover=True,
            )
            assert result is not None
            assert result.result == CommandResult("success", " recovered")
        elif outcome == "recover_error":
            with pytest.raises(RuntimeError, match="recover failure"):
                await view.execute(
                    "/probe input", session_key="chat", channel="test",
                    chat_id="room", sender="user", message_id="recover-error", recover=True,
                )
        else:
            with pytest.raises(CommandRecoveryRequired, match="禁止自动重跑"):
                await view.execute(
                    "/probe input", session_key="chat", channel="test",
                    chat_id="room", sender="user", message_id="no-recover", recover=True,
                )
        expected_message = {
            "normal": "normal",
            "handler_error": "error",
            "result_error": "bad-result",
            "cancel": "cancel",
        }
        assert handler_calls == (
            [] if outcome in {"recover", "recover_error", "no_recover"}
            else [expected_message[outcome]]
        )
        assert len(recover_calls) == int(outcome in {"recover", "recover_error"})
        if outcome in {"handler_error", "result_error", "cancel", "recover_error"}:
            health_result = await view.execute(
                "/health input", session_key="chat", channel="test",
                chat_id="room", sender="user", message_id="health",
            )
            assert health_result is not None
            assert health_result.result == CommandResult("success", " input")
            assert health_calls == ["health"]
        else:
            assert health_calls == []
        assert not provider_context._fiber._in_flight_calls
        assert not contributor_context._fiber._in_flight_calls
    finally:
        release.set()
        if command_task is not None and not command_task.done():
            command_task.cancel()
        cleanup_errors = []
        if command_task is not None:
            try:
                await command_task
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            await root.dispose()
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup("Commands outcome test cleanup failed", cleanup_errors)

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pytest

import agent.plugin_composition as plugin_composition

from agent.plugin_composition import (
    CompositionAudit,
    CompositionError,
    CompositionOverlay,
    CompositionRoot,
    EmitEventKey,
    Fiber,
    FiberState,
    HealthHandle,
    ParallelEventKey,
    PluginRuntime,
    ServiceKey,
)
from agent.plugin_composition.access import ExternalEffectGate, PluginDataAccess
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore
from agent.plugins.manager import PluginManager

GREETING = ServiceKey[str]("greeting")
FORMATTER = ServiceKey[Callable[[str], str]]("formatter")


def _runtime(tmp_path: Path, plugin_id: str) -> PluginRuntime:
    return PluginRuntime(
        plugin_id=plugin_id,
        generation_id=f"generation:{plugin_id}",
        plugin_dir=tmp_path / "plugins" / plugin_id,
        data_dir=tmp_path / "plugin-data" / plugin_id,
        workspace=tmp_path,
        config=None,
    )


class GreetingProvider:
    name = "greeting-provider"
    inject = ()

    def __init__(self, value: str = "hello") -> None:
        self.value = value

    async def apply(self, ctx) -> None:
        await ctx.provide(GREETING, self.value)


async def _mount_greeting(
    root: CompositionRoot,
    value: str = "hello",
    *,
    name: str = "greeting-provider",
) -> Fiber:
    plugin = GreetingProvider(value)
    return await root.mount(plugin.apply, name=name)


@pytest.mark.asyncio
async def test_overlay_topology_matches_formal_event_key_order(tmp_path: Path) -> None:
    first = EmitEventKey[str]("fixture.first")
    second = EmitEventKey[str]("fixture.second")

    async def mount(root: CompositionRoot, plugin_id: str, keys) -> None:
        async def apply(ctx) -> None:
            for key in keys:
                await ctx.on(key, lambda _payload: None)

        _ = await root.mount(
            apply,
            name=plugin_id,
            runtime=_runtime(tmp_path, plugin_id),
        )

    stable = CompositionRoot("stable")
    candidate = CompositionRoot("candidate")
    formal = CompositionRoot("formal")
    await mount(stable, "a", (first, second))
    await mount(stable, "b", (second, first))
    await mount(candidate, "b", (second, first))
    await mount(formal, "a", (first, second))
    await mount(formal, "b", (second, first))
    overlay = CompositionOverlay(
        stable,
        candidate,
        plugin_ids=frozenset({"a", "b"}),
        replaced_plugin_ids=frozenset({"b"}),
    )

    assert overlay.topology_view().listeners == formal.topology_view().listeners
    assert overlay.topology_identity() == formal.topology_identity()

    await overlay.dispose()
    await stable.dispose()
    await formal.dispose()


@pytest.mark.asyncio
async def test_overlay_keeps_unchanged_stable_listener_order(tmp_path: Path) -> None:
    event = EmitEventKey[str]("fixture.shared")

    async def mount(root: CompositionRoot, plugin_id: str, *, listen: bool) -> None:
        async def apply(ctx) -> None:
            if listen:
                await ctx.on(event, lambda _payload: None)

        _ = await root.mount(
            apply,
            name=plugin_id,
            runtime=_runtime(tmp_path, plugin_id),
        )

    stable = CompositionRoot("stable")
    candidate = CompositionRoot("candidate")
    formal = CompositionRoot("formal")
    for root in (stable, formal):
        await mount(root, "models", listen=True)
        await mount(root, "akasha", listen=True)
    await mount(candidate, "restart_probe", listen=False)
    await mount(formal, "restart_probe", listen=False)
    overlay = CompositionOverlay(
        stable,
        candidate,
        plugin_ids=frozenset({"akasha", "models", "restart_probe"}),
        replaced_plugin_ids=frozenset({"restart_probe"}),
    )

    assert overlay.topology_view().listeners == formal.topology_view().listeners
    assert overlay.topology_identity() == formal.topology_identity()

    await overlay.dispose()
    await stable.dispose()
    await formal.dispose()


@pytest.mark.asyncio
async def test_overlay_keeps_candidate_parallel_listeners_concurrent(
    tmp_path: Path,
) -> None:
    event = ParallelEventKey[None]("fixture.parallel")
    both_started = asyncio.Event()
    started = 0

    async def mount(root: CompositionRoot, plugin_id: str) -> None:
        async def listener(_payload: None) -> None:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=0.2)

        async def apply(ctx) -> None:
            await ctx.on(event, listener)

        _ = await root.mount(
            apply,
            name=plugin_id,
            runtime=_runtime(tmp_path, plugin_id),
        )

    stable = CompositionRoot("stable")
    candidate = CompositionRoot("candidate")
    await mount(candidate, "a")
    await mount(candidate, "b")
    overlay = CompositionOverlay(
        stable,
        candidate,
        plugin_ids=frozenset({"a", "b"}),
        replaced_plugin_ids=frozenset({"a", "b"}),
    )

    await overlay.context.parallel(event, None)

    assert started == 2
    await overlay.dispose()
    await stable.dispose()


@pytest.mark.asyncio
async def test_overlay_event_dispatch_only_reaches_candidate_plugins(
    tmp_path: Path,
) -> None:
    stable_events: list[str] = []
    candidate_events: list[str] = []

    async def mount_lifecycle(
        root: CompositionRoot,
        plugin_id: str,
        events: list[str],
    ) -> None:
        from agent.plugin_composition import RUNTIME_STARTED, RUNTIME_STOPPING

        async def apply(ctx) -> None:
            await ctx.on(RUNTIME_STARTED, lambda _event: events.append("started"))
            await ctx.on(RUNTIME_STOPPING, lambda _event: events.append("stopping"))

        _ = await root.mount(
            apply,
            name=plugin_id,
            runtime=_runtime(tmp_path, plugin_id),
        )

    stable = CompositionRoot("stable")
    candidate = CompositionRoot("candidate")
    await mount_lifecycle(stable, "stable", stable_events)
    await mount_lifecycle(candidate, "candidate", candidate_events)
    overlay = CompositionOverlay(
        stable,
        candidate,
        plugin_ids=frozenset({"stable", "candidate"}),
        replaced_plugin_ids=frozenset({"candidate"}),
    )
    from agent.plugin_composition import (
        RUNTIME_STARTED,
        RUNTIME_STOPPING,
        RuntimeStarted,
        RuntimeStopping,
    )

    assert await overlay.context.serial(RUNTIME_STARTED, RuntimeStarted()) is None
    assert await overlay.context.serial(RUNTIME_STOPPING, RuntimeStopping()) is None

    assert stable_events == []
    assert candidate_events == ["started", "stopping"]
    await overlay.dispose()
    await stable.dispose()


def test_internal_access_helpers_are_not_v3_public_exports() -> None:
    assert not hasattr(plugin_composition, "ExternalEffectGate")
    assert not hasattr(plugin_composition, "PluginDataAccess")
    assert not hasattr(plugin_composition, "ScopedPluginData")


@pytest.mark.asyncio
async def test_data_root_is_core_assigned_and_shared_by_nested_fibers(tmp_path) -> None:
    data_root = tmp_path / "plugin-data" / "probe-builtin"
    data_root.mkdir(parents=True)
    runtime = PluginRuntime(
        plugin_id="probe@builtin",
        generation_id="test-generation",
        plugin_dir=tmp_path / "plugin",
        data_dir=data_root,
        workspace=tmp_path,
        config=None,
    )
    observed = []

    async def child(ctx) -> None:
        observed.append(ctx.data_root)

    async def parent(ctx) -> None:
        observed.append(ctx.data_root)
        _ = await ctx.mount(child, name="child")

    root = CompositionRoot("data-root")
    _ = await root.mount(parent, name="parent", runtime=runtime)

    assert observed == [data_root, data_root]


@pytest.mark.asyncio
async def test_workspace_root_is_declared_and_shared_by_nested_fibers(tmp_path) -> None:
    memes = tmp_path / "memes"
    memes.mkdir()
    runtime = PluginRuntime(
        plugin_id="probe@builtin",
        generation_id="test-generation",
        plugin_dir=tmp_path / "plugin",
        data_dir=tmp_path / "plugin-data" / "probe-builtin",
        workspace=tmp_path,
        config=None,
        workspace_roots=("memes",),
    )
    observed = []

    async def child(ctx) -> None:
        observed.append(ctx.workspace_root("memes"))

    async def parent(ctx) -> None:
        observed.append(ctx.workspace_root("memes"))
        with pytest.raises(CompositionError) as caught:
            _ = ctx.workspace_root("attachments")
        assert caught.value.code == "WORKSPACE_ROOT_UNDECLARED"
        _ = await ctx.mount(child, name="child")

    root = CompositionRoot("workspace-root")
    _ = await root.mount(parent, name="parent", runtime=runtime)

    assert observed == [memes.resolve(), memes.resolve()]


def test_data_root_requires_core_assigned_plugin_runtime() -> None:
    root = CompositionRoot("missing-data-root")

    with pytest.raises(CompositionError) as caught:
        _ = root.context.data_root

    assert caught.value.code == "PLUGIN_RUNTIME_UNAVAILABLE"


@pytest.mark.asyncio
async def test_public_mount_rejects_object_apply_abi() -> None:
    class LegacyPlugin:
        async def apply(self, _ctx) -> None:
            return None

    root = CompositionRoot("callable-only")
    legacy = LegacyPlugin()
    with pytest.raises(TypeError, match="callable"):
        await root.mount(legacy)  # type: ignore[arg-type]

    async def parent(ctx) -> None:
        with pytest.raises(TypeError, match="child callable"):
            await ctx.mount(legacy)  # type: ignore[arg-type]

    _ = await root.mount(parent, name="parent")


@pytest.mark.asyncio
async def test_required_dependency_follows_provider_lifecycle() -> None:
    events: list[str] = []

    class Consumer:
        name = "consumer"
        inject = (GREETING,)

        async def apply(self, ctx) -> None:
            events.append(f"load:{ctx.require(GREETING)}")
            await ctx.effect(
                lambda: lambda: events.append("unload"),
                label="consumer",
            )

    root = CompositionRoot("required-lifecycle")
    consumer_plugin = Consumer()
    consumer = await root.mount(
        consumer_plugin.apply,
        name=consumer_plugin.name,
        inject=consumer_plugin.inject,
    )
    assert consumer.state == FiberState.PENDING
    assert root.receipt().required_pending == ("consumer",)

    first_plugin = GreetingProvider("first")
    first_provider = await root.mount(first_plugin.apply, name=first_plugin.name)
    assert consumer.state == FiberState.ACTIVE
    assert events == ["load:first"]
    assert root.receipt().ready is True

    await first_provider.dispose()
    assert consumer.state == FiberState.PENDING
    assert events == ["load:first", "unload"]

    second_plugin = GreetingProvider("second")
    await root.mount(second_plugin.apply, name=second_plugin.name)
    assert consumer.state == FiberState.ACTIVE
    assert events == ["load:first", "unload", "load:second"]


@pytest.mark.asyncio
async def test_nested_inject_is_optional_for_candidate_readiness() -> None:
    events: list[str] = []

    class Parent:
        name = "parent"
        inject = ()

        async def apply(self, ctx) -> None:
            async def use_formatter(inner) -> None:
                formatter = inner.require(FORMATTER)
                events.append(formatter("ready"))

            await ctx.inject((FORMATTER,), use_formatter, name="optional-formatter")

    root = CompositionRoot("optional-inject")
    parent_plugin = Parent()
    parent = await root.mount(parent_plugin.apply, name=parent_plugin.name)
    receipt = root.receipt()
    assert parent.state == FiberState.ACTIVE
    assert receipt.ready is True
    assert receipt.optional_pending == ("optional-formatter",)

    class FormatterProvider:
        name = "formatter-provider"
        inject = ()

        async def apply(self, ctx) -> None:
            await ctx.provide(FORMATTER, lambda value: value.upper())

    formatter_plugin = FormatterProvider()
    await root.mount(formatter_plugin.apply, name=formatter_plugin.name)
    assert events == ["READY"]
    assert root.receipt().optional_pending == ()


@pytest.mark.asyncio
async def test_effect_cleanup_is_lifo_and_public_dispose_is_single_shot() -> None:
    events: list[str] = []
    root = CompositionRoot("effect-lifo")

    async def apply(ctx) -> None:
        effect = await ctx.effect(
            lambda: (
                lambda: events.append("first"),
                lambda: events.append("second"),
            ),
            label="pair",
        )
        await effect.aclose()
        await effect.aclose()

    fiber = await root.mount(apply, name="effect-owner")
    assert fiber.state == FiberState.ACTIVE
    assert events == ["second", "first"]
    assert fiber.effects == []


@pytest.mark.asyncio
async def test_effect_setup_failure_rolls_back_collected_cleanup() -> None:
    cleanups: list[str] = []
    root = CompositionRoot("effect-rollback")

    def broken_setup():
        yield lambda: cleanups.append("rolled-back")
        raise RuntimeError("setup failed")

    async def apply(ctx) -> None:
        await ctx.effect(broken_setup, label="broken")

    fiber = await root.mount(apply, name="broken-plugin")
    assert fiber.state == FiberState.FAILED
    assert cleanups == ["rolled-back"]
    assert fiber.effects == []
    assert root.receipt().ready is False


@pytest.mark.asyncio
async def test_reentrant_dispose_awaits_setup_and_async_cleanup() -> None:
    setup_gate = asyncio.Event()
    cleanup_gate = asyncio.Event()
    cleanup_started = asyncio.Event()
    dispose_created = asyncio.Event()
    dispose_task: asyncio.Task[None] | None = None
    root = CompositionRoot("reentrant-dispose")

    async def apply(ctx) -> None:
        async def setup():
            nonlocal dispose_task
            dispose_task = asyncio.create_task(ctx.fiber.dispose())
            dispose_created.set()
            await setup_gate.wait()

            async def cleanup() -> None:
                cleanup_started.set()
                await cleanup_gate.wait()

            return cleanup

        await ctx.effect(setup, label="reentrant")

    mount_task = asyncio.create_task(root.mount(apply, name="owner"))
    await dispose_created.wait()
    assert dispose_task is not None
    assert dispose_task.done() is False
    setup_gate.set()
    await cleanup_started.wait()
    assert dispose_task.done() is False
    cleanup_gate.set()
    await dispose_task
    fiber = await mount_task
    assert fiber.state == FiberState.DISPOSED
    assert fiber.effects == []


@pytest.mark.asyncio
async def test_reentrant_restart_awaits_setup_cleanup_and_reloads() -> None:
    setup_gate = asyncio.Event()
    cleanup_gate = asyncio.Event()
    cleanup_started = asyncio.Event()
    restart_created = asyncio.Event()
    restart_task: asyncio.Task[None] | None = None
    apply_calls = 0
    root = CompositionRoot("reentrant-restart")

    async def apply(ctx) -> None:
        nonlocal apply_calls, restart_task
        apply_calls += 1
        if apply_calls != 1:
            return

        async def setup():
            nonlocal restart_task
            restart_task = asyncio.create_task(ctx.fiber.restart())
            restart_created.set()
            await setup_gate.wait()

            async def cleanup() -> None:
                cleanup_started.set()
                await cleanup_gate.wait()

            return cleanup

        _ = await ctx.effect(setup, label="reentrant")

    mount_task = asyncio.create_task(root.mount(apply, name="owner"))
    await restart_created.wait()
    assert restart_task is not None
    setup_gate.set()
    await cleanup_started.wait()
    assert restart_task.done() is False
    cleanup_gate.set()
    await restart_task
    fiber = await mount_task
    assert fiber.state == FiberState.ACTIVE
    assert fiber.effects == []
    assert apply_calls == 2


@pytest.mark.asyncio
async def test_direct_reentrant_lifecycle_wait_fails_loud_instead_of_deadlock() -> None:
    observed: list[str] = []
    root = CompositionRoot("direct-reentrant-wait")

    async def apply(ctx) -> None:
        for operation in (ctx.fiber.dispose, ctx.fiber.restart):
            with pytest.raises(CompositionError) as caught:
                await operation()
            observed.append(caught.value.code)

    fiber = await asyncio.wait_for(root.mount(apply, name="owner"), timeout=0.5)
    assert observed == [
        "REENTRANT_LIFECYCLE_WAIT",
        "REENTRANT_LIFECYCLE_WAIT",
    ]
    assert fiber.state == FiberState.ACTIVE


@pytest.mark.asyncio
async def test_unloading_rejects_new_effect_registration() -> None:
    errors: list[CompositionError] = []
    root = CompositionRoot("inactive-effect")

    async def apply(ctx) -> None:
        async def cleanup() -> None:
            try:
                await ctx.effect(lambda: None, label="too-late")
            except CompositionError as error:
                errors.append(error)

        await ctx.effect(lambda: cleanup, label="owner")

    fiber = await root.mount(apply, name="plugin")
    await fiber.restart()
    assert [error.code for error in errors] == ["INACTIVE_EFFECT"]
    assert fiber.state == FiberState.ACTIVE


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_truncate_disposal() -> None:
    cleanup_started = asyncio.Event()
    cleanup_gate = asyncio.Event()
    cleanup_finished = False
    root = CompositionRoot("cancel-safe-cleanup")

    async def apply(ctx) -> None:
        async def cleanup() -> None:
            nonlocal cleanup_finished
            cleanup_started.set()
            await cleanup_gate.wait()
            cleanup_finished = True

        await ctx.effect(lambda: cleanup, label="slow-cleanup")

    fiber = await root.mount(apply, name="owner")
    caller = asyncio.create_task(fiber.dispose())
    await cleanup_started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    assert caller.done() is False
    cleanup_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert cleanup_finished is True
    assert fiber.state == FiberState.DISPOSED
    assert root.receipt().effects == ()


@pytest.mark.asyncio
async def test_mount_cancellation_rolls_back_published_fiber() -> None:
    apply_started = asyncio.Event()
    cleanup_finished = False
    root = CompositionRoot("cancelled-mount")

    async def apply(ctx) -> None:
        nonlocal cleanup_finished
        await ctx.effect(
            lambda: lambda: _mark_cleanup(),
            label="mount-cleanup",
        )
        apply_started.set()
        await asyncio.Event().wait()

    def _mark_cleanup() -> None:
        nonlocal cleanup_finished
        cleanup_finished = True

    mount_task = asyncio.create_task(root.mount(apply, name="cancelled"))
    await apply_started.wait()
    mount_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await mount_task
    assert cleanup_finished is True
    assert root.receipt().fibers == ()
    assert root.root_fiber.children == []


@pytest.mark.asyncio
async def test_stale_dependency_epoch_never_becomes_active() -> None:
    apply_started = asyncio.Event()
    apply_gate = asyncio.Event()
    events: list[str] = []
    root = CompositionRoot("stale-epoch")
    provider = await _mount_greeting(root)

    async def consume(ctx) -> None:
        events.append(f"start:{ctx.require(GREETING)}")
        apply_started.set()
        await apply_gate.wait()
        await ctx.effect(lambda: lambda: events.append("cleanup"), label="owned")

    mount_task = asyncio.create_task(
        root.mount(consume, name="slow-consumer", inject=(GREETING,))
    )
    await apply_started.wait()
    remove_task = asyncio.create_task(provider.dispose())
    apply_gate.set()
    consumer = await mount_task
    await remove_task
    assert consumer.state == FiberState.PENDING
    assert events == ["start:hello", "cleanup"]


@pytest.mark.asyncio
async def test_mount_observer_failure_rolls_back_parent_ownership() -> None:
    root = CompositionRoot("publication-rollback")

    def fail_publication(fiber) -> None:
        if fiber.name == "broken-publication":
            raise RuntimeError("publication failed")

    root.on_mount(fail_publication)
    with pytest.raises(RuntimeError, match="publication failed"):
        await root.mount(lambda _: None, name="broken-publication")
    assert root.root_fiber.children == []
    assert root.receipt().fibers == ()


@pytest.mark.asyncio
async def test_parent_disposal_during_publication_drains_pending_child() -> None:
    root = CompositionRoot("parent-child-quiescence")
    owner_context = None
    cleanup_started = asyncio.Event()
    cleanup_gate = asyncio.Event()
    child_apply_calls = 0
    parent_dispose_task: asyncio.Task[None] | None = None

    async def owner_apply(ctx) -> None:
        nonlocal owner_context
        owner_context = ctx

    owner = await root.mount(owner_apply, name="owner")

    async def observe_child(fiber) -> None:
        nonlocal parent_dispose_task
        if fiber.name != "child":
            return

        async def cleanup() -> None:
            cleanup_started.set()
            await cleanup_gate.wait()

        _ = await fiber.context.effect(lambda: cleanup, label="pending-child")
        parent_dispose_task = asyncio.create_task(owner.dispose())
        await cleanup_started.wait()

    root.on_mount(observe_child)

    async def child_apply(_) -> None:
        nonlocal child_apply_calls
        child_apply_calls += 1

    assert owner_context is not None
    child_mount = asyncio.create_task(owner_context.mount(child_apply, name="child"))
    await cleanup_started.wait()
    assert child_apply_calls == 0
    assert parent_dispose_task is not None
    assert parent_dispose_task.done() is False
    cleanup_gate.set()
    child = await child_mount
    await parent_dispose_task
    assert child.state == FiberState.DISPOSED
    assert owner.state == FiberState.DISPOSED


@pytest.mark.asyncio
async def test_dispose_observer_failure_is_contained_and_peers_run() -> None:
    observed: list[str] = []
    root = CompositionRoot("observer-containment")

    def broken(fiber) -> None:
        if fiber.name == "child":
            raise RuntimeError("broken observer")

    root.on_dispose(broken)
    root.on_dispose(lambda fiber: observed.append(fiber.name))
    child = await root.mount(lambda _: None, name="child")
    await child.dispose()
    assert observed == ["child"]
    assert any(
        "broken observer" in incident.message for incident in root.receipt().incidents
    )
    assert root.receipt().ready is True


@pytest.mark.asyncio
async def test_duplicate_provider_fails_without_replacing_first_owner() -> None:
    root = CompositionRoot("duplicate-service")
    first = await _mount_greeting(root, "first")
    duplicate = await _mount_greeting(root, "second", name="second-provider")
    assert first.state == FiberState.ACTIVE
    assert duplicate.state == FiberState.FAILED
    assert root.context.require(GREETING) == "first"
    assert root.receipt().services == ("greeting",)


@pytest.mark.asyncio
async def test_root_disposal_drains_children_effects_and_services() -> None:
    root = CompositionRoot("root-dispose")
    await _mount_greeting(root)
    await root.dispose()
    receipt = root.receipt()
    assert receipt.fibers == ()
    assert receipt.services == ()
    assert receipt.effects == ()
    assert receipt.ready is False


@pytest.mark.asyncio
async def test_root_disposal_rejects_mount_during_slow_cleanup() -> None:
    cleanup_started = asyncio.Event()
    cleanup_gate = asyncio.Event()
    root = CompositionRoot("root-dispose-mount-race")

    async def cleanup() -> None:
        cleanup_started.set()
        await cleanup_gate.wait()

    _ = await root.context.effect(lambda: cleanup, label="slow-root-cleanup")
    dispose_task = asyncio.create_task(root.dispose())
    await cleanup_started.wait()
    with pytest.raises(CompositionError) as caught:
        await root.mount(lambda _: None, name="too-late")
    assert caught.value.code == "INACTIVE_PLUGIN_OWNER"
    cleanup_gate.set()
    await dispose_task
    assert root.receipt().fibers == ()


@pytest.mark.asyncio
async def test_provider_cleanup_survives_all_dependent_cleanup_failures() -> None:
    cleaned: list[str] = []
    root = CompositionRoot("dependent-cleanup-failure")
    provider = await _mount_greeting(root)

    async def consume(ctx) -> None:
        name = ctx.fiber.name

        def fail_cleanup() -> None:
            cleaned.append(name)
            raise RuntimeError(f"cleanup failed: {name}")

        _ = await ctx.effect(lambda: fail_cleanup, label=f"cleanup:{name}")

    _ = await root.mount(consume, name="consumer-a", inject=(GREETING,))
    _ = await root.mount(consume, name="consumer-b", inject=(GREETING,))
    with pytest.raises(BaseExceptionGroup, match="Fiber 卸载失败"):
        await provider.dispose()

    assert sorted(cleaned) == ["consumer-a", "consumer-b"]
    assert provider.state == FiberState.DISPOSED
    assert root.context.get(GREETING) is None
    replacement = await _mount_greeting(root, name="replacement-provider")
    assert replacement.state == FiberState.ACTIVE
    assert root.context.require(GREETING) == "hello"


@pytest.mark.asyncio
async def test_observer_cancelled_error_is_contained_when_owner_not_cancelled() -> None:
    observed: list[str] = []
    root = CompositionRoot("observer-cancelled-error")

    async def cancelled(_fiber) -> None:
        raise asyncio.CancelledError("observer cancelled itself")

    root.on_dispose(cancelled)
    root.on_dispose(lambda fiber: observed.append(fiber.name))
    child = await root.mount(lambda _: None, name="child")
    await child.dispose()
    assert observed == ["child"]
    assert any(
        incident.error_type == "CancelledError" for incident in root.receipt().incidents
    )
    assert root.receipt().ready is True


@pytest.mark.asyncio
async def test_explicit_health_and_incident_are_independent_facts() -> None:
    root = CompositionRoot("health-incident")
    handles: list[HealthHandle] = []

    async def plugin(ctx) -> None:
        handles.append(await ctx.health("poller", required=True))
        _ = ctx.report_incident("poll_failed", "first attempt failed")

    await root.mount(plugin, name="watcher")
    initial = root.receipt()
    assert initial.ready is True
    assert initial.required_degraded == ()
    assert initial.incident_sequence == 1

    handles[0].degrade("upstream unavailable")
    degraded = root.receipt()
    assert degraded.ready is False
    assert degraded.required_degraded == ("watcher:poller",)
    assert degraded.incident_sequence == 1

    handles[0].recover()
    recovered = root.receipt()
    assert recovered.ready is True
    assert recovered.required_degraded == ()
    assert recovered.incidents == initial.incidents
    assert "watcher:health:poller" in recovered.effects
    _ = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    await root.dispose()
    with pytest.raises(CompositionError) as caught:
        handles[0].recover()
    assert caught.value.code == "INACTIVE_HEALTH"


@pytest.mark.asyncio
async def test_fiber_restart_invalidates_old_health_handle() -> None:
    root = CompositionRoot("health-restart")
    handles: list[HealthHandle] = []

    async def plugin(ctx) -> None:
        handles.append(await ctx.health("worker", required=True))

    fiber = await root.mount(plugin, name="plugin")
    first = handles[0]
    first.degrade("first epoch failed")

    await fiber.restart()

    assert len(handles) == 2
    assert root.receipt().required_degraded == ()
    with pytest.raises(CompositionError) as caught:
        first.recover()
    assert caught.value.code == "INACTIVE_HEALTH"
    handles[1].degrade("second epoch failed")
    assert root.receipt().required_degraded == ("plugin:worker",)
    await root.dispose()


@pytest.mark.asyncio
async def test_optional_health_degradation_does_not_block_readiness() -> None:
    root = CompositionRoot("optional-health")
    handles: list[HealthHandle] = []

    async def plugin(ctx) -> None:
        handles.append(await ctx.health("telemetry", required=False))

    await root.mount(plugin, name="observer")
    handles[0].degrade("metrics endpoint unavailable")

    receipt = root.receipt()
    assert receipt.ready is True
    assert receipt.required_degraded == ()
    assert receipt.health[0].healthy is False
    await root.dispose()


@pytest.mark.asyncio
async def test_optional_fiber_failure_records_incident_without_poisoning_root() -> None:
    root = CompositionRoot("optional-failure")

    async def broken(_) -> None:
        raise RuntimeError("optional failed")

    _ = await root.context.inject((), broken, name="optional")

    receipt = root.receipt()
    assert receipt.ready is True
    assert receipt.optional_pending == ("optional",)
    assert any(incident.message == "optional failed" for incident in receipt.incidents)
    await root.dispose()


@pytest.mark.asyncio
async def test_candidate_incident_overflow_fails_loud() -> None:
    root = CompositionRoot("candidate-incidents", candidate_incident_limit=2)
    _ = await root.mount(lambda _: None, name="plugin")

    for index in range(3):
        _ = root.context.report_incident("probe", f"failure {index}")

    receipt = root.receipt()
    assert receipt.ready is False
    assert receipt.incident_sequence == 3
    assert receipt.incident_overflowed is True
    assert tuple(item.sequence for item in receipt.incidents) == (1, 2)
    with pytest.raises(RuntimeError, match="incident_overflowed=True"):
        RuntimeSnapshotCompiler().compile({}, composition_root=root)
    await root.dispose()


@pytest.mark.asyncio
async def test_stable_incident_buffer_is_bounded_without_poisoning_health() -> None:
    root = CompositionRoot("stable-incidents")
    _ = await root.mount(lambda _: None, name="plugin")

    for index in range(root.RECENT_INCIDENT_LIMIT + 2):
        _ = root.context.report_incident("probe", f"failure {index}")

    receipt = root.receipt()
    assert receipt.ready is True
    assert receipt.incident_overflowed is False
    assert receipt.incident_sequence == root.RECENT_INCIDENT_LIMIT + 2
    assert len(receipt.incidents) == root.RECENT_INCIDENT_LIMIT
    assert receipt.incidents[0].sequence == 3
    await root.dispose()


@pytest.mark.asyncio
async def test_incident_after_candidate_seal_invalidates_validation_receipt() -> None:
    root = CompositionRoot("sealed-incidents", candidate_incident_limit=8)
    _ = await root.mount(lambda _: None, name="plugin")
    compiler = RuntimeSnapshotCompiler()
    candidate = compiler.compile({}, composition_root=root)
    store = RuntimeSnapshotStore()
    store.install(compiler.compile({}))
    transaction = store.begin_publish(candidate)
    await store.commit_latest(transaction)
    store.pause_candidate_admission(candidate)
    store.seal_candidate_validation(candidate)

    _ = root.context.report_incident("late_failure", "after seal")

    with pytest.raises(RuntimeError, match="验证回执在封存后发生变化"):
        await store.promote_latest()
    _ = await store.discard_latest(candidate)
    await store.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_rebuilt_formal_root_health_is_rechecked_after_candidate_seal() -> None:
    compiler = RuntimeSnapshotCompiler()
    validation_root = CompositionRoot("validation-root")
    _ = await validation_root.mount(lambda _: None, name="plugin")
    candidate = compiler.compile({}, composition_root=validation_root)
    store = RuntimeSnapshotStore()
    store.install(compiler.compile({}))
    await store.commit_latest(store.begin_publish(candidate))
    store.pause_candidate_admission(candidate)
    store.seal_candidate_validation(candidate)

    formal_root = CompositionRoot("formal-root")
    handles: list[HealthHandle] = []

    async def plugin(ctx) -> None:
        handles.append(await ctx.health("worker", required=True))

    _ = await formal_root.mount(plugin, name="plugin")
    formal_snapshot = compiler.compile({}, composition_root=formal_root)
    candidate.composition_root = formal_root
    candidate.composition_topology = formal_snapshot.composition_topology
    handles[0].degrade("formal worker unavailable")

    with pytest.raises(RuntimeError, match="required_degraded"):
        await store.promote_latest()

    _ = await store.discard_latest(candidate)
    await store.close()
    await validation_root.dispose()
    await formal_root.dispose()


@pytest.mark.asyncio
async def test_reused_stable_root_must_still_be_ready() -> None:
    compiler = RuntimeSnapshotCompiler()
    validation_root = CompositionRoot("validation-root")
    _ = await validation_root.mount(lambda _: None, name="plugin")
    candidate = compiler.compile({}, composition_root=validation_root)
    store = RuntimeSnapshotStore()
    store.install(compiler.compile({}))
    await store.commit_latest(store.begin_publish(candidate))
    store.pause_candidate_admission(candidate)
    store.seal_candidate_validation(candidate)

    reused_stable_root = CompositionRoot("reused-stable")
    _ = await reused_stable_root.mount(
        lambda _: None,
        name="missing-consumer",
        inject=(GREETING,),
    )
    reused_snapshot = compiler.compile(
        {},
        composition_root=reused_stable_root,
        require_composition_ready=False,
    )
    candidate.composition_root = reused_stable_root
    candidate.composition_topology = reused_snapshot.composition_topology

    with pytest.raises(RuntimeError, match="required_pending"):
        await store.promote_latest()

    _ = await store.discard_latest(candidate)
    await store.close()
    await validation_root.dispose()
    await reused_stable_root.dispose()


@pytest.mark.asyncio
async def test_snapshot_compiler_rejects_unready_required_topology() -> None:
    root = CompositionRoot("candidate-unready")
    await root.mount(
        lambda _: None,
        name="missing-consumer",
        inject=(GREETING,),
    )
    with pytest.raises(RuntimeError, match="required_pending"):
        RuntimeSnapshotCompiler().compile({}, composition_root=root)


@pytest.mark.asyncio
async def test_snapshot_store_publishes_complete_composition_root() -> None:
    drained: list[str] = []

    async def dispose_snapshot(snapshot) -> None:
        if snapshot.composition_root is not None:
            drained.append(snapshot.composition_root.generation_id)
            await snapshot.composition_root.dispose()

    compiler = RuntimeSnapshotCompiler()
    stable = compiler.compile({}, snapshot_revision="stable")
    root = CompositionRoot("candidate-ready")
    await _mount_greeting(root)
    candidate = compiler.compile(
        {},
        snapshot_revision="candidate",
        composition_root=root,
    )
    store = RuntimeSnapshotStore(dispose_snapshot)
    store.install(stable)
    transaction = store.begin_publish(candidate)
    await store.commit_latest(transaction)

    lease = store.lease(selector="latest")
    assert lease.snapshot.composition_root is root
    assert lease.snapshot.composition_topology == root.topology_view()
    assert lease.snapshot.composition_root.context.require(GREETING) == "hello"
    await lease.release()

    store.pause_candidate_admission(candidate)
    store.seal_candidate_validation(candidate)
    await store.promote_latest()
    await store.retry_drains()
    assert store.current is candidate
    assert drained == []
    await store.close()
    assert drained == ["candidate-ready"]


@pytest.mark.asyncio
async def test_snapshot_store_rejects_topology_drift_after_compile() -> None:
    root = CompositionRoot("candidate-drift")
    provider = await _mount_greeting(root)
    await root.mount(
        lambda _: None,
        name="consumer",
        inject=(GREETING,),
    )
    candidate = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    compiled_view = candidate.composition_topology
    await provider.dispose()

    assert compiled_view is not None
    assert candidate.composition_topology is compiled_view
    assert compiled_view.identity != root.topology_identity()

    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}))
    with pytest.raises(RuntimeError, match="组合拓扑未就绪"):
        store.begin_publish(candidate)


@pytest.mark.asyncio
async def test_topology_identity_excludes_mutable_state_and_generic_effects() -> None:
    root = CompositionRoot("immutable-topology")
    fiber = await root.mount(lambda _: None, name="optional")
    compiled = root.topology_view()
    compiled_snapshot = RuntimeSnapshotCompiler().compile(
        {},
        snapshot_revision="immutable-topology",
        composition_root=root,
    )

    fiber.state = FiberState.PENDING
    effect = await root.context.effect(lambda: None, label="diagnostic")
    observed = root.topology_view()

    assert observed.identity == compiled.identity
    assert observed.composition_revision == compiled.composition_revision
    assert observed.effects != compiled.effects

    await effect.aclose()
    fiber.state = FiberState.ACTIVE
    assert root.topology_identity() == compiled.identity
    assert root.composition_revision == compiled.composition_revision
    rebuilt_snapshot = RuntimeSnapshotCompiler().compile(
        {},
        snapshot_revision="immutable-topology",
        composition_root=root,
    )
    assert rebuilt_snapshot.snapshot_id == compiled_snapshot.snapshot_id
    await root.dispose()


@pytest.mark.asyncio
async def test_plugin_fiber_handle_hides_core_owned_mutable_state() -> None:
    root = CompositionRoot("fiber-handle")
    handles: list[object] = []

    async def apply(ctx) -> None:
        handles.append(ctx.fiber)
        handles.append(await ctx.mount(lambda _: None, name="child"))

    await root.mount(apply, name="owner")

    for handle in handles:
        assert not hasattr(handle, "effects")
        assert not hasattr(handle, "children")
        assert not hasattr(handle, "dependencies")
        with pytest.raises(AttributeError):
            setattr(handle, "state", FiberState.DISPOSED)

    await root.dispose()


@pytest.mark.asyncio
async def test_provider_restart_alone_invalidates_sealed_revision() -> None:
    root = CompositionRoot("service-revision")
    provider = await _mount_greeting(root)
    compiler = RuntimeSnapshotCompiler()
    candidate = compiler.compile({}, composition_root=root)
    compiled = candidate.composition_topology
    assert compiled is not None
    store = RuntimeSnapshotStore()
    store.install(compiler.compile({}))
    transaction = store.begin_publish(candidate)
    await store.commit_latest(transaction)

    await provider.restart()

    restored = root.topology_view()
    assert restored.identity == compiled.identity
    assert restored.composition_revision == compiled.composition_revision + 2
    store.pause_candidate_admission(candidate)
    with pytest.raises(RuntimeError, match="发生过结构变化"):
        store.seal_candidate_validation(candidate)
    _ = await store.discard_latest(candidate)
    await store.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_publication_participant_can_retain_closed_exact_target() -> None:
    compiler = RuntimeSnapshotCompiler()
    store = RuntimeSnapshotStore()
    stable = compiler.compile({}, snapshot_revision="stable")
    candidate = compiler.compile({}, snapshot_revision="candidate")
    store.install(stable)
    transaction = store.begin_publish(candidate)

    with pytest.raises(RuntimeError, match="不可租用"):
        store.lease(candidate.snapshot_id)
    retained = store.retain_publication_target(transaction)

    assert retained.snapshot is candidate
    assert retained.active
    assert candidate.lease_count == 1
    await retained.release()
    await store.abort(transaction)
    with pytest.raises(RuntimeError, match="target 已失效"):
        store.retain_publication_target(transaction)
    await store.close()


@pytest.mark.asyncio
async def test_fiber_replace_alone_invalidates_sealed_revision() -> None:
    root = CompositionRoot("fiber-revision")
    fiber = await root.mount(lambda _: None, name="plain")
    compiler = RuntimeSnapshotCompiler()
    candidate = compiler.compile({}, composition_root=root)
    compiled = candidate.composition_topology
    assert compiled is not None
    store = RuntimeSnapshotStore()
    store.install(compiler.compile({}))
    transaction = store.begin_publish(candidate)
    await store.commit_latest(transaction)

    await fiber.dispose()
    _ = await root.mount(lambda _: None, name="plain")

    restored = root.topology_view()
    assert restored.identity == compiled.identity
    assert restored.composition_revision == compiled.composition_revision + 2
    store.pause_candidate_admission(candidate)
    with pytest.raises(RuntimeError, match="发生过结构变化"):
        store.seal_candidate_validation(candidate)
    _ = await store.discard_latest(candidate)
    await store.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_isomorphic_roots_ignore_generation_identity() -> None:
    async def build(generation_id: str) -> CompositionRoot:
        root = CompositionRoot(generation_id)
        _ = await root.mount(lambda _: None, name="same-plugin")
        return root

    first = await build("candidate-generation")
    second = await build("production-generation")
    compiler = RuntimeSnapshotCompiler()
    first_snapshot = compiler.compile(
        {},
        snapshot_revision="same-input",
        composition_root=first,
    )
    second_snapshot = compiler.compile(
        {},
        snapshot_revision="same-input",
        composition_root=second,
    )

    assert first.topology_identity() == second.topology_identity()
    assert first_snapshot.snapshot_id == second_snapshot.snapshot_id
    await first.dispose()
    await second.dispose()


@pytest.mark.asyncio
async def test_topology_identity_includes_parent_ownership() -> None:
    async def build(*, nested: bool) -> CompositionRoot:
        root = CompositionRoot("nested" if nested else "flat")

        async def apply_host(ctx) -> None:
            async def apply_group(group_ctx) -> None:
                if nested:
                    _ = await group_ctx.mount(lambda _: None, name="worker")

            _ = await ctx.mount(apply_group, name="group")
            if not nested:
                _ = await ctx.mount(lambda _: None, name="worker")

        _ = await root.mount(apply_host, name="host")
        return root

    nested = await build(nested=True)
    flat = await build(nested=False)
    nested_view = nested.topology_view()
    flat_view = flat.topology_view()
    compiler = RuntimeSnapshotCompiler()

    assert nested_view.composition_revision == flat_view.composition_revision
    assert tuple((item.name, item.parent) for item in nested_view.fibers) == (
        ("group", "host"),
        ("host", None),
        ("worker", "group"),
    )
    assert tuple((item.name, item.parent) for item in flat_view.fibers) == (
        ("group", "host"),
        ("host", None),
        ("worker", "host"),
    )
    assert nested_view.identity != flat_view.identity
    assert (
        compiler.compile(
            {},
            snapshot_revision="same-input",
            composition_root=nested,
        ).snapshot_id
        != compiler.compile(
            {},
            snapshot_revision="same-input",
            composition_root=flat,
        ).snapshot_id
    )

    await nested.dispose()
    await flat.dispose()


@pytest.mark.asyncio
async def test_topology_view_identity_includes_declared_dependencies() -> None:
    async def build(dependency: ServiceKey[object]) -> str:
        root = CompositionRoot("dependency-view")
        greeting_plugin = GreetingProvider()
        await root.mount(greeting_plugin.apply, name=greeting_plugin.name)

        class FormatterProvider:
            name = "formatter-provider"
            inject = ()

            async def apply(self, ctx) -> None:
                await ctx.provide(FORMATTER, lambda value: value)

        formatter_plugin = FormatterProvider()
        await root.mount(formatter_plugin.apply, name=formatter_plugin.name)
        await root.mount(
            lambda _: None,
            name="consumer",
            inject=(dependency,),
        )
        return root.topology_identity()

    greeting = await build(GREETING)
    formatter = await build(FORMATTER)

    assert greeting != formatter


@pytest.mark.asyncio
async def test_promotion_rechecks_candidate_topology_after_behavior_probe() -> None:
    root = CompositionRoot("candidate-promotion-recheck")
    provider = await _mount_greeting(root)
    await root.mount(
        lambda _: None,
        name="consumer",
        inject=(GREETING,),
    )
    compiler = RuntimeSnapshotCompiler()
    candidate = compiler.compile({}, composition_root=root)
    store = RuntimeSnapshotStore()
    store.install(compiler.compile({}))
    transaction = store.begin_publish(candidate)
    await store.commit_latest(transaction)

    await provider.dispose()
    store.pause_candidate_admission(candidate)
    with pytest.raises(RuntimeError, match="组合拓扑未就绪"):
        await store.promote_latest()

    _ = await _mount_greeting(root)
    assert root.topology_identity() == candidate.composition_topology.identity  # type: ignore[union-attr]
    with pytest.raises(RuntimeError, match="发生过结构变化"):
        store.seal_candidate_validation(candidate)

    _ = await store.discard_latest(candidate)
    rebuilt = compiler.compile({}, composition_root=root)
    transaction = store.begin_publish(rebuilt)
    await store.commit_latest(transaction)
    store.pause_candidate_admission(rebuilt)
    store.seal_candidate_validation(rebuilt)
    _ = await store.promote_latest()
    assert store.current is rebuilt
    await store.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_plugin_manager_drains_snapshot_composition_root() -> None:
    cleaned: list[str] = []
    root = CompositionRoot("manager-drain")

    async def apply(ctx) -> None:
        await ctx.effect(
            lambda: lambda: cleaned.append("cleaned"),
            label="owned",
        )

    await root.mount(apply, name="plugin")
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    manager = object.__new__(PluginManager)
    manager._snapshot_store = RuntimeSnapshotStore()
    manager._snapshot_skill_catalogs = {}
    manager._dashboard_validation_releaser = None
    manager._finish_drained_reload = lambda _: None
    manager._runtime_started_roots = set()
    manager._runtime_lifecycle_lock = asyncio.Lock()

    await manager._on_snapshot_drained(snapshot)
    assert cleaned == ["cleaned"]
    assert root.receipt().ready is False


def test_core_plugin_data_access_records_scoped_writes(tmp_path) -> None:
    audit = CompositionAudit()
    access = PluginDataAccess(tmp_path, audit)
    data = access.for_plugin("probe")
    target = data.write_text("state/value.json", '{"value": 1}\n')
    assert target.relative_to(tmp_path).as_posix() == (
        "plugin-data/probe/state/value.json"
    )
    assert data.read_text("state/value.json") == '{"value": 1}\n'
    assert [
        (write.plugin_id, write.operation, write.relative_path)
        for write in audit.writes
    ] == [("probe", "create", "state/value.json")]
    with pytest.raises(ValueError, match="相对路径无效"):
        data.write_text("../escape", "blocked")


def test_core_plugin_data_access_never_follows_scoped_symlinks(tmp_path) -> None:
    audit = CompositionAudit()
    data = PluginDataAccess(tmp_path, audit).for_plugin("probe")
    outside = tmp_path / "outside"
    outside.mkdir()
    (data.root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        data.write_text("escape/value.json", "blocked")
    assert list(outside.iterdir()) == []


def test_external_effect_gate_records_denial() -> None:
    audit = CompositionAudit()
    gate = ExternalEffectGate(audit)
    with pytest.raises(PermissionError, match="禁止外部效果"):
        gate.authorize(kind="http", target="https://example.invalid")
    assert [asdict(effect) for effect in audit.external_effects] == [
        {
            "kind": "http",
            "target": "https://example.invalid",
            "outcome": "denied",
        }
    ]


@pytest.mark.asyncio
async def test_external_effect_attempt_rejects_candidate_even_when_caught() -> None:
    audit = CompositionAudit()
    root = CompositionRoot("external-effect-gate", audit=audit)
    gate = ExternalEffectGate(audit)

    async def apply(_) -> None:
        with pytest.raises(PermissionError):
            gate.authorize(kind="http", target="https://example.invalid")

    _ = await root.mount(apply, name="caught-denial")
    assert root.receipt().ready is False
    with pytest.raises(RuntimeError, match="拓扑未就绪"):
        RuntimeSnapshotCompiler().compile({}, composition_root=root)


@pytest.mark.asyncio
async def test_internal_root_cleanup_is_fail_loud_and_not_in_topology() -> None:
    root = CompositionRoot("internal-cleanup")

    def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    root._defer_internal_cleanup(  # pyright: ignore[reportPrivateUsage]
        "candidate-module",
        fail_cleanup,
    )
    _ = await root.mount(lambda _: None, name="plugin")

    assert "candidate-module" not in root.topology_view().effects
    with pytest.raises(BaseExceptionGroup, match="Root Context 清理失败"):
        await root.dispose()


@pytest.mark.asyncio
async def test_promotion_requires_sealed_receipt_and_rejects_later_write(
    tmp_path,
) -> None:
    audit = CompositionAudit()
    root = CompositionRoot("sealed-validation", audit=audit)
    data = PluginDataAccess(tmp_path, audit).for_plugin("probe")
    data.write_text("state.json", "first")
    _ = await _mount_greeting(root)
    compiler = RuntimeSnapshotCompiler()
    candidate = compiler.compile({}, composition_root=root)
    store = RuntimeSnapshotStore()
    store.install(compiler.compile({}))
    await store.commit_latest(store.begin_publish(candidate))
    store.pause_candidate_admission(candidate)

    with pytest.raises(RuntimeError, match="缺少 Core 验证回执"):
        await store.promote_latest()
    store.seal_candidate_validation(candidate)
    data.write_text("state.json", "second")
    with pytest.raises(RuntimeError, match="封存后发生变化"):
        await store.promote_latest()

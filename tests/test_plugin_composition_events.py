from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

import pytest

from agent.plugin_composition import (
    Bail,
    CompositionError,
    CompositionRoot,
    Effect,
    EmitEventKey,
    FiberState,
    ObserveEventKey,
    ParallelEventKey,
    SerialEventKey,
    ServiceKey,
    TransformEventKey,
)

NOTICE = EmitEventKey[str]("notice")
TRANSFORM = SerialEventKey[list[str], str]("transform")
OBSERVE = ParallelEventKey[str]("observe")
FINAL_OBSERVE = ObserveEventKey[str]("final-observe")
DEPENDENCY = ServiceKey[str]("event-dependency")


@dataclass(frozen=True, slots=True)
class Rewrite:
    steps: tuple[str, ...]


REWRITE = TransformEventKey("rewrite", Rewrite, "test.rewrite.v1")


@pytest.mark.parametrize(
    "key",
    [NOTICE, TRANSFORM, OBSERVE, REWRITE, FINAL_OBSERVE],
)
@pytest.mark.asyncio
async def test_registration_rejects_non_callable_without_root_mutation(
    key: object,
) -> None:
    root = CompositionRoot("invalid-listener")
    before = root.topology_view()
    before_effects = root.receipt().effects

    with pytest.raises(CompositionError) as caught:
        _ = await root.context.on(cast(Any, key), cast(Any, None))

    after = root.topology_view()
    assert caught.value.code == "INVALID_EVENT_LISTENER"
    assert after.identity == before.identity
    assert after.composition_revision == before.composition_revision
    assert after.listeners == ()
    assert root.receipt().effects == before_effects
    await root.dispose()


@pytest.mark.asyncio
async def test_emit_runs_sync_listeners_in_registration_order() -> None:
    observed: list[str] = []
    root = CompositionRoot("emit-order")

    async def first(ctx) -> None:
        await ctx.on(NOTICE, lambda payload: observed.append(f"first:{payload}"))

    async def second(ctx) -> None:
        await ctx.on(NOTICE, lambda payload: observed.append(f"second:{payload}"))

    await root.mount(first, name="first")
    await root.mount(second, name="second")
    root.context.emit(NOTICE, "ready")

    assert observed == ["first:ready", "second:ready"]
    assert "event:EmitEventKey:notice" in root.receipt().effects[0]


@pytest.mark.asyncio
async def test_topology_identity_preserves_listener_registration_order() -> None:
    async def build(order: tuple[str, str]) -> str:
        root = CompositionRoot("event-order-identity")

        for owner in order:
            async def plugin(ctx) -> None:
                await ctx.on(NOTICE, lambda _: None)

            await root.mount(plugin, name=owner)
        return root.topology_identity()

    first_then_second = await build(("first", "second"))
    second_then_first = await build(("second", "first"))

    assert first_then_second != second_then_first


@pytest.mark.asyncio
async def test_topology_view_exposes_ordered_listener_revision() -> None:
    root = CompositionRoot("event-view")

    async def first(ctx) -> None:
        await ctx.on(NOTICE, lambda _: None)

    async def second(ctx) -> None:
        await ctx.on(NOTICE, lambda _: None)

    await root.mount(first, name="first")
    await root.mount(second, name="second")
    view = root.topology_view()

    assert view.identity == root.topology_identity()
    assert view.listeners == (
        "emit:notice:first",
        "emit:notice:second",
    )
    assert len(view.identity) == 64


@pytest.mark.asyncio
async def test_listener_remove_and_restore_keeps_hash_but_advances_revision() -> None:
    root = CompositionRoot("event-revision")
    effects: list[Effect] = []

    async def plugin(ctx) -> None:
        effects.append(await ctx.on(NOTICE, lambda _: None))

    fiber = await root.mount(plugin, name="listener")
    compiled = root.topology_view()

    await effects.pop().aclose()
    removed = root.topology_view()
    effects.append(await fiber.context.on(NOTICE, lambda _: None))
    restored = root.topology_view()

    assert removed.identity != compiled.identity
    assert restored.identity == compiled.identity
    assert restored.composition_revision == compiled.composition_revision + 2
    await root.dispose()


@pytest.mark.asyncio
async def test_emit_rejects_async_listener_during_registration() -> None:
    root = CompositionRoot("emit-async-listener")

    async def listener(_: str) -> None:
        return None

    async def plugin(ctx) -> None:
        await ctx.on(NOTICE, listener)

    fiber = await root.mount(plugin, name="broken")

    assert fiber.state == FiberState.FAILED
    assert any(
        "ASYNC_LISTENER_ON_EMIT" in incident.message
        for incident in root.receipt().incidents
    )


@pytest.mark.asyncio
async def test_emit_rejects_awaitable_returned_by_sync_listener() -> None:
    root = CompositionRoot("emit-awaitable-result")

    async def delayed() -> None:
        return None

    async def plugin(ctx) -> None:
        await ctx.on(NOTICE, lambda _: delayed())

    await root.mount(plugin, name="wrapper")

    with pytest.raises(CompositionError) as caught:
        root.context.emit(NOTICE, "ready")
    assert caught.value.code == "ASYNC_RESULT_FROM_EMIT"


@pytest.mark.asyncio
async def test_event_name_cannot_change_dispatch_mode_while_registered() -> None:
    root = CompositionRoot("event-mode-conflict")
    conflicting = SerialEventKey[str, str](NOTICE.name)

    async def emit_plugin(ctx) -> None:
        await ctx.on(NOTICE, lambda _: None)

    async def serial_plugin(ctx) -> None:
        await ctx.on(conflicting, lambda _: None)

    await root.mount(emit_plugin, name="emit-owner")
    fiber = await root.mount(serial_plugin, name="serial-owner")

    assert fiber.state == FiberState.FAILED
    assert any(
        "EVENT_MODE_CONFLICT" in incident.message
        for incident in root.receipt().incidents
    )


@pytest.mark.asyncio
async def test_serial_awaits_in_order_and_only_explicit_bail_stops() -> None:
    observed: list[str] = []
    payload: list[str] = []
    root = CompositionRoot("serial-bail")

    async def first_handler(value: list[str]) -> None:
        await asyncio.sleep(0)
        value.append("first")
        observed.append("first")

    def second_handler(value: list[str]) -> Bail[str]:
        value.append("second")
        observed.append("second")
        return Bail("stop")

    async def first(ctx) -> None:
        await ctx.on(TRANSFORM, first_handler)

    async def second(ctx) -> None:
        await ctx.on(TRANSFORM, second_handler)

    async def third(ctx) -> None:
        await ctx.on(TRANSFORM, lambda value: observed.append("third"))

    await root.mount(first, name="first")
    await root.mount(second, name="second")
    await root.mount(third, name="third")

    result = await root.context.serial(TRANSFORM, payload)

    assert result == Bail("stop")
    assert payload == ["first", "second"]
    assert observed == ["first", "second"]


@pytest.mark.asyncio
async def test_serial_rejects_implicit_truthy_result() -> None:
    root = CompositionRoot("serial-invalid-result")

    async def plugin(ctx) -> None:
        await ctx.on(TRANSFORM, lambda _: "implicit-stop")

    await root.mount(plugin, name="invalid")

    with pytest.raises(CompositionError) as caught:
        await root.context.serial(TRANSFORM, [])
    assert caught.value.code == "INVALID_SERIAL_RESULT"


@pytest.mark.asyncio
async def test_transform_returns_original_without_listeners() -> None:
    root = CompositionRoot("transform-empty")
    original = Rewrite(("original",))

    transformed = await root.context.transform(REWRITE, original)

    assert transformed is original


@pytest.mark.asyncio
async def test_transform_chains_sync_and_async_listeners_in_order() -> None:
    root = CompositionRoot("transform-order")

    def first(value: Rewrite) -> Rewrite:
        return Rewrite((*value.steps, "first"))

    async def second(value: Rewrite) -> Rewrite:
        await asyncio.sleep(0)
        return Rewrite((*value.steps, "second"))

    async def first_plugin(ctx) -> None:
        await ctx.on(REWRITE, first)

    async def second_plugin(ctx) -> None:
        await ctx.on(REWRITE, second)

    await root.mount(first_plugin, name="first")
    await root.mount(second_plugin, name="second")

    transformed = await root.context.transform(REWRITE, Rewrite(()))

    assert transformed == Rewrite(("first", "second"))
    assert root.topology_view().listeners == (
        "transform:rewrite[test.rewrite.v1]:first",
        "transform:rewrite[test.rewrite.v1]:second",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, Bail("stop"), "wrong-type"])
async def test_transform_rejects_implicit_or_wrong_result(invalid: object) -> None:
    root = CompositionRoot("transform-invalid")

    async def plugin(ctx) -> None:
        await ctx.on(REWRITE, lambda _: invalid)

    await root.mount(plugin, name="invalid")

    with pytest.raises(CompositionError) as caught:
        await root.context.transform(REWRITE, Rewrite(()))
    assert caught.value.code == "INVALID_TRANSFORM_RESULT"
    assert root.receipt().incidents[-1].kind == "transform_failure"


@pytest.mark.asyncio
async def test_transform_failure_stops_chain_and_records_incident() -> None:
    observed: list[str] = []
    root = CompositionRoot("transform-failure")

    def broken(_: Rewrite) -> Rewrite:
        raise RuntimeError("rewrite failed")

    async def first(ctx) -> None:
        await ctx.on(REWRITE, broken)

    async def second(ctx) -> None:
        await ctx.on(REWRITE, lambda value: observed.append("second") or value)

    await root.mount(first, name="broken")
    await root.mount(second, name="second")

    with pytest.raises(RuntimeError, match="rewrite failed"):
        await root.context.transform(REWRITE, Rewrite(()))

    assert observed == []
    incident = root.receipt().incidents[-1]
    assert (incident.owner, incident.kind, incident.error_type) == (
        "broken",
        "transform_failure",
        "RuntimeError",
    )


@pytest.mark.asyncio
async def test_transform_uses_one_frozen_listener_list_per_dispatch() -> None:
    root = CompositionRoot("transform-frozen-list")
    observed: list[str] = []
    second_fiber = None

    async def first_listener(value: Rewrite) -> Rewrite:
        assert second_fiber is not None
        observed.append("first")
        await second_fiber.dispose()
        return Rewrite((*value.steps, "first"))

    async def first(ctx) -> None:
        await ctx.on(REWRITE, first_listener)

    async def second(ctx) -> None:
        await ctx.on(
            REWRITE,
            lambda value: observed.append("second")
            or Rewrite((*value.steps, "second")),
        )

    await root.mount(first, name="first")
    second_fiber = await root.mount(second, name="second")

    transformed = await root.context.transform(REWRITE, Rewrite(()))

    assert transformed == Rewrite(("first", "second"))
    assert observed == ["first", "second"]
    assert second_fiber.state == FiberState.DISPOSED


@pytest.mark.asyncio
async def test_observe_runs_every_listener_and_contains_all_failures() -> None:
    observed: list[str] = []
    root = CompositionRoot("observe-failures")

    def sync_failure(_: str) -> None:
        observed.append("sync-failure")
        raise ValueError("sync observer failed")

    async def async_failure(_: str) -> None:
        await asyncio.sleep(0)
        observed.append("async-failure")
        raise RuntimeError("async observer failed")

    async def first(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, sync_failure)

    async def second(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, async_failure)

    async def third(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, lambda value: observed.append(value))

    await root.mount(first, name="sync")
    await root.mount(second, name="async")
    await root.mount(third, name="final")

    await root.context.observe(FINAL_OBSERVE, "settled")

    assert observed == ["sync-failure", "settled", "async-failure"]
    failures = [
        (incident.owner, incident.kind, incident.error_type)
        for incident in root.receipt().incidents
    ]
    assert failures == [
        ("sync", "observer_failure", "ValueError"),
        ("async", "observer_failure", "RuntimeError"),
    ]
    assert root.receipt().ready is True


@pytest.mark.asyncio
async def test_observe_contains_failure_with_unprintable_exception() -> None:
    observed: list[str] = []
    root = CompositionRoot("observe-unprintable")

    class UnprintableError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("coercion trap")

    def broken(_: str) -> None:
        raise UnprintableError()

    async def first(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, broken)

    async def second(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, lambda value: observed.append(value))

    await root.mount(first, name="broken")
    await root.mount(second, name="final")

    await root.context.observe(FINAL_OBSERVE, "settled")

    assert observed == ["settled"]
    incident = root.receipt().incidents[-1]
    assert incident.message == "<unprintable UnprintableError>"


@pytest.mark.asyncio
async def test_observe_caller_cancellation_cancels_and_drains_listeners() -> None:
    started = [asyncio.Event(), asyncio.Event()]
    cleaned = [asyncio.Event(), asyncio.Event()]
    root = CompositionRoot("observe-cancel")

    def listener(index: int):
        async def run(_: str) -> None:
            started[index].set()
            try:
                await asyncio.Future()
            finally:
                cleaned[index].set()

        return run

    async def first(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, listener(0))

    async def second(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, listener(1))

    await root.mount(first, name="first")
    await root.mount(second, name="second")
    dispatch = asyncio.create_task(root.context.observe(FINAL_OBSERVE, "settled"))
    await asyncio.gather(*(event.wait() for event in started))
    _ = dispatch.cancel()
    await asyncio.sleep(0)
    _ = dispatch.cancel()

    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert all(event.is_set() for event in cleaned)
    assert root.receipt().incidents == ()


@pytest.mark.asyncio
async def test_observe_sync_system_exit_closes_unstarted_listener_and_propagates() -> None:
    started = asyncio.Event()
    created = []
    root = CompositionRoot("observe-system-exit")

    async def blocking(_: str) -> None:
        started.set()
        await asyncio.Future()

    def create_blocking(value: str):
        result = blocking(value)
        created.append(result)
        return result

    def terminate(_: str) -> None:
        raise SystemExit(7)

    async def first(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, create_blocking)

    async def second(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, terminate)

    await root.mount(first, name="blocking")
    await root.mount(second, name="terminate")

    with pytest.raises(SystemExit) as caught:
        await root.context.observe(FINAL_OBSERVE, "settled")

    assert caught.value.code == 7
    assert not started.is_set()
    assert len(created) == 1
    assert created[0].cr_frame is None
    assert root.receipt().incidents == ()


@pytest.mark.asyncio
async def test_observe_cleanup_failure_does_not_mask_system_exit() -> None:
    created = []
    root = CompositionRoot("observe-cleanup-failure")

    class BrokenCloseAwaitable:
        def __await__(self):
            if False:
                yield None
            return None

        def close(self) -> None:
            raise RuntimeError("close failed")

    async def blocking(_: str) -> None:
        await asyncio.Future()

    def create_blocking(value: str):
        result = blocking(value)
        created.append(result)
        return result

    def terminate(_: str) -> None:
        raise SystemExit(8)

    async def first(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, lambda _: BrokenCloseAwaitable())

    async def second(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, create_blocking)

    async def third(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, terminate)

    await root.mount(first, name="broken-close")
    await root.mount(second, name="blocking")
    await root.mount(third, name="terminate")

    with pytest.raises(SystemExit) as caught:
        await root.context.observe(FINAL_OBSERVE, "settled")

    assert caught.value.code == 8
    assert len(created) == 1
    assert created[0].cr_frame is None
    incident = root.receipt().incidents[-1]
    assert incident.owner == "broken-close"
    assert incident.kind == "observer_cleanup_failure"
    assert incident.error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_observe_async_system_exit_propagates_after_all_callbacks() -> None:
    observed: list[str] = []
    root = CompositionRoot("observe-async-system-exit")

    async def terminate(_: str) -> None:
        raise SystemExit(9)

    async def first(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, terminate)

    async def second(ctx) -> None:
        await ctx.on(FINAL_OBSERVE, lambda _: observed.append("later"))

    await root.mount(first, name="terminate")
    await root.mount(second, name="later")

    with pytest.raises(SystemExit) as caught:
        await root.context.observe(FINAL_OBSERVE, "settled")

    assert caught.value.code == 9
    assert observed == ["later"]
    assert root.receipt().incidents == ()


@pytest.mark.asyncio
async def test_transform_event_name_cannot_change_payload_contract() -> None:
    root = CompositionRoot("transform-contract-conflict")
    conflicting = TransformEventKey("rewrite", str, "test.rewrite.v1")

    async def first(ctx) -> None:
        await ctx.on(REWRITE, lambda value: value)

    async def second(ctx) -> None:
        await ctx.on(conflicting, lambda value: value)

    await root.mount(first, name="rewrite-owner")
    fiber = await root.mount(second, name="string-owner")

    assert fiber.state == FiberState.FAILED
    assert any(
        "EVENT_MODE_CONFLICT" in incident.message
        for incident in root.receipt().incidents
    )


@pytest.mark.asyncio
async def test_transform_topology_uses_stable_payload_contract_token() -> None:
    class CandidateRewrite:
        pass

    class FormalRewrite:
        pass

    async def build(payload_type: type[object], token: str):
        root = CompositionRoot(f"transform-contract:{token}")
        key = TransformEventKey("stable-rewrite", payload_type, token)

        async def plugin(ctx) -> None:
            await ctx.on(key, lambda value: value)

        await root.mount(plugin, name="owner")
        return root.topology_view()

    candidate = await build(CandidateRewrite, "plugin.rewrite.v1")
    formal = await build(FormalRewrite, "plugin.rewrite.v1")
    changed = await build(FormalRewrite, "plugin.rewrite.v2")

    assert candidate.listeners == formal.listeners == (
        "transform:stable-rewrite[plugin.rewrite.v1]:owner",
    )
    assert candidate.identity == formal.identity
    assert changed.listeners != formal.listeners
    assert changed.identity != formal.identity


@pytest.mark.asyncio
async def test_parallel_starts_together_and_aggregates_all_failures() -> None:
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()
    root = CompositionRoot("parallel-errors")

    async def first(_: str) -> None:
        first_started.set()
        await release.wait()
        raise ValueError("first")

    async def second(_: str) -> None:
        second_started.set()
        await release.wait()
        raise RuntimeError("second")

    async def plugin_a(ctx) -> None:
        await ctx.on(OBSERVE, first)

    async def plugin_b(ctx) -> None:
        await ctx.on(OBSERVE, second)

    await root.mount(plugin_a, name="first")
    await root.mount(plugin_b, name="second")
    dispatch = asyncio.create_task(root.context.parallel(OBSERVE, "event"))
    await asyncio.gather(first_started.wait(), second_started.wait())
    release.set()

    with pytest.raises(BaseExceptionGroup) as caught:
        await dispatch
    assert {type(error) for error in caught.value.exceptions} == {
        ValueError,
        RuntimeError,
    }


@pytest.mark.asyncio
async def test_parallel_cancellation_drains_every_listener() -> None:
    started = [asyncio.Event(), asyncio.Event()]
    cleaned = [asyncio.Event(), asyncio.Event()]
    root = CompositionRoot("parallel-cancel")

    def listener(index: int):
        async def run(_: str) -> None:
            started[index].set()
            try:
                await asyncio.Future()
            finally:
                cleaned[index].set()

        return run

    async def first(ctx) -> None:
        await ctx.on(OBSERVE, listener(0))

    async def second(ctx) -> None:
        await ctx.on(OBSERVE, listener(1))

    await root.mount(first, name="first")
    await root.mount(second, name="second")
    dispatch = asyncio.create_task(root.context.parallel(OBSERVE, "event"))
    await asyncio.gather(*(event.wait() for event in started))
    _ = dispatch.cancel()
    await asyncio.sleep(0)
    _ = dispatch.cancel()

    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert all(event.is_set() for event in cleaned)


@pytest.mark.asyncio
async def test_serial_uses_one_frozen_listener_list_per_dispatch() -> None:
    observed: list[str] = []
    root = CompositionRoot("serial-frozen-list")
    second_fiber = None

    async def first_handler(_: list[str]) -> None:
        assert second_fiber is not None
        observed.append("first")
        await second_fiber.dispose()

    async def first_plugin(ctx) -> None:
        await ctx.on(TRANSFORM, first_handler)

    async def second_plugin(ctx) -> None:
        await ctx.on(TRANSFORM, lambda _: observed.append("second"))

    await root.mount(first_plugin, name="first")
    second_fiber = await root.mount(second_plugin, name="second")

    result = await root.context.serial(TRANSFORM, [])

    assert result is None
    assert observed == ["first", "second"]
    assert second_fiber.state == FiberState.DISPOSED


@pytest.mark.asyncio
async def test_dependency_loss_removes_listener_and_restore_registers_once() -> None:
    observed: list[str] = []
    root = CompositionRoot("event-dependency")

    class Consumer:
        name = "consumer"
        inject = (DEPENDENCY,)

        async def apply(self, ctx) -> None:
            await ctx.on(NOTICE, lambda payload: observed.append(payload))

    class Provider:
        name = "provider"
        inject = ()

        async def apply(self, ctx) -> None:
            await ctx.provide(DEPENDENCY, "ready")

    consumer_plugin = Consumer()
    consumer = await root.mount(
        consumer_plugin.apply,
        name=consumer_plugin.name,
        inject=consumer_plugin.inject,
    )
    provider_plugin = Provider()
    provider = await root.mount(provider_plugin.apply, name=provider_plugin.name)
    root.context.emit(NOTICE, "first")

    await provider.dispose()
    assert consumer.state == FiberState.PENDING
    root.context.emit(NOTICE, "missing")

    replacement = Provider()
    await root.mount(replacement.apply, name="replacement")
    root.context.emit(NOTICE, "second")

    assert observed == ["first", "second"]


@pytest.mark.asyncio
async def test_spawned_task_is_cancelled_with_owning_fiber() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()
    root = CompositionRoot("spawn-cleanup")

    async def worker() -> None:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cleaned.set()

    async def plugin(ctx) -> None:
        _ = await ctx.spawn(worker(), name="worker")

    fiber = await root.mount(plugin, name="task-owner")
    await started.wait()
    assert "task-owner:task:worker" in root.receipt().effects
    await fiber.dispose()

    assert cleaned.is_set()
    assert fiber.effects == []


@pytest.mark.asyncio
async def test_spawn_rejection_closes_unowned_coroutine() -> None:
    contexts: list[Any] = []
    root = CompositionRoot("spawn-rejected")

    async def plugin(ctx) -> None:
        contexts.append(ctx)

    fiber = await root.mount(plugin, name="task-owner")
    await fiber.dispose()

    async def worker() -> None:
        await asyncio.sleep(0)

    coroutine = worker()
    assert coroutine.cr_frame is not None
    with pytest.raises(CompositionError) as caught:
        _ = await contexts[0].spawn(coroutine, name="late-worker")

    assert caught.value.code == "INACTIVE_EFFECT"
    assert coroutine.cr_frame is None
    assert fiber.effects == []
    await root.dispose()


@pytest.mark.asyncio
async def test_spawned_task_failure_is_visible_to_candidate_readiness() -> None:
    failed = asyncio.Event()
    recovered = asyncio.Event()
    attempts = 0
    root = CompositionRoot("spawn-failure")

    async def worker() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            failed.set()
            raise RuntimeError("background failed")
        recovered.set()
        await asyncio.Event().wait()

    async def plugin(ctx) -> None:
        _ = await ctx.spawn(worker(), name="broken-worker")

    fiber = await root.mount(plugin, name="task-owner")
    await failed.wait()
    await asyncio.sleep(0)

    receipt = root.receipt()
    assert receipt.ready is False
    assert receipt.required_degraded == ("task-owner:task:broken-worker",)
    assert any(
        incident.kind == "task_failure"
        and "background failed" in incident.message
        for incident in receipt.incidents
    )

    await fiber.restart()
    await recovered.wait()
    recovered_receipt = root.receipt()
    assert recovered_receipt.ready is True
    assert recovered_receipt.required_degraded == ()
    assert any(
        "background failed" in incident.message
        for incident in recovered_receipt.incidents
    )
    await root.dispose()

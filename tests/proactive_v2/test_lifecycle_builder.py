from __future__ import annotations

import asyncio

import pytest

from proactive_v2.frame import ProactiveFrame
from proactive_v2.lifecycle import ProactiveLifecycleBuilder, ProactiveLifecycleSpec


class _Module:
    def __init__(
        self,
        slot: str,
        calls: list[str],
        *,
        phase: str | None = None,
        requires: tuple[str, ...] = (),
        produces: tuple[str, ...] = (),
        collects: tuple[str, ...] = (),
    ) -> None:
        self.slot = slot
        self.calls = calls
        self.requires = requires
        self.produces = produces
        self.collects = collects
        if phase is not None:
            self.phase = phase

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        self.calls.append(self.slot)
        return frame


@pytest.mark.asyncio
async def test_data_requires_orders_producer_before_consumer():
    calls: list[str] = []
    consumer = _Module(
        "filter.consumer",
        calls,
        requires=("feature:entities",),
    )
    producer = _Module(
        "feature.extract",
        calls,
        produces=("feature:entities",),
    )

    lifecycle = ProactiveLifecycleBuilder().build(
        ProactiveLifecycleSpec(id="test"),
        [consumer, producer],
    )
    await lifecycle.run(ProactiveFrame(input=object()))  # type: ignore[arg-type]

    assert calls == ["feature.extract", "filter.consumer"]


@pytest.mark.asyncio
async def test_collector_waits_for_all_matching_contributions():
    calls: list[str] = []
    collector = _Module(
        "filter.collect",
        calls,
        collects=("candidate:decision:*",),
        produces=("candidate:screened",),
    )
    filter_b = _Module(
        "filter.b",
        calls,
        produces=("candidate:decision:b",),
    )
    filter_a = _Module(
        "filter.a",
        calls,
        produces=("candidate:decision:a",),
    )

    lifecycle = ProactiveLifecycleBuilder().build(
        ProactiveLifecycleSpec(
            id="test",
            terminal_slots=("candidate:screened",),
        ),
        [collector, filter_b, filter_a],
    )
    await lifecycle.run(ProactiveFrame(input=object()))  # type: ignore[arg-type]

    assert calls[-1] == "filter.collect"
    assert set(calls[:-1]) == {"filter.a", "filter.b"}


def test_duplicate_data_producer_fails_compilation():
    modules = [
        _Module("filter.a", [], produces=("candidate:screened",)),
        _Module("filter.b", [], produces=("candidate:screened",)),
    ]

    with pytest.raises(RuntimeError, match="多 producer"):
        _ = ProactiveLifecycleBuilder().build(
            ProactiveLifecycleSpec(id="test"),
            modules,
        )


def test_missing_terminal_slot_fails_compilation():
    with pytest.raises(RuntimeError, match="终点 slot 无 producer"):
        _ = ProactiveLifecycleBuilder().build(
            ProactiveLifecycleSpec(
                id="test",
                terminal_slots=("run:next_wakeup",),
            )
        )


@pytest.mark.asyncio
async def test_start_failure_aggregates_all_rollback_errors():
    calls: list[str] = []
    start_error = RuntimeError("start failed")
    rollback_b_error = asyncio.CancelledError("rollback b cancelled")
    rollback_a_error = RuntimeError("rollback a failed")

    class StartModule(_Module):
        async def start(self) -> None:
            calls.append(f"start:{self.slot}")
            if self.slot == "b":
                raise start_error

        async def stop(self) -> None:
            calls.append(f"stop:{self.slot}")
            if self.slot == "b":
                raise rollback_b_error
            raise rollback_a_error

    lifecycle = ProactiveLifecycleBuilder().build(
        ProactiveLifecycleSpec(id="test"),
        [StartModule("a", calls), StartModule("b", calls)],
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await lifecycle.start()

    assert calls == ["start:a", "start:b", "stop:b", "stop:a"]
    assert raised.value.exceptions == (
        start_error,
        rollback_b_error,
        rollback_a_error,
    )


@pytest.mark.asyncio
async def test_stop_runs_all_modules_and_aggregates_failures_in_reverse_order():
    calls: list[str] = []
    stop_errors: dict[str, BaseException] = {
        "a": RuntimeError("stop a failed"),
        "b": RuntimeError("stop b failed"),
        "c": asyncio.CancelledError("stop c cancelled"),
    }

    class StopModule(_Module):
        async def stop(self) -> None:
            calls.append(f"stop:{self.slot}")
            raise stop_errors[self.slot]

    lifecycle = ProactiveLifecycleBuilder().build(
        ProactiveLifecycleSpec(id="test"),
        [StopModule("a", calls), StopModule("b", calls), StopModule("c", calls)],
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await lifecycle.stop()

    assert calls == ["stop:c", "stop:b", "stop:a"]
    assert raised.value.exceptions == (
        stop_errors["c"],
        stop_errors["b"],
        stop_errors["a"],
    )


@pytest.mark.asyncio
async def test_stop_finishes_cleanup_after_external_cancellation():
    calls: list[str] = []
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    class BlockingStopModule(_Module):
        async def stop(self) -> None:
            calls.append(f"stop:{self.slot}")
            if self.slot == "b":
                cleanup_started.set()
                _ = await release_cleanup.wait()

    lifecycle = ProactiveLifecycleBuilder().build(
        ProactiveLifecycleSpec(id="test"),
        [BlockingStopModule("a", calls), BlockingStopModule("b", calls)],
    )

    stop_task = asyncio.create_task(lifecycle.stop())
    _ = await cleanup_started.wait()
    _ = stop_task.cancel()
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await stop_task

    assert calls == ["stop:b", "stop:a"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slot", ""),
        ("requires", "not-a-slot-list"),
        ("produces", (object(),)),
        ("collects", (object(),)),
        ("run", object()),
        ("start", object()),
        ("stop", object()),
    ],
)
def test_bad_dynamic_module_fails_at_compile_boundary(field: str, value: object):
    module = _Module("bad", [])
    setattr(module, field, value)

    with pytest.raises(RuntimeError, match=field):
        _ = ProactiveLifecycleBuilder().build(
            ProactiveLifecycleSpec(id="test"),
            [module],
        )

from __future__ import annotations

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
        ProactiveLifecycleBuilder().build(
            ProactiveLifecycleSpec(id="test"),
            modules,
        )


def test_missing_terminal_slot_fails_compilation():
    with pytest.raises(RuntimeError, match="终点 slot 无 producer"):
        ProactiveLifecycleBuilder().build(
            ProactiveLifecycleSpec(
                id="test",
                terminal_slots=("run:next_wakeup",),
            )
        )


@pytest.mark.asyncio
async def test_start_failure_rolls_back_started_modules():
    calls: list[str] = []

    class StartModule(_Module):
        async def start(self) -> None:
            calls.append(f"start:{self.slot}")
            if self.slot == "b":
                raise RuntimeError("start failed")

        async def stop(self) -> None:
            calls.append(f"stop:{self.slot}")

    lifecycle = ProactiveLifecycleBuilder().build(
        ProactiveLifecycleSpec(id="test"),
        [StartModule("a", calls), StartModule("b", calls)],
    )

    with pytest.raises(RuntimeError, match="start failed"):
        await lifecycle.start()

    assert calls == ["start:a", "start:b", "stop:b", "stop:a"]

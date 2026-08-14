from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from agent.plugin_composition import (
    TIMER_SERVICE,
    CompositionRoot,
    TimerService,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus


@dataclass(slots=True)
class _PendingSleep:
    delay: float
    release: asyncio.Event


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[_PendingSleep] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        pending = _PendingSleep(delay, asyncio.Event())
        self.sleeps.append(pending)
        await pending.release.wait()

    async def wait_for_sleeps(self, count: int) -> None:
        async with asyncio.timeout(1):
            while len(self.sleeps) < count:
                await asyncio.sleep(0)


class _ReplayMissedTicksTimer(TimerService):
    @staticmethod
    def _next_deadline(deadline: float, delay: float, now: float) -> float:
        del now
        return deadline + delay


@pytest.mark.asyncio
async def test_interval_uses_fixed_cadence_and_skips_missed_ticks() -> None:
    clock = _ManualClock()
    timer = TimerService(clock=clock.monotonic, sleep=clock.sleep)
    root = CompositionRoot("timer-fixed-cadence")
    ticks = 0

    def callback() -> None:
        nonlocal ticks
        ticks += 1
        clock.now = 3.4

    handle = await timer.interval(root.context, callback, 1.0, name="probe")
    await clock.wait_for_sleeps(1)
    clock.now = 1.0
    clock.sleeps[0].release.set()
    await clock.wait_for_sleeps(2)

    assert ticks == 1
    assert clock.sleeps[0].delay == pytest.approx(1.0)
    assert clock.sleeps[1].delay == pytest.approx(0.6)

    await handle.aclose()
    await root.dispose()


@pytest.mark.asyncio
async def test_interval_awaits_async_callback_without_overlap() -> None:
    clock = _ManualClock()
    timer = TimerService(clock=clock.monotonic, sleep=clock.sleep)
    root = CompositionRoot("timer-no-overlap")
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()
    calls = 0

    async def callback() -> None:
        nonlocal calls
        calls += 1
        callback_started.set()
        await callback_release.wait()

    handle = await timer.interval(root.context, callback, 1.0, name="slow")
    await clock.wait_for_sleeps(1)
    clock.now = 1.0
    clock.sleeps[0].release.set()
    await callback_started.wait()
    clock.now = 5.0
    for _ in range(10):
        await asyncio.sleep(0)

    assert calls == 1
    assert len(clock.sleeps) == 1

    callback_release.set()
    await clock.wait_for_sleeps(2)
    assert clock.sleeps[1].delay == pytest.approx(1.0)
    await handle.aclose()
    await root.dispose()


@pytest.mark.asyncio
async def test_fixed_cadence_oracle_kills_replayed_tick_mutant() -> None:
    correct = await _second_sleep_delay(TimerService)
    mutant = await _second_sleep_delay(_ReplayMissedTicksTimer)

    assert correct == pytest.approx(0.6)
    assert mutant == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_timeout_releases_effect_and_reports_callback_failure() -> None:
    async def immediate(_: float) -> None:
        return

    timer = TimerService(sleep=immediate)
    root = CompositionRoot("timer-failure")

    def fail() -> None:
        raise RuntimeError("timer callback failed")

    handle = await timer.timeout(root.context, fail, 1.0, name="broken")
    async with asyncio.timeout(1):
        while not handle.done or root.receipt().effects:
            await asyncio.sleep(0)

    receipt = root.receipt()
    assert receipt.ready is False
    assert any("timer callback failed" in error for error in receipt.errors)
    await root.dispose()


@pytest.mark.asyncio
async def test_timer_rejects_invalid_public_inputs_before_spawning() -> None:
    root = CompositionRoot("timer-validation")
    timer = TimerService()

    for delay in (0, -1, float("inf"), float("nan"), True):
        with pytest.raises((TypeError, ValueError)):
            _ = await timer.timeout(root.context, lambda: None, delay)
    with pytest.raises(TypeError, match="callback"):
        _ = await timer.timeout(root.context, cast(Any, None), 1.0)

    assert root.receipt().effects == ()
    await root.dispose()


@pytest.mark.asyncio
async def test_namespace_loader_provides_timer_and_fiber_disposal_cancels_it(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins" / "timer_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import TIMER_SERVICE\n"
        "api_version = 3\n"
        "name = 'timer_probe'\n"
        "version = '1.0.0'\n"
        "inject = (TIMER_SERVICE,)\n"
        "ticks = 0\n"
        "handle = None\n"
        "async def apply(ctx, config):\n"
        "    global handle\n"
        "    async def tick():\n"
        "        global ticks\n"
        "        ticks += 1\n"
        "    handle = await ctx.require(TIMER_SERVICE).interval(\n"
        "        ctx, tick, 0.001, name='namespace-probe'\n"
        "    )\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )

    await manager.load_all()
    generation = manager.generation("timer_probe")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert isinstance(generation.instance, ComposablePlugin)
    module = generation.instance.module
    async with asyncio.timeout(1):
        while cast(int, module.ticks) < 2:
            await asyncio.sleep(0.001)
    handle = module.handle
    assert snapshot.composition_topology is not None
    assert snapshot.composition_topology.services == (
        "core.agent_input",
        "core.plugin_assets",
        "core.timer",
        "core.tools",
    )

    await manager.terminate_all()
    ticks_after_dispose = cast(int, module.ticks)
    await asyncio.sleep(0.005)

    assert handle.done is True
    assert module.ticks == ticks_after_dispose


async def _second_sleep_delay(
    timer_type: type[TimerService],
) -> float:
    """Observe the second wait after one callback overruns two cadence slots."""

    # 1. Run the production loop against a controlled monotonic clock.
    clock = _ManualClock()
    timer = timer_type(clock=clock.monotonic, sleep=clock.sleep)
    root = CompositionRoot(f"timer-oracle:{timer_type.__name__}")

    def callback() -> None:
        clock.now = 3.4

    handle = await timer.interval(root.context, callback, 1.0, name="oracle")
    await clock.wait_for_sleeps(1)
    clock.now = 1.0
    clock.sleeps[0].release.set()
    await clock.wait_for_sleeps(2)

    # 2. Close the pending wait so the fixture leaves no task behind.
    observed = clock.sleeps[1].delay
    await handle.aclose()
    await root.dispose()
    return observed

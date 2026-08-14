from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from agent.plugin_composition import (
    TIMER_SERVICE,
    CompositionError,
    CompositionRoot,
    PluginRuntime,
    TimerHandle,
    TimerService,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.jobs import PluginJobRuntime
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
    for name in ("", " leading", "trailing "):
        with pytest.raises(ValueError, match="name"):
            _ = await timer.timeout(root.context, lambda: None, 1.0, name=name)
    with pytest.raises(TypeError, match="name"):
        _ = await timer.timeout(
            root.context,
            lambda: None,
            1.0,
            name=cast(Any, None),
        )

    assert root.receipt().effects == ()
    await root.dispose()


@pytest.mark.asyncio
async def test_snapshot_timer_rejects_duplicate_and_frozen_declarations(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("timer-declarations")
    timer = TimerService.for_snapshot()
    runtime = PluginRuntime(
        plugin_id="timer_probe",
        plugin_dir=tmp_path / "plugin",
        data_dir=tmp_path / "data",
        workspace=tmp_path / "workspace",
        config={},
    )
    fiber = await root.mount(lambda _: None, name="timer_probe", runtime=runtime)
    handle = await timer.interval(
        fiber.context,
        lambda: None,
        1.0,
        name="poll",
    )

    with pytest.raises(CompositionError, match="稳定键重复"):
        _ = await timer.timeout(
            fiber.context,
            lambda: None,
            1.0,
            name="poll",
        )
    frozen = timer.freeze()
    with pytest.raises(CompositionError, match="声明已冻结"):
        _ = await timer.interval(
            fiber.context,
            lambda: None,
            1.0,
            name="after-freeze",
        )

    await handle.aclose()
    assert handle.done is True
    assert frozen["timer_probe:poll"].active is False
    await root.dispose()


@pytest.mark.asyncio
async def test_snapshot_interval_can_close_its_own_handle(tmp_path: Path) -> None:
    root = CompositionRoot("timer-self-close")
    timer = TimerService.for_snapshot()
    runtime = PluginRuntime(
        plugin_id="timer_probe",
        plugin_dir=tmp_path / "plugin",
        data_dir=tmp_path / "data",
        workspace=tmp_path / "workspace",
        config={},
    )
    fiber = await root.mount(lambda _: None, name="timer_probe", runtime=runtime)
    handles: list[TimerHandle] = []

    async def callback() -> None:
        await handles[0].aclose()

    handle = await timer.interval(fiber.context, callback, 1.0, name="self-close")
    handles.append(handle)
    registration = timer.freeze()["timer_probe:self-close"]

    await asyncio.wait_for(registration.invoke(), timeout=1)

    assert handle.done is True
    assert registration.active is False
    await root.dispose()


@pytest.mark.asyncio
async def test_namespace_loader_provides_timer_and_fiber_disposal_cancels_it(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins" / "timer_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        _timer_plugin_source("1.0.0", kind="interval"),
        encoding="utf-8",
    )
    event_bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
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
    handle = module.handle
    await asyncio.sleep(0.01)
    assert module.ticks == 0
    assert tuple(snapshot.timers) == ("timer_probe:namespace-probe",)
    assert snapshot.composition_topology is not None
    assert snapshot.composition_topology.services == (
        "core.agent_input",
        "core.commands",
        "core.skills",
        "core.timer",
        "core.tools",
        "core.ui_slots",
    )

    runtime = PluginJobRuntime(
        event_bus=event_bus,
        llm=cast(Any, object()),
        snapshot_store=manager.snapshot_store,
    )
    runtime_task = asyncio.create_task(runtime.run())
    try:
        async with asyncio.timeout(1):
            while cast(int, module.ticks) < 2:
                await asyncio.sleep(0.001)
        await handle.aclose()
        ticks_after_close = cast(int, module.ticks)
        await asyncio.sleep(0.03)
        assert module.ticks == ticks_after_close
    finally:
        runtime.stop()
        await runtime.wait_stopped()
        await runtime_task

    await manager.terminate_all()

    assert handle.done is True


@pytest.mark.asyncio
async def test_snapshot_interval_ignores_candidate_then_switches_generation(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins" / "timer_probe"
    plugin_dir.mkdir(parents=True)
    plugin_path = plugin_dir / "plugin.py"
    plugin_path.write_text(
        _timer_plugin_source("1.0.0", kind="interval"),
        encoding="utf-8",
    )
    event_bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    stable = manager.generation("timer_probe")
    assert stable is not None and isinstance(stable.instance, ComposablePlugin)
    runtime = PluginJobRuntime(
        event_bus=event_bus,
        llm=cast(Any, object()),
        snapshot_store=manager.snapshot_store,
    )
    runtime_task = asyncio.create_task(runtime.run())
    try:
        await _wait_for_ticks(stable.instance.module, 1)
        plugin_path.write_text(
            _timer_plugin_source("2.0.0", kind="interval"),
            encoding="utf-8",
        )
        candidate = await manager.prepare_candidate("timer_probe")
        assert candidate is not None
        assert isinstance(candidate.instance, ComposablePlugin)
        await asyncio.sleep(0.03)
        assert candidate.instance.module.ticks == 0

        result = await manager.publish_prepared("timer_probe")
        assert result["publication_state"] == "committed"
        await _wait_for_ticks(candidate.instance.module, 1)
        await asyncio.sleep(0.03)
        retired_ticks = stable.instance.module.ticks
        await asyncio.sleep(0.03)
        assert stable.instance.module.ticks == retired_ticks
    finally:
        runtime.stop()
        await runtime.wait_stopped()
        await runtime_task
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_snapshot_interval_serializes_retired_and_new_generations(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins" / "timer_probe"
    plugin_dir.mkdir(parents=True)
    plugin_path = plugin_dir / "plugin.py"
    plugin_path.write_text(_blocking_timer_plugin_source("1.0.0"), encoding="utf-8")
    event_bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    first = manager.generation("timer_probe")
    assert first is not None and isinstance(first.instance, ComposablePlugin)
    first_module = first.instance.module
    runtime = PluginJobRuntime(
        event_bus=event_bus,
        llm=cast(Any, object()),
        snapshot_store=manager.snapshot_store,
    )
    runtime_task = asyncio.create_task(runtime.run())
    try:
        await asyncio.wait_for(first_module.started.wait(), timeout=1)
        plugin_path.write_text(
            _timer_plugin_source("2.0.0", kind="interval"),
            encoding="utf-8",
        )
        assert await manager.prepare_candidate("timer_probe") is not None
        result = await manager.publish_prepared("timer_probe")
        assert result["publication_state"] == "committed"
        second = manager.generation("timer_probe")
        assert second is not None and isinstance(second.instance, ComposablePlugin)

        await asyncio.sleep(0.03)
        assert first_module.ticks == 1
        assert second.instance.module.ticks == 0
        assert first_module.handle.done is False

        first_module.release.set()
        await _wait_for_ticks(second.instance.module, 1)
        async with asyncio.timeout(1):
            while not first_module.handle.done:
                await asyncio.sleep(0.001)
    finally:
        first_module.release.set()
        runtime.stop()
        await runtime.wait_stopped()
        await runtime_task
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_snapshot_timeout_runs_once_per_stable_generation(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "timer_probe"
    plugin_dir.mkdir(parents=True)
    plugin_path = plugin_dir / "plugin.py"
    plugin_path.write_text(
        _timer_plugin_source("1.0.0", kind="timeout"),
        encoding="utf-8",
    )
    event_bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    first = manager.generation("timer_probe")
    assert first is not None and isinstance(first.instance, ComposablePlugin)
    runtime = PluginJobRuntime(
        event_bus=event_bus,
        llm=cast(Any, object()),
        snapshot_store=manager.snapshot_store,
    )
    paused = manager.snapshot_store.pause_admission()
    runtime_task = asyncio.create_task(runtime.run())
    try:
        await asyncio.sleep(0.03)
        assert first.instance.module.ticks == 0
        await manager.snapshot_store.resume(paused)
        paused = None
        await _wait_for_ticks(first.instance.module, 1)
        await asyncio.sleep(0.03)
        assert first.instance.module.ticks == 1
        assert first.instance.module.handle.done is True

        plugin_path.write_text(
            _timer_plugin_source("2.0.0", kind="timeout"),
            encoding="utf-8",
        )
        assert await manager.prepare_candidate("timer_probe") is not None
        result = await manager.publish_prepared("timer_probe")
        assert result["publication_state"] == "committed"
        second = manager.generation("timer_probe")
        assert second is not None and isinstance(second.instance, ComposablePlugin)
        await _wait_for_ticks(second.instance.module, 1)
        await asyncio.sleep(0.03)
        assert second.instance.module.ticks == 1
        assert second.instance.module.handle.done is True
    finally:
        await manager.snapshot_store.resume(paused)
        runtime.stop()
        await runtime.wait_stopped()
        await runtime_task
        await manager.terminate_all()


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


async def _wait_for_ticks(module: Any, expected: int) -> None:
    async with asyncio.timeout(1):
        while cast(int, module.ticks) < expected:
            await asyncio.sleep(0.001)


def _timer_plugin_source(version: str, *, kind: str) -> str:
    return (
        "from agent.plugin_composition import TIMER_SERVICE\n"
        "api_version = 3\n"
        "name = 'timer_probe'\n"
        f"version = {version!r}\n"
        "inject = (TIMER_SERVICE,)\n"
        "ticks = 0\n"
        "handle = None\n"
        "async def apply(ctx, config):\n"
        "    global handle\n"
        "    async def tick():\n"
        "        global ticks\n"
        "        ticks += 1\n"
        f"    handle = await ctx.require(TIMER_SERVICE).{kind}(\n"
        "        ctx, tick, 0.01, name='namespace-probe'\n"
        "    )\n"
    )


def _blocking_timer_plugin_source(version: str) -> str:
    return (
        "import asyncio\n"
        "from agent.plugin_composition import TIMER_SERVICE\n"
        "api_version = 3\n"
        "name = 'timer_probe'\n"
        f"version = {version!r}\n"
        "inject = (TIMER_SERVICE,)\n"
        "ticks = 0\n"
        "handle = None\n"
        "started = asyncio.Event()\n"
        "release = asyncio.Event()\n"
        "async def apply(ctx, config):\n"
        "    global handle\n"
        "    async def tick():\n"
        "        global ticks\n"
        "        ticks += 1\n"
        "        started.set()\n"
        "        await release.wait()\n"
        "    handle = await ctx.require(TIMER_SERVICE).interval(\n"
        "        ctx, tick, 0.01, name='namespace-probe'\n"
        "    )\n"
    )

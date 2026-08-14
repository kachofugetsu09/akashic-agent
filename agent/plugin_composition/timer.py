from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, cast

from agent.plugin_composition.model import ServiceKey

if TYPE_CHECKING:
    from agent.plugin_composition.context import Context

TimerCallback = Callable[[], object]
TimerClock = Callable[[], float]
TimerSleep = Callable[[float], Awaitable[None]]


class TimerHandle:
    """Cancel and join one Fiber-owned timer task."""

    __slots__ = ("_task",)

    def __init__(self, task: asyncio.Task[None]) -> None:
        self._task = task

    @property
    def done(self) -> bool:
        return self._task.done()

    async def aclose(self) -> None:
        if not self._task.done():
            _ = self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            return


class TimerService:
    """Schedule monotonic one-shot and serial periodic Fiber-owned callbacks."""

    def __init__(
        self,
        *,
        clock: TimerClock | None = None,
        sleep: TimerSleep = asyncio.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep

    async def timeout(
        self,
        ctx: "Context",
        callback: TimerCallback,
        delay: float,
        *,
        name: str = "timeout",
    ) -> TimerHandle:
        """Run one callback after a delay and bind cancellation to its Fiber."""

        seconds = _validate_delay(delay)
        _validate_callback(callback)
        task = await ctx.spawn(
            self._run_timeout(callback, seconds),
            name=f"timer:{name}",
        )
        return TimerHandle(task)

    async def interval(
        self,
        ctx: "Context",
        callback: TimerCallback,
        delay: float,
        *,
        name: str = "interval",
    ) -> TimerHandle:
        """Run one callback per fixed cadence without replay or overlap."""

        seconds = _validate_delay(delay)
        _validate_callback(callback)
        task = await ctx.spawn(
            self._run_interval(callback, seconds),
            name=f"timer:{name}",
        )
        return TimerHandle(task)

    async def _run_timeout(
        self,
        callback: TimerCallback,
        delay: float,
    ) -> None:
        await self._sleep(delay)
        await _invoke(callback)

    async def _run_interval(
        self,
        callback: TimerCallback,
        delay: float,
    ) -> None:
        # 1. Use one monotonic deadline instead of callback completion time.
        deadline = self._now() + delay
        while True:
            await self._sleep(max(0.0, deadline - self._now()))
            await _invoke(callback)

            # 2. Skip elapsed ticks so one slow callback cannot overlap itself.
            deadline = self._next_deadline(deadline, delay, self._now())

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return asyncio.get_running_loop().time()

    @staticmethod
    def _next_deadline(deadline: float, delay: float, now: float) -> float:
        elapsed_ticks = max(1, math.floor((now - deadline) / delay) + 1)
        return deadline + elapsed_ticks * delay


TIMER_SERVICE = ServiceKey[TimerService]("core.timer")


async def _invoke(callback: TimerCallback) -> None:
    result = callback()
    if inspect.isawaitable(result):
        _ = await cast(Awaitable[object], result)


def _validate_delay(delay: float) -> float:
    if isinstance(delay, bool) or not isinstance(delay, (int, float)):
        raise TypeError("Timer delay 必须是秒数")
    seconds = float(delay)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Timer delay 必须是大于 0 的有限秒数")
    return seconds


def _validate_callback(callback: TimerCallback) -> None:
    if not callable(callback):
        raise TypeError("Timer callback 必须可调用")

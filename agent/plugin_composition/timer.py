from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Mapping, cast

from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import CompositionError, ServiceKey

if TYPE_CHECKING:
    from agent.plugin_composition.context import Context

TimerCallback = Callable[[], object]
TimerClock = Callable[[], float]
TimerSleep = Callable[[float], Awaitable[None]]
TimerKind = Literal["timeout", "interval"]


@dataclass(slots=True)
class _RegistrationState:
    closed: bool = False
    completed: bool = False
    idle: asyncio.Event = field(default_factory=asyncio.Event)
    owner_closing: bool = False
    owner_task: asyncio.Task[object] | None = None

    def __post_init__(self) -> None:
        self.idle.set()


@dataclass(frozen=True, slots=True)
class TimerRegistration:
    """把稳定 Timer 键绑定到 Fiber 所有的 callback 状态。"""

    key: str
    plugin_id: str
    name: str
    kind: TimerKind
    delay: float
    callback: TimerCallback
    state: _RegistrationState = field(repr=False, compare=False)

    @property
    def active(self) -> bool:
        return not self.state.closed and not self.state.completed

    async def invoke(self) -> None:
        """仅在 active 时执行一次，并让 close 调用方等待结束。"""

        if not self.active:
            return
        self.state.idle.clear()
        self.state.owner_task = asyncio.current_task()
        try:
            await _invoke(self.callback)
        finally:
            if self.kind == "timeout":
                self.state.completed = True
            self.state.owner_task = None
            self.state.idle.set()


class TimerHandle:
    """关闭一个 Fiber 所有的本地 task 或 snapshot Timer 声明。"""

    __slots__ = ("_effect", "_state", "_task")

    def __init__(
        self,
        task: asyncio.Task[None] | None = None,
        *,
        effect: Effect | None = None,
        state: _RegistrationState | None = None,
    ) -> None:
        self._task = task
        self._effect = effect
        self._state = state

    @property
    def done(self) -> bool:
        if self._task is not None:
            return self._task.done()
        assert self._state is not None
        return self._state.closed or self._state.completed

    async def aclose(self) -> None:
        if self._task is not None:
            if not self._task.done():
                _ = self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                return
            return
        assert self._effect is not None
        assert self._state is not None
        if self._state.owner_task is asyncio.current_task():
            self._state.owner_closing = True
        await self._effect.aclose()


class TimerService:
    """管理本地 Timer，或为 stable snapshot 冻结声明。"""

    def __init__(
        self,
        *,
        clock: TimerClock | None = None,
        sleep: TimerSleep = asyncio.sleep,
        snapshot_owned: bool = False,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._snapshot_owned = snapshot_owned
        self._next_token = 1
        self._registrations: dict[int, TimerRegistration] = {}
        self._frozen: Mapping[str, TimerRegistration] | None = None

    @classmethod
    def for_snapshot(cls) -> "TimerService":
        """为 Core stable snapshot 调度器收集声明。"""

        return cls(snapshot_owned=True)

    async def timeout(
        self,
        ctx: "Context",
        callback: TimerCallback,
        delay: float,
        *,
        name: str = "timeout",
    ) -> TimerHandle:
        """Run one callback after a delay and bind cancellation to its Fiber."""

        seconds, checked_name = _validate_registration(callback, delay, name)
        if self._snapshot_owned:
            return await self._declare(
                ctx,
                callback,
                seconds,
                kind="timeout",
                name=checked_name,
            )
        task = await ctx.spawn(
            self._run_timeout(callback, seconds),
            name=f"timer:{checked_name}",
        )
        return TimerHandle(task=task)

    async def interval(
        self,
        ctx: "Context",
        callback: TimerCallback,
        delay: float,
        *,
        name: str = "interval",
    ) -> TimerHandle:
        """Run one callback per fixed cadence without replay or overlap."""

        seconds, checked_name = _validate_registration(callback, delay, name)
        if self._snapshot_owned:
            return await self._declare(
                ctx,
                callback,
                seconds,
                kind="interval",
                name=checked_name,
            )
        task = await ctx.spawn(
            self._run_interval(callback, seconds),
            name=f"timer:{checked_name}",
        )
        return TimerHandle(task=task)

    def freeze(self) -> Mapping[str, TimerRegistration]:
        """按稳定的 plugin/name 键冻结 snapshot 声明。"""

        if self._frozen is not None:
            return self._frozen
        frozen: dict[str, TimerRegistration] = {}
        for token in sorted(self._registrations):
            registration = self._registrations[token]
            if registration.key in frozen:
                raise RuntimeError(f"插件 Timer 稳定键重复: {registration.key}")
            frozen[registration.key] = registration
        self._frozen = MappingProxyType(frozen)
        return self._frozen

    async def _declare(
        self,
        ctx: "Context",
        callback: TimerCallback,
        delay: float,
        *,
        kind: TimerKind,
        name: str,
    ) -> TimerHandle:
        """把一个 snapshot Timer 注册为插件 Fiber 的 Effect。"""

        # 1. 把 callback 和关闭状态绑定到一个稳定插件键
        state = _RegistrationState()
        registration = TimerRegistration(
            key=f"{ctx.runtime.plugin_id}:{name}",
            plugin_id=ctx.runtime.plugin_id,
            name=name,
            kind=kind,
            delay=delay,
            callback=callback,
            state=state,
        )

        # 2. 插件 Fiber dispose 时移除声明并等待 callback 结束
        def setup() -> Callable[[], Awaitable[None]]:
            if self._frozen is not None:
                raise CompositionError(
                    "TIMER_DECLARATIONS_FROZEN",
                    "插件 Timer 声明已冻结，不能在 snapshot 发布后新增",
                )
            if any(
                item.key == registration.key
                for item in self._registrations.values()
            ):
                raise CompositionError(
                    "DUPLICATE_PLUGIN_TIMER",
                    f"插件 Timer 稳定键重复: {registration.key}",
                )
            token = self._next_token
            self._next_token += 1
            self._registrations[token] = registration

            async def cleanup() -> None:
                _ = self._registrations.pop(token, None)
                state.closed = True
                if not state.owner_closing:
                    _ = await state.idle.wait()

            return cleanup

        effect = await ctx.effect(setup, label=f"timer:{kind}:{name}")
        return TimerHandle(effect=effect, state=state)

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


def _validate_registration(
    callback: TimerCallback,
    delay: float,
    name: str,
) -> tuple[float, str]:
    _validate_callback(callback)
    if not isinstance(name, str):
        raise TypeError("Timer name 必须是字符串")
    if not name or name != name.strip():
        raise ValueError("Timer name 必须是非空且无首尾空白的字符串")
    return _validate_delay(delay), name

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol


class TimerStatus(StrEnum):
    FIRED = "fired"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TimerReceipt:
    """Record the settled fact of one deadline wait."""

    timer_id: str
    deadline: datetime
    settled_at: datetime
    status: TimerStatus

    def __post_init__(self) -> None:
        if not self.timer_id:
            raise ValueError("timer_id 不能为空")
        if self.deadline.tzinfo is None or self.settled_at.tzinfo is None:
            raise ValueError("Timer 时间必须包含时区")
        object.__setattr__(self, "deadline", self.deadline.astimezone(UTC))
        object.__setattr__(self, "settled_at", self.settled_at.astimezone(UTC))
        object.__setattr__(self, "status", TimerStatus(self.status))


class TimerHandle(Protocol):
    """Wait, cancel, and clean up one deadline registration."""

    @property
    def id(self) -> str: ...

    async def result(self) -> TimerReceipt: ...

    async def cancel(self) -> TimerReceipt: ...

    async def cleanup(self) -> None: ...


class OneShotTimer(Protocol):
    """Register exactly one wakeup without recurrence or retry semantics."""

    def schedule(self, deadline: datetime) -> TimerHandle: ...


Clock = Callable[[], datetime]
Sleeper = Callable[[float], Awaitable[None]]


class AsyncioOneShotTimer:
    """Settle each registered deadline exactly once as fired or cancelled."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleeper = sleeper

    def schedule(self, deadline: datetime) -> TimerHandle:
        if deadline.tzinfo is None:
            raise ValueError("Timer deadline 必须包含时区")
        return _AsyncioTimerHandle(
            "timer:" + secrets.token_hex(16),
            deadline.astimezone(UTC),
            self._clock,
            self._sleeper,
        )


class _AsyncioTimerHandle:
    """Own one asyncio wait and preserve its terminal receipt across callers."""

    def __init__(
        self,
        timer_id: str,
        deadline: datetime,
        clock: Clock,
        sleeper: Sleeper,
    ) -> None:
        self._id = timer_id
        self._deadline = deadline
        self._clock = clock
        self._sleeper = sleeper
        self._task = asyncio.create_task(self._wait(), name=timer_id)

    @property
    def id(self) -> str:
        return self._id

    async def result(self) -> TimerReceipt:
        return await asyncio.shield(self._task)

    async def cancel(self) -> TimerReceipt:
        if not self._task.done():
            _ = self._task.cancel()
        return await asyncio.shield(self._task)

    async def cleanup(self) -> None:
        _ = await self.cancel()

    async def _wait(self) -> TimerReceipt:
        """Wait until the deadline and translate cancellation into one receipt."""

        try:
            # 1. Compute delay once; recurrence and clock reconciliation belong outside.
            delay = max(0.0, (self._deadline - self._now()).total_seconds())
            await self._sleeper(delay)
            status = TimerStatus.FIRED
        except asyncio.CancelledError:
            status = TimerStatus.CANCELLED

        # 2. Freeze the single terminal fact for all future result/cancel calls.
        return TimerReceipt(
            timer_id=self._id,
            deadline=self._deadline,
            settled_at=self._now(),
            status=status,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("Timer clock 必须返回带时区时间")
        return value.astimezone(UTC)

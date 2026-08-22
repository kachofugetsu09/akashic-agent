from __future__ import annotations

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

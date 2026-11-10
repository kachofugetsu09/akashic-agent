from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from agent.plugin_composition.model import ServiceKey


class TimerStatus(StrEnum):
    FIRED = "fired"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TimerReceipt:
    """记录一次等待的终态事实。"""

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
    """等待、取消并清理一次截止时间登记。"""

    @property
    def id(self) -> str: ...

    async def result(self) -> TimerReceipt: ...

    async def cancel(self) -> TimerReceipt: ...

    async def cleanup(self) -> None: ...


class OneShotTimer(Protocol):
    """只登记一次等待，周期和重试由调用方负责。"""

    def schedule(self, deadline: datetime) -> TimerHandle: ...




TIMERS = ServiceKey[OneShotTimer]("timers.v1")

"""定时能力的公开结构合同。

`core.timers` 提供一次性定时；插件需要在注解、结果判断与取消流程里命名
`TimerReceipt`/`TimerStatus`/`TimerHandle`/`OneShotTimer`。这些是纯值词汇与
Protocol，因此由合同层拥有；具体实现（`AsyncioOneShotTimer`）留在
`agent/control/timer.py`，按结构满足 Protocol。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable


class TimerStatus(StrEnum):
    """一次到期等待的终态。"""

    FIRED = "fired"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TimerReceipt:
    """记录一次到期等待已结算的事实。"""

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


@runtime_checkable
class TimerHandle(Protocol):
    """等待、取消并清理一次到期登记。"""

    @property
    def id(self) -> str:
        """本次登记的稳定标识。"""
        ...

    async def result(self) -> TimerReceipt:
        """等到该次登记结算。"""
        ...

    async def cancel(self) -> TimerReceipt:
        """取消该次登记。"""
        ...

    async def cleanup(self) -> None:
        """释放该次登记占用的资源。"""
        ...


@runtime_checkable
class OneShotTimer(Protocol):
    """只登记一次唤醒，不递归、不重试。"""

    def schedule(self, deadline: datetime) -> TimerHandle:
        """登记一个到期时间并返回句柄。"""
        ...


Clock = Callable[[], datetime]
Sleeper = Callable[[float], Awaitable[None]]

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from agent.control.models import TurnItem, TurnRequest, TurnUsage


@dataclass(frozen=True)
class ControlExecutionResult:
    response: str
    items: list[TurnItem] = field(default_factory=list[TurnItem])
    deltas: list[str] = field(default_factory=list[str])
    usage: TurnUsage | None = None
    assistant_data: dict[str, object] = field(default_factory=dict[str, object])


TurnExecutor = Callable[[TurnRequest], Awaitable[str | ControlExecutionResult]]


@dataclass(frozen=True, slots=True)
class TurnUserInput:
    """保存同一 turn 内一条已经准入的用户输入。"""

    item_id: str
    ordinal: int
    content: str
    media: tuple[str, ...]
    metadata: dict[str, object]
    timestamp: datetime


class InputLock(Protocol):
    """向 reasoner 提供已持久化的 interaction 输入并原子封口。"""

    async def lock(self) -> None: ...

    def used_inputs(self) -> tuple[TurnUserInput, ...]: ...

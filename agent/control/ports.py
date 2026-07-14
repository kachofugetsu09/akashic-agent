from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agent.control.models import TurnItem, TurnRequest, TurnUsage


@dataclass(frozen=True)
class ControlExecutionResult:
    response: str
    items: list[TurnItem] = field(default_factory=list[TurnItem])
    deltas: list[str] = field(default_factory=list[str])
    usage: TurnUsage | None = None
    assistant_data: dict[str, object] = field(default_factory=dict[str, object])


TurnExecutor = Callable[[TurnRequest], Awaitable[str | ControlExecutionResult]]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class TurnOutbound:
    session_key: str
    content: str


@dataclass
class TurnTrace:
    source: Literal["passive", "proactive"]
    model: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    retrieval: dict | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TurnSideEffect(Protocol):
    async def run(self) -> None: ...


@dataclass
class TurnResult:
    decision: Literal["reply", "skip"]
    outbound: TurnOutbound | None
    evidence: list[str] = field(default_factory=list)
    trace: TurnTrace | None = None
    side_effects: list[Any] = field(default_factory=list)
    success_side_effects: list[Any] = field(default_factory=list)
    failure_side_effects: list[Any] = field(default_factory=list)

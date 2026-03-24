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
    # 通用副作用：无论发送成功/失败都执行（常用于预发送状态落地）。
    side_effects: list[Any] = field(default_factory=list)
    # 成功副作用：仅在 outbound 成功发送后执行。
    success_side_effects: list[Any] = field(default_factory=list)
    # 失败副作用：仅在 outbound 发送失败后执行。
    failure_side_effects: list[Any] = field(default_factory=list)

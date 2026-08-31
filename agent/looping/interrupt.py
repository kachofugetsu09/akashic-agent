"""Legacy Channel interrupt port and active-turn progress state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass
class ActiveTurnState:
    """Track transient progress for one running legacy executor call."""

    session_key: str
    partial_reply: str = ""
    partial_thinking: str | None = None
    tools_used: list[str] = field(default_factory=list)
    tool_chain_partial: list[dict[str, object]] = field(default_factory=list)


@dataclass
class InterruptResult:
    """request_interrupt() 的返回值。"""

    status: Literal["interrupted", "idle"]
    session_key: str = ""
    message: str = ""


class InterruptController(Protocol):
    """Expose the narrow channel interrupt contract."""

    def request_interrupt(
        self,
        session_key: str,
        sender: str = "",
        command: str = "/stop",
    ) -> InterruptResult: ...

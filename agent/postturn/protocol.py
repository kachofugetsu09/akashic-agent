from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent.core.types import ToolCallGroup

if TYPE_CHECKING:
    from agent.core.runtime_support import SessionLike


@dataclass
class PostTurnEvent:
    session_key: str
    channel: str
    chat_id: str
    user_message: str
    assistant_response: str
    tools_used: list[str]
    tool_chain: list[ToolCallGroup]
    session: "SessionLike"
    timestamp: datetime | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class PostTurnPipeline(Protocol):
    def schedule(self, event: PostTurnEvent) -> None: ...

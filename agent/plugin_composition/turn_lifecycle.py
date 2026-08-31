from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.plugin_composition.events import SerialEventKey


@dataclass
class BeforeTurnCtx:
    """Public mutable context for the pre-react plugin lifecycle seam."""

    session_key: str
    channel: str
    chat_id: str
    content: str
    timestamp: datetime
    history_messages: tuple[Any, ...]
    turn_id: str | None = field(default=None, kw_only=True)
    skill_names: list[str] = field(default_factory=list)
    abort: bool = False
    abort_reply: str = ""
    extra_hints: list[str] = field(default_factory=list)
    extra_metadata: dict[str, Any] = field(default_factory=dict)


CONTEXT_PREPARED_EVENT = SerialEventKey[BeforeTurnCtx, object](
    "turn.context_prepared"
)

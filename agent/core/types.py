from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""


@dataclass
class ToolCallGroup:
    text: str
    calls: list[ToolCall] = field(default_factory=list)


@dataclass
class HistoryMessage:
    role: str
    content: str
    tools_used: list[str] = field(default_factory=list)
    tool_chain: list[ToolCallGroup] = field(default_factory=list)


@dataclass
class ContextBundle:
    skill_mentions: list[str] = field(default_factory=list)
    history_messages: list[Any] = field(default_factory=list)


@dataclass
class ContextRequest:
    history: list[dict[str, Any]]
    current_message: str
    multimodal: bool
    media: list[str] | None = None
    skill_names: list[str] | None = None
    channel: str | None = None
    chat_id: str | None = None
    message_timestamp: datetime | None = None
    disabled_sections: set[str] | None = None
    turn_injection_prompt: str | None = None


@dataclass
class ReasonerResult:
    reply: str
    thinking: str | None = None
    streamed: bool = False
    tools_used: list[str] = field(default_factory=list)
    tools_unlocked: list[str] = field(default_factory=list)
    tool_chain: list[dict[str, Any]] = field(default_factory=list)
    media: list[str] = field(default_factory=list)
    visible_names: set[str] | None = None
    react_stats: dict[str, Any] = field(default_factory=dict)
    model_state: dict[str, Any] | None = None
    mobile_attention: Literal["confirmation"] | None = None

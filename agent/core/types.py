from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any



@dataclass
class InboundMessage:
    channel: str
    session_key: str
    sender: str
    content: str
    media: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
    channel: str
    session_key: str
    content: str
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ContextBundle:
    history: list[ChatMessage] = field(default_factory=list)
    memory_blocks: list[str] = field(default_factory=list)
    skill_mentions: list[str] = field(default_factory=list)
    retrieved_memory_block: str = ""
    retrieval_trace_raw: Any | None = None
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    history_messages: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    reply: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None


@dataclass
class ReasonerResult:
    reply: str
    invocations: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnRecord:
    msg: InboundMessage
    reply: str
    invocations: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

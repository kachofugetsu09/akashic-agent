from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _empty_media() -> list[str]:
    return []


def _empty_metadata() -> dict[str, Any]:
    return {}


def _empty_skill_names() -> list[str]:
    return []


@dataclass(frozen=True)
class TurnStarted:
    session_key: str
    channel: str
    chat_id: str
    content: str
    timestamp: datetime


@dataclass(frozen=True)
class TurnCompleted:
    session_key: str
    channel: str
    chat_id: str
    reply: str
    tools_used: list[str]
    thinking: str | None = None


@dataclass(frozen=True)
class StreamDeltaReady:
    session_key: str
    channel: str
    chat_id: str
    content_delta: str = ""
    thinking_delta: str = ""


@dataclass
class BeforeReasoning:
    session_key: str
    channel: str
    chat_id: str
    content: str
    skill_names: list[str] = field(default_factory=_empty_skill_names)
    retrieved_memory_block: str = ""


@dataclass(frozen=True)
class TurnPersisted:
    session_key: str
    channel: str
    chat_id: str
    user_message: str | None
    assistant_response: str
    tools_used: list[str]
    thinking: str | None = None


@dataclass(frozen=True)
class ToolCallStarted:
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallCompleted:
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    final_arguments: dict[str, Any]
    status: str
    result_preview: str


@dataclass
class BeforeDispatch:
    channel: str
    chat_id: str
    content: str
    thinking: str | None = None
    media: list[str] = field(default_factory=_empty_media)
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)

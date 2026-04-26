from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _empty_media() -> list[str]:
    return []


def _empty_metadata() -> dict[str, Any]:
    return {}


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


@dataclass
class BeforeDispatch:
    channel: str
    chat_id: str
    content: str
    thinking: str | None = None
    media: list[str] = field(default_factory=_empty_media)
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)

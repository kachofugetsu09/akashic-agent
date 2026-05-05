from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class RetrievalHitSummary:
    item_id: str
    memory_type: str
    score: float
    summary: str
    injected: bool
    confidence_label: str = ""
    forced: bool = False


@dataclass(frozen=True)
class RetrievalCompleted:
    session_key: str
    channel: str
    chat_id: str
    query: str
    orig_query: str | None
    hits: list[RetrievalHitSummary]
    injected_count: int
    route_decision: str | None
    aux_queries: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class MemoryWritten:
    session_key: str
    channel: str
    chat_id: str
    action: Literal["write", "supersede"]
    source_ref: str
    memory_type: str | None = None
    item_id: str | None = None
    summary: str | None = None
    superseded_ids: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class TurnIngested:
    session_key: str
    channel: str
    chat_id: str
    user_message: str
    assistant_response: str
    tool_chain: list[dict[str, object]]
    source_ref: str

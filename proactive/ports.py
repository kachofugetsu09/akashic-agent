from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from proactive.decide import DefaultDecidePort
from proactive.memory_retrieval import DefaultMemoryRetrievalPort
from proactive.sensor import Sensor

DefaultSensePort = Sensor


@dataclass
class ProactiveSourceRef:
    item_id: str
    source_type: str
    source_name: str
    title: str
    url: str | None = None
    published_at: str | None = None


@dataclass
class ProactiveSendMeta:
    evidence_item_ids: list[str] = field(default_factory=list)
    source_refs: list[ProactiveSourceRef] = field(default_factory=list)
    state_summary_tag: str = "none"


@dataclass
class RecentProactiveMessage:
    content: str
    timestamp: datetime | None = None
    state_summary_tag: str = "none"
    source_refs: list[ProactiveSourceRef] = field(default_factory=list)

@dataclass
class ProactiveRetrievedMemory:
    query: str = ""
    block: str = ""
    item_ids: list[str] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)
    procedure_hits: int = 0
    history_hits: int = 0
    history_channel_open: bool = False
    history_gate_reason: str = "disabled"
    history_scope_mode: str = "disabled"
    fallback_reason: str = ""
    preference_block: str = ""  # 偏好专项 RAG 结果，独立于 procedure+event 的 block

    @classmethod
    def empty(cls, fallback_reason: str = "") -> "ProactiveRetrievedMemory":
        return cls(fallback_reason=fallback_reason)

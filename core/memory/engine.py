from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class EngineProfile(str, Enum):
    RICH_MEMORY_ENGINE = "rich_memory_engine"
    CLASSIC_MEMORY_SERVICE = "classic_memory_service"
    WORKFLOW_MEMORY_ENGINE = "workflow_memory_engine"
    CONTEXT_RESOURCE_ENGINE = "context_resource_engine"


class MemoryCapability(str, Enum):
    INGEST_TEXT = "ingest.text"
    INGEST_MESSAGES = "ingest.messages"
    INGEST_RESOURCE = "ingest.resource"
    RETRIEVE_SEMANTIC = "retrieve.semantic"
    RETRIEVE_CONTEXT_BLOCK = "retrieve.context_block"
    RETRIEVE_STRUCTURED_HITS = "retrieve.structured_hits"
    MANAGE_HISTORY = "manage.history"
    MANAGE_UPDATE = "manage.update"
    MANAGE_DELETE = "manage.delete"
    ENRICH_GRAPH_RELATIONS = "enrich.graph_relations"
    SEMANTICS_RICH_MEMORY = "semantics.rich_memory"


@dataclass(frozen=True)
class MemoryScope:
    session_key: str = ""
    channel: str = ""
    chat_id: str = ""


@dataclass(frozen=True)
class MemoryEngineDescriptor:
    name: str
    profile: EngineProfile
    capabilities: frozenset[MemoryCapability]
    notes: dict[str, object] = field(default_factory=dict)


@dataclass
class MemoryIngestRequest:
    content: object
    source_kind: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    hints: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class MemoryIngestResult:
    accepted: bool
    created_ids: list[str] = field(default_factory=list)
    summary: str = ""
    raw: dict[str, object] = field(default_factory=dict)


@dataclass
class MemoryHit:
    id: str
    summary: str
    content: str
    score: float
    source_ref: str
    engine_kind: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class MemoryEngineRetrieveRequest:
    query: str
    context: dict[str, object] = field(default_factory=dict)
    scope: MemoryScope = field(default_factory=MemoryScope)
    mode: str = "default"
    hints: dict[str, object] = field(default_factory=dict)
    top_k: int | None = None


@dataclass
class MemoryEngineRetrieveResult:
    text_block: str
    hits: list[MemoryHit] = field(default_factory=list)
    trace: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class MemoryEngine(Protocol):
    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult: ...

    async def retrieve(
        self, request: MemoryEngineRetrieveRequest
    ) -> MemoryEngineRetrieveResult: ...

    def describe(self) -> MemoryEngineDescriptor: ...

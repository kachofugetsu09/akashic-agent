from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    injected: bool = False


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


@dataclass(frozen=True)
class RememberRequest:
    summary: str
    memory_type: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    source_ref: str = "memorize_tool"
    raw_extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RememberResult:
    item_id: str
    actual_type: str
    write_status: str = "new"
    superseded_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ForgetRequest:
    ids: list[str]


@dataclass(frozen=True)
class ForgetResult:
    superseded_ids: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    items: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class InterestRetrievalRequest:
    query: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    top_k: int = 2


@dataclass
class InterestRetrievalResult:
    text_block: str = ""
    hits: list[dict[str, object]] = field(default_factory=list)
    trace: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplicitRetrievalRequest:
    query: str
    memory_type: str = ""
    search_mode: str = "semantic"
    limit: int = 8
    time_start: datetime | None = None
    time_end: datetime | None = None
    scope: MemoryScope = field(default_factory=MemoryScope)


@dataclass
class ExplicitRetrievalResult:
    hits: list[dict[str, object]] = field(default_factory=list)
    trace: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class MemoryIngestApi(Protocol):
    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult: ...


@runtime_checkable
class MemoryRetrievalApi(Protocol):
    async def retrieve(
        self, request: MemoryEngineRetrieveRequest
    ) -> MemoryEngineRetrieveResult: ...

    async def retrieve_explicit(
        self, request: ExplicitRetrievalRequest
    ) -> ExplicitRetrievalResult: ...

    async def retrieve_interest_block(
        self, request: InterestRetrievalRequest
    ) -> InterestRetrievalResult: ...


@runtime_checkable
class MemoryWriteApi(Protocol):
    async def remember(self, request: RememberRequest) -> RememberResult: ...

    async def forget(self, request: ForgetRequest) -> ForgetResult: ...

    def reinforce_items_batch(self, ids: list[str]) -> None: ...


@runtime_checkable
class MemoryAdminApi(Protocol):
    def describe(self) -> MemoryEngineDescriptor: ...

    def keyword_match_procedures(
        self, action_tokens: list[str]
    ) -> list[dict[str, object]]: ...

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]: ...

    def list_items_for_dashboard(
        self,
        *,
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, object]], int]: ...

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None: ...

    def update_item_for_dashboard(
        self,
        item_id: str,
        *,
        status: str | None = None,
        extra_json: dict[str, object] | None = None,
        source_ref: str | None = None,
        happened_at: str | None = None,
        emotional_weight: int | None = None,
    ) -> dict[str, object] | None: ...

    def delete_item(self, item_id: str) -> bool: ...

    def delete_items_batch(self, ids: list[str]) -> int: ...

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]: ...


@runtime_checkable
class MemoryEngine(
    MemoryIngestApi,
    MemoryRetrievalApi,
    MemoryWriteApi,
    MemoryAdminApi,
    Protocol,
):
    pass

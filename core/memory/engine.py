from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

MemoryQueryIntent = Literal["context", "answer", "timeline", "interest", "procedure"]
MemoryQueryEffect = Literal["stateful", "read_only"]


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
    notes: dict[str, object] = field(default_factory=dict[str, object])


@dataclass
class MemoryIngestRequest:
    content: object
    source_kind: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    hints: dict[str, object] = field(default_factory=dict[str, object])
    metadata: dict[str, object] = field(default_factory=dict[str, object])


@dataclass
class MemoryIngestResult:
    accepted: bool
    created_ids: list[str] = field(default_factory=list[str])
    summary: str = ""
    raw: dict[str, object] = field(default_factory=dict[str, object])


@dataclass
class EvidenceRef:
    kind: Literal["message", "message_range", "turn", "external"] = "message"
    refs: list[str] = field(default_factory=list[str])
    resolver: str = "session"
    source_ref: str = ""
    metadata: dict[str, object] = field(default_factory=dict[str, object])


@dataclass
class MemoryRecord:
    id: str
    kind: str
    summary: str
    score: float
    engine_kind: str
    evidence: list[EvidenceRef] = field(default_factory=list[EvidenceRef])
    signals: dict[str, object] = field(default_factory=dict[str, object])
    injected: bool = False


@dataclass(frozen=True)
class MemoryQueryFilters:
    kinds: tuple[str, ...] = ()
    time_start: datetime | None = None
    time_end: datetime | None = None
    hints: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kinds",
            tuple(str(item) for item in self.kinds if str(item).strip()),
        )
        object.__setattr__(self, "hints", MappingProxyType(dict(self.hints)))


@dataclass
class MemoryQuery:
    text: str
    # answer/timeline 由模型工具公开；context/interest/procedure 是 runtime 内部入口。
    intent: MemoryQueryIntent = "answer"
    effect: MemoryQueryEffect = "stateful"
    scope: MemoryScope = field(default_factory=MemoryScope)
    filters: MemoryQueryFilters = field(default_factory=MemoryQueryFilters)
    context: dict[str, object] = field(default_factory=dict[str, object])
    limit: int = 8
    timestamp: datetime | None = None


@dataclass
class MemoryQueryResult:
    text_block: str = ""
    records: list[MemoryRecord] = field(default_factory=list[MemoryRecord])
    trace: dict[str, object] = field(default_factory=dict[str, object])
    raw: dict[str, object] = field(default_factory=dict[str, object])


@dataclass(frozen=True)
class MemoryMutation:
    # remember 使用 summary/memory_kind/source_ref；forget 使用 ids。
    kind: Literal["remember", "forget"]
    scope: MemoryScope = field(default_factory=MemoryScope)
    summary: str = ""
    memory_kind: str = ""
    source_ref: str = ""
    ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ids",
            tuple(s for raw in self.ids if (s := str(raw).strip())),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass
class MemoryMutationResult:
    accepted: bool
    item_id: str = ""
    actual_kind: str = ""
    status: str = ""
    affected_ids: list[str] = field(default_factory=list[str])
    missing_ids: list[str] = field(default_factory=list[str])
    items: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    raw: dict[str, object] = field(default_factory=dict[str, object])


@dataclass(frozen=True)
class MemoryToolSpec:
    description: str
    parameters: dict[str, object]
    name: str = ""
    risk: Literal["read-only", "write", "external-side-effect"] = "read-only"
    search_hint: str = ""
    # 自定义工具类；必须接受 (engine, spec) 两个位置参数。
    # 留空时 register_memory_meta_tools 使用内置的标准工具类。
    tool_class: type | None = field(default=None, compare=False, hash=False)


@dataclass(frozen=True)
class MemoryToolProfile:
    recall: MemoryToolSpec | None = None
    memorize: MemoryToolSpec | None = None
    forget: MemoryToolSpec | None = None
    tools: tuple[MemoryToolSpec, ...] = field(default_factory=tuple)


@runtime_checkable
class MemoryIngestApi(Protocol):
    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult: ...


@runtime_checkable
class MemoryRetrievalApi(Protocol):
    async def query(self, request: MemoryQuery) -> MemoryQueryResult: ...


@runtime_checkable
class MemoryWriteApi(Protocol):
    async def mutate(self, request: MemoryMutation) -> MemoryMutationResult: ...

    def reinforce_items_batch(self, ids: list[str]) -> None: ...


@runtime_checkable
class MemoryAdminApi(Protocol):
    def describe(self) -> MemoryEngineDescriptor: ...

    # 工具注册也从 engine 读，避免 runtime 了解不同 engine 的工具语义。
    def tool_profile(self) -> MemoryToolProfile: ...

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

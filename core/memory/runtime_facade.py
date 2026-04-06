from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol, runtime_checkable

from core.memory.engine import (
    MemoryIngestResult,
    MemoryScope,
    RememberRequest,
    RememberResult,
)

if TYPE_CHECKING:
    from agent.core.types import HistoryMessage
    from agent.postturn.protocol import PostTurnEvent


@dataclass(frozen=True)
class ContextRetrievalRequest:
    message: str
    session_key: str
    channel: str
    chat_id: str
    history: list[HistoryMessage]
    session_metadata: dict[str, object]
    timestamp: datetime | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass
class ContextRetrievalResult:
    normative_hits: list[dict] = field(default_factory=list)
    episodic_hits: list[dict] = field(default_factory=list)
    injected_item_ids: list[str] = field(default_factory=list)
    text_block: str = ""
    trace: dict[str, object] = field(default_factory=dict)
    hyde_hypothesis: str | None = None
    scope_mode: str = ""
    sufficiency_trace: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class InterestRetrievalRequest:
    query: str
    scope: MemoryScope = field(default_factory=MemoryScope)
    top_k: int = 2


@dataclass
class InterestRetrievalResult:
    text_block: str = ""
    hits: list[dict] = field(default_factory=list)
    trace: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)


ContextRetriever = Callable[[ContextRetrievalRequest], Awaitable[ContextRetrievalResult]]
ConsolidationRunner = Callable[[object, bool, bool], Awaitable[None]]


@runtime_checkable
class MemoryRuntimeFacade(Protocol):
    def bind_context_retriever(self, retriever: ContextRetriever) -> None: ...

    def bind_retrieval_semantics(self, retriever: ContextRetriever) -> None: ...

    def bind_consolidation_runner(self, runner: ConsolidationRunner) -> None: ...

    async def ingest_post_turn(self, event: PostTurnEvent) -> MemoryIngestResult: ...

    async def retrieve_context(
        self, request: ContextRetrievalRequest
    ) -> ContextRetrievalResult: ...

    async def run_consolidation(
        self,
        session: object,
        *,
        archive_all: bool = False,
        await_vector_store: bool = False,
    ) -> None: ...

    async def retrieve_interest_block(
        self, request: InterestRetrievalRequest
    ) -> InterestRetrievalResult: ...

    async def remember_explicit(self, request: RememberRequest) -> RememberResult: ...

    def read_long_term_context(self) -> str: ...

    def read_self(self) -> str: ...

    def read_recent_history(self, *, max_chars: int = 0) -> str: ...

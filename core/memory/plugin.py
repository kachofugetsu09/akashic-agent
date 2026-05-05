from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from core.memory.engine import (
    EngineProfile,
    ExplicitRetrievalRequest,
    ExplicitRetrievalResult,
    ForgetRequest,
    ForgetResult,
    InterestRetrievalRequest,
    InterestRetrievalResult,
    MemoryAdminApi,
    MemoryEngine,
    MemoryEngineDescriptor,
    MemoryEngineRetrieveRequest,
    MemoryEngineRetrieveResult,
    MemoryIngestRequest,
    MemoryIngestResult,
    RememberRequest,
    RememberResult,
)

if TYPE_CHECKING:
    from agent.config_models import Config
    from agent.provider import LLMProvider
    from agent.tools.base import Tool
    from bus.event_bus import EventBus
    from core.memory.markdown import MarkdownMemoryRuntime
    from core.net.http import SharedHttpResources


@dataclass(frozen=True)
class MemoryPluginBuildDeps:
    config: "Config"
    workspace: Path
    provider: "LLMProvider"
    light_provider: "LLMProvider | None"
    http_resources: "SharedHttpResources"
    event_publisher: "EventBus | None"
    markdown: "MarkdownMemoryRuntime"


@dataclass(frozen=True)
class MemoryToolBundle:
    recall_memory: "Tool | None" = None
    memorize: "Tool | None" = None
    forget_memory: "Tool | None" = None


@dataclass
class MemoryPluginRuntime:
    engine: MemoryEngine
    tools: MemoryToolBundle = field(default_factory=MemoryToolBundle)
    closeables: list[object] = field(default_factory=lambda: [])
    admin: MemoryAdminApi | None = None


@runtime_checkable
class MemoryPlugin(Protocol):
    plugin_id: str

    def build(
        self,
        deps: MemoryPluginBuildDeps,
    ) -> MemoryPluginRuntime: ...


class DisabledMemoryEngine:
    DESCRIPTOR = MemoryEngineDescriptor(
        name="disabled",
        profile=EngineProfile.CONTEXT_RESOURCE_ENGINE,
        capabilities=frozenset(),
        notes={"reason": "semantic memory disabled"},
    )

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        return MemoryIngestResult(accepted=False, raw={"reason": "disabled"})

    async def retrieve(
        self,
        request: MemoryEngineRetrieveRequest,
    ) -> MemoryEngineRetrieveResult:
        return MemoryEngineRetrieveResult(text_block="", trace={"mode": "disabled"})

    async def retrieve_explicit(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        return ExplicitRetrievalResult(trace={"mode": "disabled"})

    async def retrieve_interest_block(
        self,
        request: InterestRetrievalRequest,
    ) -> InterestRetrievalResult:
        return InterestRetrievalResult(trace={"mode": "disabled"})

    async def remember(self, request: RememberRequest) -> RememberResult:
        raise RuntimeError("semantic memory disabled")

    async def forget(self, request: ForgetRequest) -> ForgetResult:
        return ForgetResult(missing_ids=list(request.ids))

    def reinforce_items_batch(self, ids: list[str]) -> None:
        return None

    def describe(self) -> MemoryEngineDescriptor:
        return self.DESCRIPTOR

    def keyword_match_procedures(
        self,
        action_tokens: list[str],
    ) -> list[dict[str, object]]:
        return []

    def list_events_by_time_range(
        self,
        time_start: datetime,
        time_end: datetime,
        *,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        return []

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
    ) -> tuple[list[dict[str, object]], int]:
        return [], 0

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        return None

    def update_item_for_dashboard(
        self,
        item_id: str,
        *,
        status: str | None = None,
        extra_json: dict[str, object] | None = None,
        source_ref: str | None = None,
        happened_at: str | None = None,
        emotional_weight: int | None = None,
    ) -> dict[str, object] | None:
        return None

    def delete_item(self, item_id: str) -> bool:
        return False

    def delete_items_batch(self, ids: list[str]) -> int:
        return 0

    def find_similar_items_for_dashboard(
        self,
        item_id: str,
        *,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> list[dict[str, object]]:
        return []

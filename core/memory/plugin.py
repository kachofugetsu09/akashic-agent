from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from core.memory.engine import (
    EngineProfile,
    MemoryAdminApi,
    MemoryEngine,
    MemoryEngineDescriptor,
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryMutation,
    MemoryMutationResult,
    MemoryQuery,
    MemoryQueryResult,
    MemoryToolProfile,
)

if TYPE_CHECKING:
    from agent.config_models import Config
    from agent.provider import LLMProvider
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


@dataclass
class MemoryPluginRuntime:
    engine: MemoryEngine
    closeables: list[object] = field(default_factory=list[object])
    admin: MemoryAdminApi | None = None


@runtime_checkable
class MemoryPlugin(Protocol):
    plugin_id: str

    def build(
        self,
        deps: MemoryPluginBuildDeps,
    ) -> MemoryPluginRuntime: ...


class DisabledMemoryEngine(MemoryEngine):
    DESCRIPTOR = MemoryEngineDescriptor(
        name="disabled",
        profile=EngineProfile.CONTEXT_RESOURCE_ENGINE,
        capabilities=frozenset(),
        notes={"reason": "semantic memory disabled"},
    )

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        return MemoryIngestResult(accepted=False, raw={"reason": "disabled"})

    async def query(
        self,
        request: MemoryQuery,
    ) -> MemoryQueryResult:
        return MemoryQueryResult(trace={"mode": "disabled", "intent": request.intent})

    async def mutate(self, request: MemoryMutation) -> MemoryMutationResult:
        if request.kind == "forget":
            return MemoryMutationResult(
                accepted=False,
                status="disabled",
                missing_ids=list(request.ids),
            )
        return MemoryMutationResult(accepted=False, status="disabled")

    def reinforce_items_batch(self, ids: list[str]) -> None:
        return None

    def describe(self) -> MemoryEngineDescriptor:
        return self.DESCRIPTOR

    def tool_profile(self) -> MemoryToolProfile:
        return MemoryToolProfile()

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

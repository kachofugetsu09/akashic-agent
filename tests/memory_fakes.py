from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from plugins.markdown_memory.store import MarkdownProfileStore
from core.memory.engine import (
    EngineProfile,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryMutation,
    MemoryMutationResult,
    MemoryQuery,
    MemoryQueryResult,
    MemoryToolProfile,
)


class FakeMemoryEngine:
    def __init__(self, workspace: Path | None = None) -> None:
        self._store = (
            MarkdownProfileStore(
                workspace / "memory/MEMORY.md",
                workspace / "memory/SELF.md",
                workspace / "memory/consolidation_writes.db",
            )
            if workspace is not None
            else None
        )
        self.retrieve_result = MemoryQueryResult(text_block="")

    def describe(self) -> MemoryEngineDescriptor:
        return MemoryEngineDescriptor(
            name="fake",
            profile=EngineProfile.CLASSIC_MEMORY_SERVICE,
            capabilities=frozenset({MemoryCapability.RETRIEVE_CONTEXT_BLOCK}),
        )

    def tool_profile(self) -> MemoryToolProfile:
        return MemoryToolProfile()

    async def query(
        self,
        request: MemoryQuery,
    ) -> MemoryQueryResult:
        return self.retrieve_result

    async def mutate(
        self,
        request: MemoryMutation,
    ) -> MemoryMutationResult:
        if request.kind == "forget":
            return MemoryMutationResult(accepted=False, missing_ids=list(request.ids))
        return MemoryMutationResult(
            accepted=True,
            item_id="mem-1",
            actual_kind=request.memory_kind,
            status="new",
        )

    def reinforce_items_batch(self, ids: list[str]) -> None:
        return None

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        return MemoryIngestResult(accepted=True)

    def read_long_term(self) -> str:
        return self._store.read_memory() if self._store is not None else ""

    def write_long_term(self, content: str) -> None:
        if self._store is not None:
            self._store.memory_path.write_text(content, encoding="utf-8")

    def read_self(self) -> str:
        return self._store.read_self() if self._store is not None else ""

    def write_self(self, content: str) -> None:
        if self._store is not None:
            self._store.self_path.write_text(content, encoding="utf-8")

    def backup_long_term(self, backup_name: str = "MEMORY.bak.md") -> None:
        return None

    def backup_self(self, backup_name: str = "SELF.bak.md") -> None:
        return None

    def get_memory_context(self) -> str:
        memory = self.read_long_term()
        return f"## Long-term Memory\n{memory}" if memory else ""

    def has_long_term_memory(self) -> bool:
        return bool(self.read_long_term().strip())
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

    def list_items_for_dashboard(self, **kwargs: Any) -> tuple[list[dict[str, object]], int]:
        return [], 0

    def get_item_for_dashboard(
        self,
        item_id: str,
        *,
        include_embedding: bool = False,
    ) -> dict[str, object] | None:
        return None

    def update_item_for_dashboard(self, item_id: str, **kwargs: Any) -> dict[str, object] | None:
        return None

    def delete_item(self, item_id: str) -> bool:
        return False

    def delete_items_batch(self, ids: list[str]) -> int:
        return 0

    def find_similar_items_for_dashboard(self, item_id: str, **kwargs: Any) -> list[dict[str, object]]:
        return []

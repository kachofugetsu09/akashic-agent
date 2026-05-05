from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from agent.memory import MemoryStore
from core.memory.engine import (
    ConsolidateRequest,
    ConsolidateResult,
    EngineProfile,
    ExplicitRetrievalRequest,
    ExplicitRetrievalResult,
    ForgetRequest,
    ForgetResult,
    InterestRetrievalRequest,
    InterestRetrievalResult,
    MemoryCapability,
    MemoryEngineDescriptor,
    MemoryEngineRetrieveRequest,
    MemoryEngineRetrieveResult,
    MemoryIngestRequest,
    MemoryIngestResult,
    RememberRequest,
    RememberResult,
)


class FakeMemoryEngine:
    def __init__(self, workspace: Path | None = None) -> None:
        self._store = MemoryStore(workspace) if workspace is not None else None
        self.consolidate_calls: list[ConsolidateRequest] = []
        self.retrieve_result = MemoryEngineRetrieveResult(text_block="")

    def describe(self) -> MemoryEngineDescriptor:
        return MemoryEngineDescriptor(
            name="fake",
            profile=EngineProfile.CLASSIC_MEMORY_SERVICE,
            capabilities=frozenset({MemoryCapability.RETRIEVE_CONTEXT_BLOCK}),
        )

    async def retrieve(
        self,
        request: MemoryEngineRetrieveRequest,
    ) -> MemoryEngineRetrieveResult:
        return self.retrieve_result

    async def retrieve_explicit(
        self,
        request: ExplicitRetrievalRequest,
    ) -> ExplicitRetrievalResult:
        return ExplicitRetrievalResult()

    async def retrieve_interest_block(
        self,
        request: InterestRetrievalRequest,
    ) -> InterestRetrievalResult:
        return InterestRetrievalResult()

    async def remember(self, request: RememberRequest) -> RememberResult:
        return RememberResult(item_id="mem-1", actual_type=request.memory_type)

    async def forget(self, request: ForgetRequest) -> ForgetResult:
        return ForgetResult(missing_ids=list(request.ids))

    def reinforce_items_batch(self, ids: list[str]) -> None:
        return None

    async def ingest(self, request: MemoryIngestRequest) -> MemoryIngestResult:
        return MemoryIngestResult(accepted=True)

    async def consolidate(self, request: ConsolidateRequest) -> ConsolidateResult:
        self.consolidate_calls.append(request)
        return ConsolidateResult()

    async def refresh_recent_turns(self, request) -> None:
        return None

    def read_long_term(self) -> str:
        return self._store.read_long_term() if self._store is not None else ""

    def write_long_term(self, content: str) -> None:
        if self._store is not None:
            self._store.write_long_term(content)

    def read_self(self) -> str:
        return self._store.read_self() if self._store is not None else ""

    def write_self(self, content: str) -> None:
        if self._store is not None:
            self._store.write_self(content)

    def read_recent_history(self, *, max_chars: int = 0) -> str:
        return self.read_history(max_chars=max_chars)

    def read_history(self, max_chars: int = 0) -> str:
        if self._store is None:
            return ""
        return self._store.read_history(max_chars=max_chars)

    def read_recent_context(self) -> str:
        return self._store.read_recent_context() if self._store is not None else ""

    def write_recent_context(self, content: str) -> None:
        if self._store is not None:
            self._store.write_recent_context(content)

    def backup_long_term(self, backup_name: str = "MEMORY.bak.md") -> None:
        return None

    def get_memory_context(self) -> str:
        return self._store.get_memory_context() if self._store is not None else ""

    def has_long_term_memory(self) -> bool:
        return bool(self.read_long_term().strip())

    def read_pending(self) -> str:
        return self._store.read_pending() if self._store is not None else ""

    def append_pending(self, facts: str) -> None:
        if self._store is not None:
            self._store.append_pending(facts)

    def append_pending_once(
        self,
        facts: str,
        source_ref: str,
        kind: str = "pending",
    ) -> bool:
        if self._store is None:
            return False
        return self._store.append_pending_once(
            facts,
            source_ref=source_ref,
            kind=kind,
        )

    def snapshot_pending(self) -> str:
        return self._store.snapshot_pending() if self._store is not None else ""

    def commit_pending_snapshot(self) -> None:
        if self._store is not None:
            self._store.commit_pending_snapshot()

    def rollback_pending_snapshot(self) -> None:
        if self._store is not None:
            self._store.rollback_pending_snapshot()

    def append_history(self, entry: str) -> None:
        if self._store is not None:
            self._store.append_history(entry)

    def append_history_once(
        self,
        entry: str,
        source_ref: str,
        kind: str = "history_entry",
    ) -> bool:
        if self._store is None:
            return False
        return self._store.append_history_once(
            entry,
            source_ref=source_ref,
            kind=kind,
        )

    def append_journal(
        self,
        date_str: str,
        entry: str,
        *,
        source_ref: str = "",
        kind: str = "journal",
    ) -> bool:
        if self._store is None:
            return False
        return self._store.append_journal(
            date_str,
            entry,
            source_ref=source_ref,
            kind=kind,
        )

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

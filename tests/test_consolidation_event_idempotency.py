from __future__ import annotations

import asyncio

from core.memory.events import ConsolidationCommitted
from plugins.default_memory.engine import DefaultMemoryEngine


class _CommitMarkerStore:
    def __init__(self) -> None:
        self.completed: dict[str, str] = {}

    def has_completed_consolidation_commit(self, *, source_ref: str, digest: str) -> bool:
        existing = self.completed.get(source_ref)
        if existing is None:
            return False
        if existing != digest:
            raise ValueError("digest conflict")
        return True

    def mark_consolidation_commit_completed(self, *, source_ref: str, digest: str) -> None:
        existing = self.completed.get(source_ref)
        if existing is not None and existing != digest:
            raise ValueError("digest conflict")
        self.completed[source_ref] = digest


class _ProbeEngine(DefaultMemoryEngine):
    def __init__(self) -> None:
        self._v2_store = _CommitMarkerStore()
        self._memorizer = object()
        self.provider_calls = 0
        self.saved_entries = 0

    def _require_memorizer(self):
        return self._memorizer

    async def _save_from_consolidation(self, **kwargs):
        self.saved_entries += 1

    async def _extract_implicit_long_term(self, **kwargs):
        self.provider_calls += 1
        return None


def test_consolidation_event_replay_skips_implicit_provider_after_success() -> None:
    engine = _ProbeEngine()
    event = ConsolidationCommitted(
        history_entry_payloads=[],
        source_ref="session:compaction:1",
        scope_channel="web",
        scope_chat_id="chat",
        conversation="conversation",
    )

    asyncio.run(engine._on_consolidation_committed(event))
    asyncio.run(engine._on_consolidation_committed(event))

    assert engine.provider_calls == 1
    assert engine.saved_entries == 0
    assert engine._v2_store.completed

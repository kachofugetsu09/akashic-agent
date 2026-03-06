"""
core/memory/port.py — Unified MemoryPort protocol + DefaultMemoryPort adapter

Design:
- MemoryPort is a Protocol that covers all memory read/write operations
  previously split between v1 (agent/memory.py, Markdown files) and
  v2 (memory2/, SQLite + vector search).
- DefaultMemoryPort wraps MemoryStore (v1) and optionally Memorizer /
  Retriever (v2) so all callers can depend on one interface.
- The underlying implementations (MemoryStore, MemoryStore2, Memorizer,
  Retriever) are NOT changed — this is a pure adapter layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent.memory import MemoryStore
    from memory2.memorizer import Memorizer
    from memory2.retriever import Retriever

logger = logging.getLogger(__name__)


# ── Protocol ─────────────────────────────────────────────────────────────────


@runtime_checkable
class MemoryPort(Protocol):
    """Unified read/write interface for all memory layers.

    v1 (Markdown files) operations:
      read_long_term / write_long_term   — MEMORY.md stable user profile
      read_self / write_self             — SELF.md Akashic self-model
      read_now / write_now               — NOW.md short-term state
      read_now_ongoing                   — NOW.md "近期进行中" section
      update_now_ongoing                 — mutate "近期进行中" section
      append_pending / read_pending      — PENDING.md incremental facts
      snapshot_pending                   — two-phase commit phase-1
      commit_pending_snapshot            — two-phase commit phase-2 (success)
      rollback_pending_snapshot          — two-phase commit phase-2 (fail)
      append_history                     — HISTORY.md event log
      get_memory_context                 — formatted context string for prompts
      has_long_term_memory               — bool: is MEMORY.md non-empty?

    v2 (SQLite + vector) operations:
      retrieve_related                   — vector search → list[dict]
      format_injection_block             — format retrieved items for prompt
      save_item                          — embed + upsert a single memory item
      save_from_consolidation            — bulk write from LLM consolidation
    """

    # ── v1: long-term profile (MEMORY.md) ─────────────────────────
    def read_long_term(self) -> str: ...
    def write_long_term(self, content: str) -> None: ...

    # ── v1: self-model (SELF.md) ───────────────────────────────────
    def read_self(self) -> str: ...
    def write_self(self, content: str) -> None: ...

    # ── v1: short-term state (NOW.md) ─────────────────────────────
    def read_now(self) -> str: ...
    def write_now(self, content: str) -> None: ...
    def read_now_ongoing(self) -> str: ...
    def update_now_ongoing(
        self, add: list[str], remove_keywords: list[str]
    ) -> None: ...

    # ── v1: pending facts buffer (PENDING.md) ─────────────────────
    def read_pending(self) -> str: ...
    def append_pending(self, facts: str) -> None: ...
    def snapshot_pending(self) -> str: ...
    def commit_pending_snapshot(self) -> None: ...
    def rollback_pending_snapshot(self) -> None: ...

    # ── v1: history log (HISTORY.md) ──────────────────────────────
    def append_history(self, entry: str) -> None: ...
    def read_history(self, max_chars: int = 0) -> str: ...

    # ── v1: context helpers ────────────────────────────────────────
    def get_memory_context(self) -> str: ...
    def has_long_term_memory(self) -> bool: ...

    # ── v2: vector retrieval ───────────────────────────────────────
    async def retrieve_related(
        self,
        query: str,
        memory_types: list[str] | None = None,
    ) -> list[dict]: ...

    def format_injection_block(self, items: list[dict]) -> str: ...

    # ── v2: write ─────────────────────────────────────────────────
    async def save_item(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
    ) -> str: ...

    async def save_from_consolidation(
        self,
        history_entry: str,
        behavior_updates: list[dict],
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
    ) -> None: ...

    def supersede_batch(self, ids: list[str]) -> None: ...


# ── Adapter ───────────────────────────────────────────────────────────────────


class DefaultMemoryPort:
    """Adapts MemoryStore (v1) + optional Memorizer/Retriever (v2).

    Pass memorizer=None and retriever=None to run v1-only (all v2
    methods become safe no-ops or return empty results).
    """

    def __init__(
        self,
        store: "MemoryStore",
        memorizer: "Memorizer | None" = None,
        retriever: "Retriever | None" = None,
    ) -> None:
        # 1. Store v1 and v2 dependencies
        self._store = store
        self._memorizer = memorizer
        self._retriever = retriever

    # ── v1: long-term profile ──────────────────────────────────────

    def read_long_term(self) -> str:
        return self._store.read_long_term()

    def write_long_term(self, content: str) -> None:
        self._store.write_long_term(content)

    # ── v1: self-model ─────────────────────────────────────────────

    def read_self(self) -> str:
        return self._store.read_self()

    def write_self(self, content: str) -> None:
        self._store.write_self(content)

    # ── v1: short-term state ───────────────────────────────────────

    def read_now(self) -> str:
        return self._store.read_now()

    def write_now(self, content: str) -> None:
        self._store.write_now(content)

    def read_now_ongoing(self) -> str:
        return self._store.read_now_ongoing()

    def update_now_ongoing(self, add: list[str], remove_keywords: list[str]) -> None:
        self._store.update_now_ongoing(add, remove_keywords)

    # ── v1: pending facts buffer ───────────────────────────────────

    def read_pending(self) -> str:
        return self._store.read_pending()

    def append_pending(self, facts: str) -> None:
        self._store.append_pending(facts)

    def snapshot_pending(self) -> str:
        return self._store.snapshot_pending()

    def commit_pending_snapshot(self) -> None:
        self._store.commit_pending_snapshot()

    def rollback_pending_snapshot(self) -> None:
        self._store.rollback_pending_snapshot()

    # ── v1: history log ────────────────────────────────────────────

    def append_history(self, entry: str) -> None:
        self._store.append_history(entry)

    def read_history(self, max_chars: int = 0) -> str:
        """Read HISTORY.md; if max_chars > 0, return only the last max_chars."""
        try:
            if not self._store.history_file.exists():
                return ""
            text = self._store.history_file.read_text(encoding="utf-8")
            if max_chars > 0 and len(text) > max_chars:
                return text[-max_chars:]
            return text
        except Exception:
            return ""

    # ── v1: context helpers ────────────────────────────────────────

    def get_memory_context(self) -> str:
        return self._store.get_memory_context()

    def has_long_term_memory(self) -> bool:
        try:
            return bool(self._store.read_long_term().strip())
        except Exception:
            return False

    # ── v2: vector retrieval ───────────────────────────────────────

    async def retrieve_related(
        self,
        query: str,
        memory_types: list[str] | None = None,
    ) -> list[dict]:
        """Embed query and return top-k memory items; empty list if no retriever."""
        if not self._retriever:
            return []
        try:
            return await self._retriever.retrieve(query, memory_types=memory_types)
        except Exception as e:
            logger.warning("[memory_port] retrieve_related failed: %s", e)
            return []

    def format_injection_block(self, items: list[dict]) -> str:
        """Format retrieved items for prompt injection; empty string if no retriever."""
        if not self._retriever:
            return ""
        return self._retriever.format_injection_block(items)

    # ── v2: write ──────────────────────────────────────────────────

    async def save_item(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
    ) -> str:
        """Embed and upsert a single memory item; returns '' if no memorizer."""
        if not self._memorizer:
            return ""
        try:
            return await self._memorizer.save_item(
                summary=summary,
                memory_type=memory_type,
                extra=extra,
                source_ref=source_ref,
                happened_at=happened_at,
            )
        except Exception as e:
            logger.warning("[memory_port] save_item failed: %s", e)
            return ""

    async def save_from_consolidation(
        self,
        history_entry: str,
        behavior_updates: list[dict],
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
    ) -> None:
        """Write consolidation output to SQLite; no-op if no memorizer."""
        if not self._memorizer:
            return
        try:
            await self._memorizer.save_from_consolidation(
                history_entry=history_entry,
                behavior_updates=behavior_updates,
                source_ref=source_ref,
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
            )
        except Exception as e:
            logger.warning("[memory_port] save_from_consolidation failed: %s", e)

    def supersede_batch(self, ids: list[str]) -> None:
        if self._memorizer:
            self._memorizer.supersede_batch(ids)

    # ── pass-through: expose v1 store for MemoryOptimizer ─────────

    @property
    def _v1_store(self) -> "MemoryStore":
        """Direct access to the underlying MemoryStore for MemoryOptimizer.

        MemoryOptimizer needs access to memory_file.with_suffix('.md.bak')
        and history_file directly. Rather than duplicate those paths in
        MemoryPort, we expose the store only to the optimizer (not to
        general callers).
        """
        return self._store

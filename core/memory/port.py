"""
core/memory/port.py — 文件层 memory port + 存量向量适配器
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
    """文件层读写接口，外加少量仍在外围使用的向量能力。

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

    存量向量 operations:
      retrieve_related                   — vector search → list[dict]
      build_injection_block              — build prompt block + ids
      save_item                          — embed + upsert a single memory item
      save_from_consolidation            — bulk write from LLM consolidation
    """

    # ── v1: long-term profile (MEMORY.md) ─────────────────────────
    def read_long_term(self) -> str: ...
    def read_profile(self) -> str: ...
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
    def read_recent_context(self) -> str: ...
    def write_recent_context(self, content: str) -> None: ...

    # ── v1: pending facts buffer (PENDING.md) ─────────────────────
    def read_pending(self) -> str: ...
    def append_pending(self, facts: str) -> None: ...
    def append_pending_once(
        self,
        facts: str,
        source_ref: str,
        kind: str = "pending",
    ) -> bool: ...
    def snapshot_pending(self) -> str: ...
    def commit_pending_snapshot(self) -> None: ...
    def rollback_pending_snapshot(self) -> None: ...

    # ── v1: history log (HISTORY.md) ──────────────────────────────
    def append_history(self, entry: str) -> None: ...
    def append_history_once(
        self,
        entry: str,
        source_ref: str,
        kind: str = "history_entry",
    ) -> bool: ...
    def read_history(self, max_chars: int = 0) -> str: ...

    # ── v1: context helpers ────────────────────────────────────────
    def get_memory_context(self) -> str: ...
    def has_long_term_memory(self) -> bool: ...

    # ── v2: vector retrieval ───────────────────────────────────────
    async def retrieve_related(
        self,
        query: str,
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
    ) -> list[dict]: ...

    async def embed_query(self, query: str) -> list[float]: ...

    async def retrieve_related_vec(
        self,
        query_vec: list[float],
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
    ) -> list[dict]: ...

    def build_injection_block(self, items: list[dict]) -> tuple[str, list[str]]: ...

    # ── v2: write ─────────────────────────────────────────────────
    async def save_item(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str: ...

    async def save_item_with_supersede(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str: ...

    async def save_from_consolidation(
        self,
        history_entry: str,
        behavior_updates: list[dict],
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
        emotional_weight: int = 0,
    ) -> None: ...

    def reinforce_items_batch(self, ids: list[str]) -> None: ...

    def keyword_match_procedures(self, action_tokens: list[str]) -> list[dict]: ...


# ── Adapter ───────────────────────────────────────────────────────────────────


class DefaultMemoryPort:
    """Adapts MemoryStore (v1) + optional Memorizer/Retriever (v2).

    `memorizer` / `retriever` 可选；未接入时仅提供文件层能力。
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

    def read_profile(self) -> str:
        return self.get_memory_context()

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

    def read_recent_context(self) -> str:
        if hasattr(self._store, "read_recent_context"):
            try:
                return self._store.read_recent_context()
            except Exception:
                return ""
        return ""

    def write_recent_context(self, content: str) -> None:
        if hasattr(self._store, "write_recent_context"):
            self._store.write_recent_context(content)

    # ── v1: pending facts buffer ───────────────────────────────────

    def read_pending(self) -> str:
        return self._store.read_pending()

    def append_pending(self, facts: str) -> None:
        self._store.append_pending(facts)

    def append_pending_once(
        self,
        facts: str,
        source_ref: str,
        kind: str = "pending",
    ) -> bool:
        if hasattr(self._store, "append_pending_once"):
            return self._store.append_pending_once(
                facts,
                source_ref=source_ref,
                kind=kind,
            )
        self._store.append_pending(facts)
        return bool((facts or "").strip())

    def snapshot_pending(self) -> str:
        return self._store.snapshot_pending()

    def commit_pending_snapshot(self) -> None:
        self._store.commit_pending_snapshot()

    def rollback_pending_snapshot(self) -> None:
        self._store.rollback_pending_snapshot()

    # ── v1: history log ────────────────────────────────────────────

    def append_history(self, entry: str) -> None:
        self._store.append_history(entry)

    def append_history_once(
        self,
        entry: str,
        source_ref: str,
        kind: str = "history_entry",
    ) -> bool:
        if hasattr(self._store, "append_history_once"):
            return self._store.append_history_once(
                entry,
                source_ref=source_ref,
                kind=kind,
            )
        self._store.append_history(entry)
        return bool((entry or "").strip())

    def read_history(self, max_chars: int = 0) -> str:
        """Read HISTORY.md; if max_chars > 0, return only the last max_chars."""
        if hasattr(self._store, "read_history"):
            try:
                return self._store.read_history(max_chars=max_chars)
            except Exception:
                return ""
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
        top_k: int | None = None,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
    ) -> list[dict]:
        """Embed query and return top-k memory items; empty list if no retriever."""
        if not self._retriever:
            return []
        try:
            return await self._retriever.retrieve(
                query,
                memory_types=memory_types,
                top_k=top_k,
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
                require_scope_match=require_scope_match,
            )
        except Exception as e:
            logger.warning("[memory_port] retrieve_related failed: %s", e)
            return []

    async def embed_query(self, query: str) -> list[float]:
        """仅 embed 一次，供调用方复用 query_vec 做多路 vector_search。"""
        if not self._retriever:
            return []
        try:
            return await self._retriever.embed(query)
        except Exception as e:
            logger.warning("[memory_port] embed_query failed: %s", e)
            return []

    async def retrieve_related_vec(
        self,
        query_vec: list[float],
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
    ) -> list[dict]:
        """用已有 query_vec 做 vector_search，跳过 embedding 步骤。"""
        if not self._retriever or not query_vec:
            return []
        try:
            return await self._retriever.retrieve_with_vec(
                query_vec,
                memory_types=memory_types,
                top_k=top_k,
                scope_channel=scope_channel,
                scope_chat_id=scope_chat_id,
                require_scope_match=require_scope_match,
            )
        except Exception as e:
            logger.warning("[memory_port] retrieve_related_vec failed: %s", e)
            return []

    def build_injection_block(self, items: list[dict]) -> tuple[str, list[str]]:
        if not self._retriever:
            return "", []
        try:
            return self._retriever.build_injection_block(items)
        except Exception as e:
            logger.warning("[memory_port] build_injection_block failed: %s", e)
            return "", []

    # ── v2: write ──────────────────────────────────────────────────

    async def save_item(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
        emotional_weight: int = 0,
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
                emotional_weight=emotional_weight,
            )
        except Exception as e:
            logger.warning("[memory_port] save_item failed: %s", e)
            return ""

    async def save_item_with_supersede(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        """Embed, supersede conflicting old items, then upsert; returns '' if no memorizer."""
        if not self._memorizer:
            return ""
        try:
            return await self._memorizer.save_item_with_supersede(
                summary=summary,
                memory_type=memory_type,
                extra=extra,
                source_ref=source_ref,
                happened_at=happened_at,
                emotional_weight=emotional_weight,
            )
        except Exception as e:
            logger.warning("[memory_port] save_item_with_supersede failed: %s", e)
            return ""

    async def save_from_consolidation(
        self,
        history_entry: str,
        behavior_updates: list[dict],
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
        emotional_weight: int = 0,
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
                emotional_weight=emotional_weight,
            )
        except Exception as e:
            logger.warning("[memory_port] save_from_consolidation failed: %s", e)

    def reinforce_items_batch(self, ids: list[str]) -> None:
        if not self._memorizer:
            return
        try:
            self._memorizer.reinforce_items_batch(ids)
        except Exception as e:
            logger.warning("[memory_port] reinforce_items_batch failed: %s", e)

    def keyword_match_procedures(self, action_tokens: list[str]) -> list[dict]:
        """对 trigger_tags 做纯关键字匹配，无需向量检索。"""
        if not self._retriever:
            return []
        store = getattr(self._retriever, "_store", None)
        if store is None:
            return []
        try:
            return store.keyword_match_procedures(action_tokens)
        except Exception as e:
            logger.warning("[memory_port] keyword_match_procedures failed: %s", e)
            return []

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

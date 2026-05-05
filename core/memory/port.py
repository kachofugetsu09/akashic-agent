from __future__ import annotations

from typing import Any

from core.memory.engine import MemoryEngine as MemoryPort


class DefaultMemoryPort:
    # TODO(memory-engine-cleanup): 外部测试和插件全部迁到 MemoryEngine 后删除这个兼容壳。
    def __init__(
        self,
        store: Any,
        memorizer: Any | None = None,
        retriever: Any | None = None,
    ) -> None:
        self._store = store
        self._memorizer = memorizer
        self._retriever = retriever

    def read_long_term(self) -> str:
        return self._store.read_long_term()

    def read_profile(self) -> str:
        return self.get_memory_context()

    def write_long_term(self, content: str) -> None:
        self._store.write_long_term(content)

    def read_self(self) -> str:
        return self._store.read_self()

    def write_self(self, content: str) -> None:
        self._store.write_self(content)

    def read_recent_context(self) -> str:
        return self._store.read_recent_context()

    def write_recent_context(self, content: str) -> None:
        self._store.write_recent_context(content)

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
        return self._store.append_pending_once(
            facts,
            source_ref=source_ref,
            kind=kind,
        )

    def snapshot_pending(self) -> str:
        return self._store.snapshot_pending()

    def commit_pending_snapshot(self) -> None:
        self._store.commit_pending_snapshot()

    def rollback_pending_snapshot(self) -> None:
        self._store.rollback_pending_snapshot()

    def append_history(self, entry: str) -> None:
        self._store.append_history(entry)

    def append_history_once(
        self,
        entry: str,
        source_ref: str,
        kind: str = "history_entry",
    ) -> bool:
        return self._store.append_history_once(
            entry,
            source_ref=source_ref,
            kind=kind,
        )

    def read_history(self, max_chars: int = 0) -> str:
        return self._store.read_history(max_chars=max_chars)

    def read_recent_history(self, *, max_chars: int = 0) -> str:
        return self.read_history(max_chars=max_chars)

    def append_journal(
        self,
        date_str: str,
        entry: str,
        *,
        source_ref: str = "",
        kind: str = "journal",
    ) -> bool:
        return self._store.append_journal(
            date_str,
            entry,
            source_ref=source_ref,
            kind=kind,
        )

    def get_memory_context(self) -> str:
        return self._store.get_memory_context()

    def has_long_term_memory(self) -> bool:
        return bool(self.read_long_term().strip())

    async def retrieve_related(self, query: str, **kwargs: Any) -> list[dict]:
        if self._retriever is None:
            return []
        return await self._retriever.retrieve(query, **kwargs)

    def build_injection_block(self, items: list[dict]) -> tuple[str, list[str]]:
        if self._retriever is None:
            return "", []
        return self._retriever.build_injection_block(items)

    def reinforce_items_batch(self, ids: list[str]) -> None:
        if self._memorizer is not None:
            self._memorizer.reinforce_items_batch(ids)

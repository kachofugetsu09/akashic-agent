"""
Memory v2 写入器：将 consolidation 结果保存到 SQLite
"""

from __future__ import annotations

import logging

from memory2.store import MemoryStore2
from memory2.embedder import Embedder

logger = logging.getLogger(__name__)


class Memorizer:
    def __init__(self, store: MemoryStore2, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    async def save_item(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
    ) -> str:
        """embed → content_hash → upsert，返回 'new:id' 或 'reinforced:id'"""
        embedding = await self._embedder.embed(summary)
        return self._store.upsert_item(
            memory_type=memory_type,
            summary=summary,
            embedding=embedding,
            source_ref=source_ref,
            extra=extra,
            happened_at=happened_at,
        )

    async def save_from_consolidation(
        self,
        history_entry: str,
        behavior_updates: list[dict],
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
    ) -> None:
        """将 consolidation 的产出写入 SQLite"""
        # 1. history_entry → event
        if history_entry and history_entry.strip():
            try:
                result = await self.save_item(
                    summary=history_entry.strip(),
                    memory_type="event",
                    extra={
                        "scope_channel": scope_channel,
                        "scope_chat_id": scope_chat_id,
                    },
                    source_ref=source_ref,
                )
                logger.info(f"memory2 event saved: {result}")
            except Exception as e:
                logger.warning(f"memory2 event save 失败: {e}")

        # 2. behavior_updates 统一由 post-response worker 处理，避免与 consolidation 重复写入
        if behavior_updates:
            logger.info(
                "memory2 consolidation skip behavior_updates (%d): handled by post-response worker",
                len(behavior_updates),
            )

    def supersede_batch(self, ids: list[str]) -> None:
        self._store.mark_superseded_batch(ids)
        logger.info(f"memory2 superseded {len(ids)} items: {ids}")

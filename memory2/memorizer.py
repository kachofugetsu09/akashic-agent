"""
Memory v2 写入器：将 consolidation 结果保存到 SQLite
"""

from __future__ import annotations

import logging

from memory2.store import MemoryStore2
from memory2.embedder import Embedder

logger = logging.getLogger(__name__)


def _coerce_emotional_weight(value: object) -> int:
    try:
        return max(0, min(10, int(value or 0)))
    except (TypeError, ValueError):
        return 0


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
        emotional_weight: int = 0,
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
            emotional_weight=emotional_weight,
        )

    async def save_item_with_supersede(
        self,
        summary: str,
        memory_type: str,
        extra: dict,
        source_ref: str,
        happened_at: str | None = None,
        emotional_weight: int = 0,
        merge_threshold: float = 0.70,
        supersede_threshold: float = 0.90,
    ) -> str:
        """先 supersede 高相似旧条目，再写入新条目。

        - procedure / preference：退休相似度 >= supersede_threshold 的旧条目；
          procedure 额外尝试 merge 同工具要求的近似条目。
        - profile（status / purchase 类别）：退休相同 category 中相似度 >= supersede_threshold
          的旧条目，防止同类状态事实堆积。
        """
        embedding = await self._embedder.embed(summary)

        if memory_type in ("procedure", "preference"):
            similar = self._store.vector_search(
                query_vec=embedding,
                top_k=5,
                memory_types=[memory_type],
                score_threshold=min(merge_threshold, supersede_threshold),
            )
            if memory_type == "procedure":
                merge_target = self._pick_explicit_merge_target(similar, extra, merge_threshold)
                if merge_target is not None:
                    merged_summary = self._merge_summary_text(
                        merge_target.get("summary", ""),
                        summary,
                    )
                    await self.merge_item(
                        merge_target["id"],
                        merged_summary,
                        extra_patch=extra,
                    )
                    logger.info(
                        "memorizer save_with_supersede: merged explicit procedure into %s",
                        merge_target["id"],
                    )
                    return f"merged:{merge_target['id']}"
            similar = [
                item
                for item in similar
                if isinstance(score := item.get("score"), int | float)
                and float(score) >= supersede_threshold
            ]
            if similar:
                supersede_ids = [str(item["id"]) for item in similar]
                self._store.mark_superseded_batch(supersede_ids)
                logger.info(
                    "memorizer save_with_supersede: superseded %d %s items: %s",
                    len(supersede_ids), memory_type, supersede_ids,
                )

        elif memory_type == "profile":
            category = str((extra or {}).get("category") or "")
            if category in ("status", "purchase"):
                similar = self._store.vector_search(
                    query_vec=embedding,
                    top_k=5,
                    memory_types=["profile"],
                    score_threshold=supersede_threshold,
                )
                same_cat = [
                    item for item in similar
                    if isinstance(extra_json := item.get("extra_json"), dict)
                    and extra_json.get("category") == category
                    and isinstance(score := item.get("score"), int | float)
                    and float(score)
                    >= (
                        0.92
                        if _coerce_emotional_weight(
                            extra_json.get("_emotional_weight", 0)
                        )
                        >= 7
                        else supersede_threshold
                    )
                ]
                if same_cat:
                    supersede_ids = [str(item["id"]) for item in same_cat]
                    self._store.mark_superseded_batch(supersede_ids)
                    logger.info(
                        "memorizer save_with_supersede: superseded %d profile/%s items: %s",
                        len(supersede_ids), category, supersede_ids,
                    )

        return self._store.upsert_item(
            memory_type=memory_type,
            summary=summary,
            embedding=embedding,
            source_ref=source_ref,
            extra=extra,
            happened_at=happened_at,
            emotional_weight=emotional_weight,
        )

    async def save_from_consolidation(
        self,
        history_entry: str,
        behavior_updates: list[dict],
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
        emotional_weight: int = 0,
    ) -> None:
        """将 consolidation 的产出写入 SQLite"""
        # 1. history_entry → event
        if history_entry and history_entry.strip():
            try:
                text = history_entry.strip()
                if self._store.has_consolidation_source_ref(source_ref):
                    logger.info(
                        "memory2 consolidation skip duplicated source_ref=%s",
                        source_ref,
                    )
                    text = ""
                if text:
                    embedding = await self._embedder.embed(text)
                    if self._should_semantic_dedup_event(
                        embedding,
                        emotional_weight=emotional_weight,
                    ):
                        text = ""
                if text:
                    result = self._store.upsert_consolidation_event(
                        source_ref=source_ref,
                        summary=text,
                        embedding=embedding,
                        extra={
                            "scope_channel": scope_channel,
                            "scope_chat_id": scope_chat_id,
                        },
                        emotional_weight=emotional_weight,
                    )
                    if result.startswith("skipped:"):
                        logger.info(
                            "memory2 consolidation skip duplicated source_ref=%s",
                            source_ref,
                        )
                    else:
                        logger.info(f"memory2 event saved: {result}")
            except Exception as e:
                logger.warning(f"memory2 event save 失败: {e}")

        # 2. behavior_updates 统一由 post-response worker 处理，避免与 consolidation 重复写入
        if behavior_updates:
            logger.info(
                "memory2 consolidation skip behavior_updates (%d): handled by post-response worker",
                len(behavior_updates),
            )

    def _should_semantic_dedup_event(
        self,
        embedding: list[float] | None,
        *,
        emotional_weight: int = 0,
    ) -> bool:
        if embedding is None:
            return False
        similar_ids = self._store.find_similar_recent_events(
            embedding,
            threshold=0.92,
            days_back=7,
        )
        if not similar_ids:
            return False
        self._store.reinforce_items_batch(
            similar_ids[:1],
            emotional_weight=emotional_weight,
        )
        logger.info(
            "memory2 event semantic-dedup: similar=%s",
            similar_ids[:1],
        )
        return True

    def supersede_batch(self, ids: list[str]) -> None:
        self._store.mark_superseded_batch(ids)
        logger.info(f"memory2 superseded {len(ids)} items: {ids}")

    def reinforce_items_batch(self, ids: list[str]) -> None:
        self._store.reinforce_items_batch(ids)

    @staticmethod
    def _merge_summary_text(old_summary: str, new_summary: str) -> str:
        old_summary = (old_summary or "").strip()
        new_summary = (new_summary or "").strip()
        if not old_summary:
            return new_summary
        if not new_summary:
            return old_summary
        if new_summary in old_summary:
            return old_summary
        if old_summary in new_summary:
            return new_summary
        return f"{old_summary.rstrip('。；;，, ')}；{new_summary}"

    @staticmethod
    def _pick_explicit_merge_target(
        similar: list[dict],
        extra: dict,
        merge_threshold: float,
    ) -> dict | None:
        wanted_tool = str(extra.get("tool_requirement") or "").strip()
        if not wanted_tool:
            return None
        for item in similar:
            if float(item.get("score", 0.0)) < merge_threshold:
                continue
            item_extra = item.get("extra_json") or {}
            item_tool = str(item_extra.get("tool_requirement") or "").strip()
            if item_tool == wanted_tool:
                return item
        return None

    async def merge_item(
        self,
        item_id: str,
        merged_summary: str,
        extra_patch: dict | None = None,
    ) -> None:
        """原子更新 merge 目标：summary + content_hash + embedding + extra_json。
        对 procedure 类型同步重建 rule_schema，并写入 _merge_note 供溯源。
        调用方保证 merged_summary 非空且 item_id 存在。
        """
        import json as _json

        from memory2.store import _content_hash

        merged_summary = (merged_summary or "").strip()
        if not merged_summary or not item_id:
            return

        row = self._store._db.execute(
            "SELECT memory_type, extra_json FROM memory_items WHERE id=?", (item_id,)
        ).fetchone()
        if not row:
            logger.warning("merge_item: item %s not found", item_id)
            return

        memory_type, extra_json_str = row
        old_extra: dict = {}
        if extra_json_str:
            try:
                old_extra = _json.loads(extra_json_str) or {}
            except Exception:
                pass

        new_embedding = await self._embedder.embed(merged_summary)
        new_hash = _content_hash(merged_summary, memory_type)

        # 构建更新后的 extra_json
        new_extra = dict(old_extra)
        new_extra["_merge_note"] = merged_summary   # 溯源：记录 merge 时的摘要
        if extra_patch:
            if extra_patch.get("tool_requirement"):
                new_extra["tool_requirement"] = extra_patch.get("tool_requirement")
            if extra_patch.get("steps"):
                merged_steps: list[str] = []
                seen_steps: set[str] = set()
                for step in (old_extra.get("steps") or []) + (extra_patch.get("steps") or []):
                    text = str(step or "").strip()
                    if not text or text in seen_steps:
                        continue
                    seen_steps.add(text)
                    merged_steps.append(text)
                new_extra["steps"] = merged_steps
        if memory_type == "procedure":
            from memory2.rule_schema import build_procedure_rule_schema
            new_extra["rule_schema"] = build_procedure_rule_schema(
                summary=merged_summary,
                tool_requirement=new_extra.get("tool_requirement"),
                steps=new_extra.get("steps") or [],
                rule_schema=old_extra.get("rule_schema"),
            )
            # trigger_tags 依赖 tagger（需要 LLM），merge 时无法在线重建。
            # 保留旧标签比没有标签更危险（可能触发错误的关键词匹配），故直接清除。
            # 下次通过 _save_item_direct 路径写入时，tagger 会重新生成。
            new_extra.pop("trigger_tags", None)

        self._store.merge_item_raw(
            item_id=item_id,
            new_summary=merged_summary,
            new_hash=new_hash,
            new_embedding=new_embedding,
            new_extra=new_extra,
        )
        logger.info("memorizer merge_item id=%s", item_id)

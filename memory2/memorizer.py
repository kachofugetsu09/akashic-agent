"""
Memory v2 写入器：将 consolidation 结果保存到 SQLite
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from memory2.embedder import Embedder
from memory2.rule_schema import (
    parse_procedure_steps,
    parse_procedure_tool_requirement,
    resolve_procedure_rule_schema,
)
from memory2.store import MemoryHit, MemoryStore2, memory_hit_score

logger = logging.getLogger(__name__)

_TIME_PREFIX_RE = re.compile(
    r"^\[(?P<date>\d{4}-\d{2}-\d{2})(?:[ T](?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?)?\]"
)


def _validate_procedure_metadata(
    summary: str,
    memory_type: str,
    extra: Mapping[str, object],
) -> None:
    if memory_type == "procedure":
        _ = resolve_procedure_rule_schema(summary, extra)


def _coerce_emotional_weight(value: object) -> int:
    if value is None or value == "":
        return 0
    if not isinstance(value, str | int | float):
        return 0
    try:
        return max(0, min(10, int(value)))
    except (TypeError, ValueError):
        return 0


def _parse_history_entry_happened_at(summary: str) -> str | None:
    match = _TIME_PREFIX_RE.match(summary.strip())
    if not match:
        return None
    date = match.group("date")
    hour = match.group("hour") or "00"
    minute = match.group("minute") or "00"
    second = match.group("second") or "00"
    return f"{date}T{hour}:{minute}:{second}"


class Memorizer:
    def __init__(self, store: MemoryStore2, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    async def save_item(
        self,
        summary: str,
        memory_type: str,
        extra: dict[str, object],
        source_ref: str,
        happened_at: str | None = None,
        emotional_weight: int = 0,
    ) -> str:
        """embed → content_hash → upsert，返回 'new:id' 或 'reinforced:id'"""
        _validate_procedure_metadata(summary, memory_type, extra)
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
        extra: dict[str, object],
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
        _validate_procedure_metadata(summary, memory_type, extra)
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
                if memory_hit_score(item) >= supersede_threshold
            ]
            if similar:
                supersede_ids = [item["id"] for item in similar]
                self._store.mark_superseded_batch(supersede_ids)
                logger.info(
                    "memorizer save_with_supersede: superseded %d %s items: %s",
                    len(supersede_ids), memory_type, supersede_ids,
                )

        elif memory_type == "profile":
            category = str(extra.get("category") or "")
            if category in ("status", "purchase"):
                similar = self._store.vector_search(
                    query_vec=embedding,
                    top_k=5,
                    memory_types=["profile"],
                    score_threshold=supersede_threshold,
                )
                same_cat: list[MemoryHit] = []
                for item in similar:
                    item_extra = _memory_hit_extra(item)
                    threshold = (
                        0.92
                        if _coerce_emotional_weight(
                            item_extra.get("_emotional_weight", 0)
                        )
                        >= 7
                        else supersede_threshold
                    )
                    if (
                        item_extra.get("category") == category
                        and memory_hit_score(item) >= threshold
                    ):
                        same_cat.append(item)
                if same_cat:
                    supersede_ids = [item["id"] for item in same_cat]
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
        behavior_updates: list[dict[str, object]],
        source_ref: str,
        scope_channel: str,
        scope_chat_id: str,
        emotional_weight: int = 0,
    ) -> None:
        """将 consolidation 的产出写入 SQLite"""
        # 1. history_entry → event
        if history_entry and history_entry.strip():
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
                    happened_at=_parse_history_entry_happened_at(text),
                    emotional_weight=emotional_weight,
                )
                if result.startswith("skipped:"):
                    logger.info(
                        "memory2 consolidation skip duplicated source_ref=%s",
                        source_ref,
                    )
                else:
                    logger.info("memory2 event saved: %s", result)

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
        old_summary = old_summary.strip()
        new_summary = new_summary.strip()
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
        similar: list[MemoryHit],
        extra: dict[str, object],
        merge_threshold: float,
    ) -> MemoryHit | None:
        wanted_tool = parse_procedure_tool_requirement(
            extra.get("tool_requirement")
        )
        if wanted_tool is None or not wanted_tool.strip():
            return None
        for item in similar:
            if memory_hit_score(item) < merge_threshold:
                continue
            item_extra = _memory_hit_extra(item)
            item_tool = parse_procedure_tool_requirement(
                item_extra.get("tool_requirement")
            )
            if item_tool is not None and item_tool.strip() == wanted_tool.strip():
                return item
        return None

    async def merge_item(
        self,
        item_id: str,
        merged_summary: str,
        extra_patch: Mapping[str, object] | None = None,
    ) -> None:
        """合并记忆摘要和元数据，并原子更新持久化结果。"""

        # 1. 校验调用契约并读取当前持久化元数据。
        merged_summary = merged_summary.strip()
        if not merged_summary or not item_id:
            raise ValueError("merge_item 需要非空 item_id 和 merged_summary")

        memory_type, old_extra = self._store.get_item_merge_metadata(item_id)
        new_embedding = await self._embedder.embed(merged_summary)

        # 2. 合并显式更新，并严格解析 procedure 字段。
        new_extra = dict(old_extra)
        new_extra["_merge_note"] = merged_summary
        if extra_patch:
            if "tool_requirement" in extra_patch:
                tool_requirement = parse_procedure_tool_requirement(
                    extra_patch["tool_requirement"]
                )
                if tool_requirement:
                    new_extra["tool_requirement"] = tool_requirement
            if "steps" in extra_patch:
                incoming_steps = parse_procedure_steps(
                    extra_patch["steps"], context="merge extra_patch steps"
                )
                if incoming_steps:
                    existing_steps = (
                        parse_procedure_steps(
                            old_extra["steps"],
                            context=f"memory item {item_id} steps",
                        )
                        if "steps" in old_extra
                        else []
                    )
                    new_extra["steps"] = self._merge_steps(
                        existing_steps, incoming_steps
                    )
        if memory_type == "procedure":
            new_extra["rule_schema"] = resolve_procedure_rule_schema(
                merged_summary,
                new_extra,
            )
            # trigger_tags 依赖 LLM tagger，旧标签与新摘要不再具备一致性。
            _ = new_extra.pop("trigger_tags", None)

        # 3. 将主记录和向量索引作为同一次存储操作提交。
        self._store.merge_item_raw(
            item_id=item_id,
            new_summary=merged_summary,
            new_embedding=new_embedding,
            new_extra=new_extra,
        )
        logger.info("memorizer merge_item id=%s", item_id)

    @staticmethod
    def _merge_steps(existing: list[str], incoming: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for step in [*existing, *incoming]:
            text = step.strip()
            if text in seen:
                continue
            seen.add(text)
            merged.append(text)
        return merged


def _memory_hit_extra(item: MemoryHit) -> dict[str, object]:
    extra = item.get("extra_json")
    if extra is None:
        raise KeyError(f"memory hit {item['id']!r} has no extra_json")
    return extra

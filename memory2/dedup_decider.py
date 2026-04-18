"""
Memory v2 去重决策器：两阶段去重（向量预筛 + LLM 决策）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import json_repair
import numpy as np

from agent.provider import LLMProvider

logger = logging.getLogger(__name__)


class DedupDecision(str, Enum):
    SKIP   = "skip"
    CREATE = "create"
    NONE   = "none"


class MemoryAction(str, Enum):
    MERGE  = "merge"
    DELETE = "delete"


@dataclass
class ExistingAction:
    item_id: str
    summary: str
    action: MemoryAction
    reason: str = ""


@dataclass
class DedupResult:
    decision: DedupDecision
    candidate_summary: str
    candidate_type: str
    similar_items: list[dict]
    actions: list[ExistingAction] = field(default_factory=list)
    reason: str = ""
    query_vector: list[float] | None = None


class DedupDecider:

    def __init__(
        self,
        store,
        embedder,
        provider: LLMProvider,
        model: str,
        similarity_threshold: float = 0.45,   # 库内预筛阈值（宽松，找候选送 LLM）
        batch_dedup_threshold: float = 0.90,  # 批内去重阈值（严格，判同义）
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._provider = provider
        self._model = model
        self._similarity_threshold = similarity_threshold
        self._batch_dedup_threshold = batch_dedup_threshold
        self.MAX_SIMILAR_TO_LLM = 5

    async def decide(
        self,
        candidate: dict,
        *,
        batch_vecs: list[tuple[list[float], dict]] | None = None,
    ) -> DedupResult:
        summary = (candidate.get("summary") or "").strip()
        mtype   = candidate.get("memory_type", "procedure")

        query_vec = await self._embedder.embed(summary)

        similar = self._find_similar(
            query_vec=query_vec,
            memory_type=mtype,
            batch_vecs=batch_vecs,
        )

        if not similar:
            return DedupResult(
                decision=DedupDecision.CREATE,
                candidate_summary=summary,
                candidate_type=mtype,
                similar_items=[],
                actions=[],
                reason="no similar found",
                query_vector=query_vec,
            )

        decision, reason, actions = await self._llm_decide(summary, similar)

        return DedupResult(
            decision=decision,
            candidate_summary=summary,
            candidate_type=mtype,
            similar_items=similar,
            actions=actions,
            reason=reason,
            query_vector=query_vec,
        )

    def _find_similar(
        self,
        query_vec: list[float],
        memory_type: str,
        batch_vecs: list[tuple[list[float], dict]] | None,
    ) -> list[dict]:
        # 库内检索：用 _similarity_threshold（宽松，找候选）
        db_results = self._store.vector_search(
            query_vec=query_vec,
            top_k=self.MAX_SIMILAR_TO_LLM,
            memory_types=[memory_type],
            score_threshold=self._similarity_threshold,
        )
        similar: list[dict] = []
        seen_ids: set[str] = set()
        for item in db_results:
            item = dict(item)
            item["_dedup_score"] = item.get("score", 0.0)
            similar.append(item)
            seen_ids.add(item["id"])

        # 批内检索：用 _batch_dedup_threshold（严格，判同义）
        if batch_vecs:
            q = np.array(query_vec, dtype=np.float32)
            q_norm = float(np.linalg.norm(q)) + 1e-9
            for vec, ctx in batch_vecs:
                cid = ctx.get("id", "")
                if cid in seen_ids:
                    continue
                v = np.array(vec, dtype=np.float32)
                score = float(v @ q) / (float(np.linalg.norm(v)) + 1e-9) / q_norm
                if score >= self._batch_dedup_threshold:
                    ctx_copy = dict(ctx)
                    ctx_copy["_dedup_score"] = score
                    ctx_copy["_batch_internal"] = True
                    similar.append(ctx_copy)
                    seen_ids.add(cid)

        return similar[:self.MAX_SIMILAR_TO_LLM]

    async def _llm_decide(
        self,
        candidate_summary: str,
        similar: list[dict],
    ) -> tuple[DedupDecision, str, list[ExistingAction]]:
        existing_block = "\n".join(
            f"{i+1}. id={item['id']} score={item.get('_dedup_score', 0):.4f}\n"
            f"   summary={item.get('summary', '')}"
            for i, item in enumerate(similar)
        )
        prompt = _DEDUP_DECISION_PROMPT.format(
            candidate_summary=candidate_summary,
            existing_memories=existing_block,
        )
        try:
            resp = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self._model,
                max_tokens=256,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json_repair.loads(text) or {}
            if not isinstance(data, dict):
                return DedupDecision.CREATE, "invalid_llm_payload", []
            return self._parse_payload(data, similar)
        except Exception as e:
            logger.warning("dedup_decider llm failed: %s", e)
            return DedupDecision.CREATE, f"llm failed: {e}", []

    def _parse_payload(
        self,
        data: dict,
        similar: list[dict],
    ) -> tuple[DedupDecision, str, list[ExistingAction]]:
        decision_str = str(data.get("decision", "create")).lower().strip()
        reason = str(data.get("reason", "") or "")

        decision_map = {
            "skip":   DedupDecision.SKIP,
            "create": DedupDecision.CREATE,
            "none":   DedupDecision.NONE,
            "merge":  DedupDecision.NONE,   # legacy: LLM 直接输出 merge 时降级为 none
        }
        decision = decision_map.get(decision_str, DedupDecision.CREATE)

        raw_list = data.get("list", [])
        if not isinstance(raw_list, list):
            raw_list = []

        # legacy 兼容：LLM 输出 decision=merge 但 list 为空时，取 similar[0] 作为 merge 目标
        if decision_str == "merge" and not raw_list and similar:
            raw_list = [{"id": similar[0]["id"], "decide": "merge", "reason": "legacy"}]
            if not reason:
                reason = "legacy merge mapped to none"

        id_to_item = {item["id"]: item for item in similar}
        actions: list[ExistingAction] = []
        seen: dict[str, MemoryAction] = {}

        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            action_str = str(entry.get("decide", "")).lower().strip()
            action = {"merge": MemoryAction.MERGE, "delete": MemoryAction.DELETE}.get(action_str)
            if not action:
                continue

            item = id_to_item.get(entry.get("id", ""))
            if item is None:
                # 兜底：LLM 可能用 1-based index 而非 id
                idx = entry.get("index")
                if isinstance(idx, int):
                    i = (idx - 1) if 1 <= idx <= len(similar) else idx
                    if 0 <= i < len(similar):
                        item = similar[i]
            if item is None:
                continue

            iid = item["id"]
            if iid in seen and seen[iid] != action:
                # 同一 id 出现了冲突动作，丢弃
                actions = [a for a in actions if a.item_id != iid]
                seen.pop(iid)
                logger.warning("dedup: conflicting actions for %s, dropping both", iid)
                continue
            if iid in seen:
                continue

            seen[iid] = action
            actions.append(ExistingAction(
                item_id=iid,
                summary=item.get("summary", ""),
                action=action,
                reason=str(entry.get("reason", "") or ""),
            ))

        # SKIP 不应该带任何动作
        if decision == DedupDecision.SKIP:
            return decision, reason, []

        has_merge = any(a.action == MemoryAction.MERGE for a in actions)

        # create + merge 矛盾：降级为 none
        if decision == DedupDecision.CREATE and has_merge:
            decision = DedupDecision.NONE
            reason = (reason + " | normalized:create+merge->none").strip(" |")

        # create 只能带 delete
        if decision == DedupDecision.CREATE:
            actions = [a for a in actions if a.action == MemoryAction.DELETE]

        # MVP 守卫：merge 目标超过 1 个 → 降级为 SKIP（不丢旧信息，也不写新信息）
        if decision == DedupDecision.NONE:
            merge_actions = [a for a in actions if a.action == MemoryAction.MERGE]
            if len(merge_actions) > 1:
                logger.warning(
                    "dedup: %d merge targets exceeds MVP limit 1, downgrading to SKIP",
                    len(merge_actions),
                )
                return DedupDecision.SKIP, reason + " | multi-merge->skip", []

        return decision, reason, actions


_DEDUP_DECISION_PROMPT = """\
你是记忆去重决策器。根据候选新记忆和已有相似记忆，输出处理决策。

候选新记忆：
{candidate_summary}

已有相似记忆：
{existing_memories}

目标：保持记忆一致、有用，同时避免破坏性编辑。

候选级决策（decision）：
- skip: 候选是重复/同义/信息量不足，不做任何变更
- create: 候选是全新独立记忆，可写入
- none: 候选本身不写入，但需处理旧记忆（通常配合 merge）

旧条目动作（list）：
- merge: 旧条目与候选同主题，需合并/更新（list 中最多 1 条 merge）
- delete: 旧条目已被候选完全替代，整条失效

删除边界（强制规则）：
- 若冲突是局部的（部分内容仍有效），必须用 merge，禁止 delete
- delete 仅用于旧条目整条完全失效的情况
- 主题/方面不同的旧条目，不得出现在 list 里

约束：
- decision 为 skip 时，list 必须为空
- list 中有 merge 时，decision 必须为 none
- decision 为 create 时，list 只能包含 delete
- 若不确定，选保守路径（skip 或 merge），不选 delete

仅返回 JSON，不加说明：
{{
  "decision": "skip|create|none",
  "reason": "简短原因",
  "list": [
    {{"id": "<旧条目id>", "decide": "merge|delete", "reason": "原因"}}
  ]
}}"""

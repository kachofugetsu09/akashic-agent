"""
Memory v2 检索器：查询 → top-k items + 格式化注入块
"""

from __future__ import annotations

import logging

from memory2.store import MemoryStore2
from memory2.embedder import Embedder

logger = logging.getLogger(__name__)


class Retriever:
    INJECT_MAX_CHARS = 1200
    INJECT_MAX_FORCED = 3
    INJECT_MAX_EVENTS = 4
    INJECT_LINE_MAX = 180

    def __init__(
        self,
        store: MemoryStore2,
        embedder: Embedder,
        top_k: int = 8,
        score_threshold: float = 0.45,
        score_thresholds: dict[str, float] | None = None,
        relative_delta: float = 0.06,
        inject_max_chars: int = 1200,
        inject_max_forced: int = 3,
        inject_max_procedure_preference: int = 4,
        inject_max_event_profile: int = 2,
        sop_guard_enabled: bool = True,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        self._score_threshold = score_threshold
        thresholds = score_thresholds or {}
        self._score_thresholds = {
            "procedure": float(thresholds.get("procedure", score_threshold)),
            "preference": float(thresholds.get("preference", score_threshold)),
            "event": float(thresholds.get("event", score_threshold)),
            "profile": float(thresholds.get("profile", score_threshold)),
        }
        self._relative_delta = max(0.0, float(relative_delta))
        self._inject_max_chars = max(200, int(inject_max_chars))
        self._inject_max_forced = max(1, int(inject_max_forced))
        self._inject_max_procedure_preference = max(
            1, int(inject_max_procedure_preference)
        )
        self._inject_max_event_profile = max(0, int(inject_max_event_profile))
        self._sop_guard_enabled = bool(sop_guard_enabled)

    async def retrieve(
        self,
        query: str,
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
    ) -> list[dict]:
        """embed query → cosine search → 返回命中条目列表"""
        query_vec = await self._embedder.embed(query)
        actual_top_k = self._top_k if top_k is None else max(1, int(top_k))
        items = self._store.vector_search(
            query_vec=query_vec,
            top_k=actual_top_k,
            memory_types=memory_types,
            score_threshold=self._score_threshold,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            require_scope_match=require_scope_match,
        )
        logger.debug(f"memory2 retrieve: query={query[:60]!r} hits={len(items)}")
        return items

    @staticmethod
    def _shorten(text: str, max_len: int) -> str:
        text = (text or "").strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "…"

    def format_injection_block(self, items: list[dict]) -> str:
        block, _ = self.format_injection_with_ids(items)
        return block

    def format_injection_with_ids(self, items: list[dict]) -> tuple[str, list[str]]:
        """
        格式化为 system prompt 注入块：
        - procedure with tool_requirement → ## 【强制约束】段
        - procedure without tool_requirement, preference → ## 【流程规范】段
        - event → ## 【相关历史】段
        """
        if not items:
            return "", []

        sorted_items = sorted(
            items,
            key=lambda x: float(x.get("score", 0.0) or 0.0),
            reverse=True,
        )

        forced: list[tuple[str, str]] = []
        norms: list[tuple[str, str]] = []
        events: list[tuple[str, str]] = []

        for item in sorted_items:
            item_id = str(item.get("id", "") or "")
            mtype = item.get("memory_type", "")
            summary = self._shorten(item.get("summary", ""), self.INJECT_LINE_MAX)
            if not summary:
                continue
            extra = item.get("extra_json") or {}
            happened_at = item.get("happened_at") or ""

            if mtype == "procedure":
                tool_req = extra.get("tool_requirement")
                if tool_req:
                    if len(forced) >= self._inject_max_forced:
                        continue
                    line = f"- {summary}（必须调用工具：{tool_req}）"
                    forced.append((item_id, line))
                else:
                    if len(norms) >= self._inject_max_procedure_preference:
                        continue
                    steps = extra.get("steps") or []
                    if steps:
                        step_text = "；".join(str(s) for s in steps)
                        line = f"- {summary}（步骤：{step_text}）"
                    else:
                        line = f"- {summary}"
                    norms.append((item_id, line))
            elif mtype == "preference":
                if len(norms) >= self._inject_max_procedure_preference:
                    continue
                norms.append((item_id, f"- {summary}"))
            elif mtype in ("event", "profile"):
                if len(events) >= min(
                    self.INJECT_MAX_EVENTS, self._inject_max_event_profile
                ):
                    continue
                ts = f"[{happened_at}] " if happened_at else ""
                events.append((item_id, f"- {ts}{summary}"))

        parts: list[tuple[str, list[str]]] = []
        if forced:
            parts.append(
                (
                    "## 【强制约束】记忆规则（必须执行）\n"
                    + "\n".join(line for _, line in forced),
                    [item_id for item_id, _ in forced if item_id],
                )
            )
        if norms:
            parts.append(
                (
                    "## 【流程规范】用户偏好与规则\n"
                    + "\n".join(line for _, line in norms),
                    [item_id for item_id, _ in norms if item_id],
                )
            )
        if events:
            parts.append(
                (
                    "## 【相关历史】你与当前用户的过往对话（来自记忆检索，时间戳可信，可直接引用，不得自行否定）\n"
                    + "\n".join(line for _, line in events),
                    [item_id for item_id, _ in events if item_id],
                )
            )

        if not parts:
            return "", []

        final_parts: list[str] = []
        injected_ids: list[str] = []
        seen_ids: set[str] = set()
        total = 0
        for idx, (part, part_ids) in enumerate(parts):
            add_len = len(part) + (2 if final_parts else 0)
            is_forced_part = idx == 0 and bool(forced)
            if total + add_len > self._inject_max_chars and not is_forced_part:
                continue
            final_parts.append(part)
            total += add_len
            for item_id in part_ids:
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    injected_ids.append(item_id)

        return "\n\n".join(final_parts), injected_ids

    def select_for_injection(self, items: list[dict]) -> list[dict]:
        """按分类型阈值 + 相对阈值过滤检索结果，减少无关上下文注入。"""
        if not items:
            return []

        sorted_items = sorted(
            [i for i in items if isinstance(i, dict)],
            key=lambda x: float(x.get("score", 0.0) or 0.0),
            reverse=True,
        )
        if not sorted_items:
            return []

        type_best: dict[str, float] = {}
        for item in sorted_items:
            mtype = str(item.get("memory_type", "") or "")
            score = float(item.get("score", 0.0) or 0.0)
            if mtype not in type_best or score > type_best[mtype]:
                type_best[mtype] = score

        selected: list[dict] = []
        for item in sorted_items:
            mtype = str(item.get("memory_type", "") or "")
            score = float(item.get("score", 0.0) or 0.0)
            extra = item.get("extra_json") or {}
            if (
                self._sop_guard_enabled
                and mtype == "procedure"
                and extra.get("tool_requirement")
            ):
                selected.append(item)
                continue
            type_th = self._score_thresholds.get(mtype, self._score_threshold)
            floor = type_best.get(mtype, score) - self._relative_delta
            if score < type_th:
                continue
            if score < floor:
                continue
            selected.append(item)

        return selected

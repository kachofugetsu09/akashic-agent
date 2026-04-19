"""
Memory v2 检索器：查询 → top-k items + 格式化注入块
"""

from __future__ import annotations

from datetime import datetime
import json
import logging

from memory2.store import MemoryStore2
from memory2.embedder import Embedder

logger = logging.getLogger(__name__)

_LOW_CONFIDENCE_PHRASES = (
    "未在对话中明确记录",
    "无法凭记忆确认",
    "没有记录",
    "真的没有",
    "未找到",
    "不确定",
)


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
        inject_line_max: int = 180,
        procedure_guard_enabled: bool = True,
        high_inject_delta: float = 0.15,
        hotness_alpha: float = 0.20,
        hotness_half_life_days: float = 14.0,
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
        self._inject_line_max = max(60, int(inject_line_max))
        self._procedure_guard_enabled = bool(procedure_guard_enabled)
        self._high_inject_delta = max(0.0, float(high_inject_delta))
        self._hotness_alpha = max(0.0, min(1.0, float(hotness_alpha)))
        self._hotness_half_life_days = max(1.0, float(hotness_half_life_days))

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
            hotness_alpha=self._hotness_alpha,
            hotness_half_life_days=self._hotness_half_life_days,
        )
        logger.debug(f"memory2 retrieve: query={query[:60]!r} hits={len(items)}")
        return items

    async def embed(self, query: str) -> list[float]:
        """仅做 embedding，不触发 vector_search。"""
        return await self._embedder.embed(query)

    async def retrieve_with_vec(
        self,
        query_vec: list[float],
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        scope_channel: str | None = None,
        scope_chat_id: str | None = None,
        require_scope_match: bool = False,
    ) -> list[dict]:
        """复用已有 query_vec 做本地 vector_search，跳过 embedding 步骤。"""
        actual_top_k = self._top_k if top_k is None else max(1, int(top_k))
        items = self._store.vector_search(
            query_vec=query_vec,
            top_k=actual_top_k,
            memory_types=memory_types,
            score_threshold=self._score_threshold,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            require_scope_match=require_scope_match,
            hotness_alpha=self._hotness_alpha,
            hotness_half_life_days=self._hotness_half_life_days,
        )
        logger.debug(f"memory2 retrieve_with_vec: hits={len(items)}")
        return items

    def build_injection_block(self, items: list[dict]) -> tuple[str, list[str]]:
        """单次流程：筛选条目 → 分段格式化 → 应用字符预算。"""
        selected, forced, norms, events = self._select_injection_sections(items)
        if not selected:
            return "", []

        parts = self._build_section_parts(forced, norms, events)
        return self._apply_char_budget(parts, has_forced=bool(forced))

    def _select_for_injection(self, items: list[dict]) -> list[dict]:
        selected, _forced, _norms, _events = self._select_injection_sections(items)
        return selected

    def _select_injection_sections(
        self,
        items: list[dict],
    ) -> tuple[list[dict], list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
        """1. 筛选条目 2. 按段落准备格式化文本。"""
        if not items:
            return [], [], [], []

        sorted_items = sorted(
            [i for i in items if isinstance(i, dict)],
            key=lambda x: float(x.get("score", 0.0) or 0.0),
            reverse=True,
        )
        if not sorted_items:
            return [], [], [], []

        selected: list[dict] = []
        forced: list[tuple[str, str]] = []
        norms: list[tuple[str, str]] = []
        events: list[tuple[str, str]] = []
        forced_count = 0
        norm_count = 0
        event_count = 0
        for item in sorted_items:
            mtype = str(item.get("memory_type", "") or "")
            score = float(item.get("score", 0.0) or 0.0)
            extra = item.get("extra_json") or {}
            item_id = str(item.get("id", "") or "")
            summary = str(item.get("summary", "") or "").strip()
            happened_at = item.get("happened_at") or ""
            if (
                self._procedure_guard_enabled
                and mtype == "procedure"
                and extra.get("tool_requirement")
            ):
                if forced_count >= self._inject_max_forced:
                    continue
                forced_count += 1
                selected.append(item)
                if summary:
                    tool_req = extra.get("tool_requirement")
                    forced.append((item_id, f"- [{item_id}] {summary}（必须调用工具：{tool_req}）"))
                continue
            type_th = self._score_thresholds.get(mtype, self._score_threshold)
            if score < type_th:
                continue
            if mtype in ("procedure", "preference"):
                if norm_count >= self._inject_max_procedure_preference:
                    continue
                norm_count += 1
            elif mtype in ("event", "profile"):
                if event_count >= self._inject_max_event_profile:
                    continue
                event_count += 1
            else:
                continue
            selected.append(item)
            if not summary:
                continue
            confidence_label = ""
            if score < type_th + self._high_inject_delta:
                confidence_label = "有印象，不确定"
            if mtype == "procedure":
                steps = extra.get("steps") or []
                if steps:
                    step_text = "；".join(str(s) for s in steps)
                    norms.append(
                        (
                            item_id,
                            f"- [{item_id}] {summary}{_format_memory_meta(item, mtype, confidence_label=confidence_label)}（步骤：{step_text}）",
                        )
                    )
                else:
                    norms.append(
                        (
                            item_id,
                            f"- [{item_id}] {summary}{_format_memory_meta(item, mtype, confidence_label=confidence_label)}",
                        )
                    )
            elif mtype == "preference":
                norms.append(
                    (
                        item_id,
                        f"- [{item_id}] {summary}{_format_memory_meta(item, mtype, confidence_label=confidence_label)}",
                    )
                )
            elif mtype in ("event", "profile"):
                ts = f"[{happened_at}] " if happened_at else ""
                events.append(
                    (
                        item_id,
                        f"- [{item_id}] {ts}{summary}{_format_memory_meta(item, mtype, confidence_label=confidence_label)}",
                    )
                )

        return selected, forced, norms, events

    def _build_section_parts(
        self,
        forced: list[tuple[str, str]],
        norms: list[tuple[str, str]],
        events: list[tuple[str, str]],
    ) -> list[tuple[str, list[str]]]:
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
                    "## 【相关历史】你与当前用户的过往对话（来自记忆检索，时间戳可信，可直接引用，不得自行否定；数字/金额/地名等具体值以记录为准，不得用常识替换；可根据上下文合理推断，如去某城市探望姐姐可推断姐姐住在该城市）\n"
                    + "\n".join(line for _, line in events),
                    [item_id for item_id, _ in events if item_id],
                )
            )
        return parts

    def _apply_char_budget(
        self,
        parts: list[tuple[str, list[str]]],
        *,
        has_forced: bool,
    ) -> tuple[str, list[str]]:
        if not parts:
            return "", []

        final_parts: list[str] = []
        injected_ids: list[str] = []
        seen_ids: set[str] = set()
        total = 0
        for idx, (part, part_ids) in enumerate(parts):
            add_len = len(part) + (2 if final_parts else 0)
            is_forced_part = idx == 0 and has_forced
            if total + add_len > self._inject_max_chars and not is_forced_part:
                continue
            final_parts.append(part)
            total += add_len
            for item_id in part_ids:
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    injected_ids.append(item_id)
        return "\n\n".join(final_parts), injected_ids


def _format_source_tag(source_ref: str | None) -> str:
    """从 source_ref（格式如 '["id1","id2"]#h:abc' 或 'channel@seq1-seq2#tag'）中提取消息 ID，
    返回供注入块附加的短标记，如 ' (src: telegram:<chat_id>:<message_id>)'。
    最多显示 2 个 ID，保持注入文本简洁。
    """
    if not source_ref:
        return ""
    raw = source_ref.split("#h:")[0].strip()
    ids: list[str] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            ids = [str(i) for i in parsed if i]
    except (json.JSONDecodeError, ValueError):
        if raw:
            ids = [raw]
    if not ids:
        return ""
    shown = ids[:2]
    tag = ", ".join(shown)
    return f" (src: {tag})"


def _format_memory_meta(
    item: dict,
    memory_type: str,
    *,
    confidence_label: str = "",
) -> str:
    parts: list[str] = []
    if confidence_label:
        parts.append(confidence_label)
    happened_at_raw = item.get("happened_at")
    happened_at = _normalize_happened_at(happened_at_raw)
    if happened_at:
        parts.append(f"发生于: {happened_at}")
        age = _format_relative_age(happened_at_raw)
        if age:
            parts.append(age)
    source_ref = item.get("source_ref")
    src_tag = _format_source_tag(source_ref)
    if src_tag:
        parts.append("证据: 可回源原文")
        parts.append(src_tag.strip())
    else:
        parts.append("证据: 记忆摘要")
    if memory_type == "preference" and _looks_low_confidence_memory(item.get("summary", "")):
        parts.append("低置信线索: 不能单独证明历史细节")
    if not parts:
        return ""
    return "（" + "；".join(parts) + "）"


def _normalize_happened_at(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return text
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and "T" not in text:
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d %H:%M")


def _format_relative_age(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    dt: datetime | None = None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.fromisoformat(f"{text}T00:00:00")
        except ValueError:
            return ""
    now = datetime.now(dt.tzinfo)
    delta = now - dt
    if delta.days >= 1:
        return f"距今约 {delta.days} 天"
    hours = max(0, int(delta.total_seconds() // 3600))
    if hours >= 1:
        return f"距今约 {hours} 小时"
    minutes = max(0, int(delta.total_seconds() // 60))
    return f"距今约 {minutes} 分钟"


def _looks_low_confidence_memory(summary: object) -> bool:
    text = str(summary or "")
    return any(phrase in text for phrase in _LOW_CONFIDENCE_PHRASES)

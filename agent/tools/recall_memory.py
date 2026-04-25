"""
recall_memory 工具：主动检索记忆数据库。

工作流：
  1. 并行生成 2 条 HyDE 假想记忆条目（"用户在X月重构了akashic…"）
  2. 向量路：embed(query) + embed(hypothesis×2) → 三路 vector_search → union
  3. 关键字路：提取 query 关键词 → keyword_search_summary（OR-LIKE，按命中词数排序）
  4. RRF 融合两路排名，取 top-N
  5. 每条结果携带 source_ref，可直接传给 fetch_messages 回溯原始对话
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

from agent.tools.base import Tool

if TYPE_CHECKING:
    from agent.provider import LLMProvider, LLMResponse
    from memory2.embedder import Embedder
    from memory2.store import MemoryStore2

logger = logging.getLogger(__name__)

_RRF_K = 60
_HYPOTHESIS_MAX_TOKENS = 80
_HYPOTHESIS_TIMEOUT_S = 3.0
_EMBED_TIMEOUT_S = 8.0
_VECTOR_SCORE_THRESHOLD = 0.35   # 比自动注入低一档，宁可多召回
_VECTOR_TOP_K = 15
_LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_MemoryHit = dict[str, object]
_ChatCall = Callable[..., Awaitable["LLMResponse"]]
_RECENT_PRESETS = {
    "recent_3d": 3,
    "recent_7d": 7,
    "recent_30d": 30,
}


class RecallMemoryTool(Tool):
    name = "recall_memory"
    description = (
        "检索长期记忆中的提炼事实、偏好、流程与历史事件线索（L1 记忆线索层）。\n"
        "用户问'你还记得吗''以前做过吗''偏好是什么''通常怎么做'时，默认先调用此工具。\n"
        "它返回的是记忆摘要，不是原文证据，不能单独作为回复依据。\n"
        "遇到隐式问题时，先抽象用户真正想找的高层需求，再写 query；不要直接照抄用户原话里的表层词。\n"
        "若输入里带候选项、例子或冗长背景，只保留真正的检索主题，不要把选项措辞整段塞进 query。\n"
        "search_mode 必须显式选择：semantic=按 query 做向量+关键词召回；grep=按 time_filter 列出 event 时间线。\n"
        "用户问'最近三天 DeepSeek 相关事件'这类主题+时间问题，用 semantic + time_filter。\n"
        "用户问'今天我都做了什么''最近三天聊了什么'这类纯时间回顾问题，用 grep + time_filter。\n"
        "用户问'有关重构做过什么'这类无时间主题问题，用 semantic 且不传 time_filter。\n"
        "【使用流程】召回后先评估结果是否足以回答用户问题：\n"
        "  - 相关且有 source_ref → fetch_messages(source_refs) 取原文，基于原文作答\n"
        "  - 结果为空 / 无 source_ref / 与问题不符 / 全是元对话噪声 → 改用 search_messages 关键词补搜，再 fetch\n"
        "禁止跳过此工具直接用 search_messages；禁止只凭摘要作答，不去 fetch 原文。\n"
        "query 写成陈述句效果更好：\n"
        "  ✓ '用户在三月完成了 akashic 运行时架构重构'\n"
        "  ✗ '我们有做过重构吗'\n"
        "  ✓ '用户更喜欢低压力、能长期坚持的创作方式'\n"
        "  ✗ '用户 灵感 焦虑 作品 比赛 点评'\n"
        "【引用协议（必须执行）】只要最终回复使用了本工具返回的任何记忆条目，无论是否继续 fetch 原文，"
        "都必须在正文末尾另起一行输出：\n"
        "  §cited:[id1,id2,...]§\n"
        "  列出本次实际引用的所有条目 id，逗号分隔无空格。未引用任何条目则不输出。\n"
        "工具结果里的 cited_item_ids / citation_required / citation_format 是给你执行这条协议用的，不要忽略。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "检索描述，写成陈述句（HyDE 风格）：先概括用户真正想找的需求或事实，"
                    "再描述你假设存在的记忆内容，不要照抄选项、例子或表层关键词。"
                    "例如：'用户重构了 akashic-agent 的运行时架构'；"
                    "'用户更喜欢低压力、能长期坚持的创作方式'"
                ),
            },
            "memory_type": {
                "type": "string",
                "enum": ["event", "profile", "preference", "procedure", ""],
                "description": "限定记忆类型（留空=全类型）",
                "default": "",
            },
            "search_mode": {
                "type": "string",
                "enum": ["semantic", "grep"],
                "description": (
                    "semantic=按 query 做向量+关键词召回，可叠加 time_filter；"
                    "grep=必须传 time_filter，只按时间范围列出 event，不做主题相关性判断。"
                    "不要用 auto，也不要把纯时间回顾问题交给 semantic。"
                ),
                "default": "semantic",
            },
            "time_filter": {
                "type": "string",
                "description": (
                    "时间过滤：today / yesterday / recent_3d / recent_7d / recent_30d / "
                    "YYYY-MM-DD / YYYY-MM-DD~YYYY-MM-DD。留空=不限时间。"
                ),
                "default": "",
            },
            "limit": {
                "type": "integer",
                "description": "最多返回条数；semantic 最大 20，grep 最大 200",
                "minimum": 1,
                "maximum": 200,
                "default": 8,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        store: "MemoryStore2",
        embedder: "Embedder",
        provider: "LLMProvider",
        model: str,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._provider = provider
        self._model = model

    async def execute(
        self,
        query: str,
        memory_type: str = "",
        search_mode: str = "semantic",
        time_filter: str = "",
        limit: int = 8,
        **_: Any,
    ) -> str:
        query = (query or "").strip()
        if not query:
            return json.dumps({"count": 0, "items": []}, ensure_ascii=False)

        mode = search_mode if search_mode in {"semantic", "grep"} else "semantic"
        max_limit = 200 if mode == "grep" else 20
        limit = max(1, min(int(limit), max_limit))
        time_window = _parse_time_filter(time_filter)
        if time_filter and time_window is None:
            return json.dumps({"count": 0, "items": [], "error": "invalid_time_filter"}, ensure_ascii=False)
        if mode == "grep" and time_window is None:
            return json.dumps({"count": 0, "items": [], "error": "time_filter_required"}, ensure_ascii=False)

        if mode == "grep":
            assert time_window is not None
            start, end = time_window
            items = self._store.list_events_by_time_range(start, end, limit=limit)
            return _build_response(items)

        types = [memory_type] if memory_type else None
        time_start = None
        time_end = None
        if time_window is not None:
            time_start, time_end = time_window

        # 并行：embed(query) + 生成 2 条 HyDE 假想条目
        embed_task = asyncio.create_task(self._embed(query, label="query"))
        hyp1_task = asyncio.create_task(self._gen_hypothesis(query, style="event"))
        hyp2_task = asyncio.create_task(self._gen_hypothesis(query, style="general"))

        query_vec, hyp1, hyp2 = await asyncio.gather(embed_task, hyp1_task, hyp2_task)

        vector_results: list[_MemoryHit] = []
        if query_vec is not None:
            hypotheses = [h for h in (hyp1, hyp2) if h]
            vector_results = await self._vector_union(
                query_vec, hypotheses, types, time_start, time_end
            )

        # 关键字路：从 query 中提取词（中文按字符切分 + 英文按空格）
        terms = _extract_terms(query)
        kw_results = self._store.keyword_search_summary(
            terms,
            memory_types=types,
            limit=30,
            time_start=time_start,
            time_end=time_end,
        )

        # RRF 融合
        merged = _rrf_merge(vector_results, kw_results, top_n=limit)

        logger.info(
            "recall_memory: mode=%s time_filter=%r query=%r vector=%d kw=%d merged=%d hyp=[%r, %r]",
            mode,
            time_filter,
            query[:60],
            len(vector_results),
            len(kw_results),
            len(merged),
            hyp1[:50] if hyp1 else None,
            hyp2[:50] if hyp2 else None,
        )
        return _build_response(merged)

    async def _vector_union(
        self,
        query_vec: list[float],
        hypotheses: list[str],
        types: list[str] | None,
        time_start: datetime | None,
        time_end: datetime | None,
    ) -> list[_MemoryHit]:
        """embed query + hypotheses，各自 vector_search，union dedup。"""
        hyp_vecs: list[list[float]] = []
        if hypotheses:
            hyp_vecs = [
                v
                for v in await asyncio.gather(
                    *[self._embed(h, label="hypothesis") for h in hypotheses]
                )
                if v is not None
            ]

        all_vecs = [query_vec] + list(hyp_vecs)
        seen: dict[str, _MemoryHit] = {}
        hit_groups: list[list[_MemoryHit]] = []
        if time_start is not None or time_end is not None:
            try:
                hit_groups = self._store.vector_search_batch(
                    all_vecs,
                    top_k=_VECTOR_TOP_K,
                    memory_types=types,
                    score_threshold=_VECTOR_SCORE_THRESHOLD,
                    time_start=time_start,
                    time_end=time_end,
                )
            except Exception as e:
                logger.debug("recall_memory: vector_search_batch failed: %s", e)

        if hit_groups:
            for hits in hit_groups:
                for hit in hits:
                    _remember_vector_hit(seen, hit)
            return list(seen.values())

        for vec in all_vecs:
            try:
                hits = self._store.vector_search(
                    vec,
                    top_k=_VECTOR_TOP_K,
                    memory_types=types,
                    score_threshold=_VECTOR_SCORE_THRESHOLD,
                    time_start=time_start,
                    time_end=time_end,
                )
            except Exception as e:
                logger.debug("recall_memory: vector_search failed: %s", e)
                continue
            for hit in hits:
                _remember_vector_hit(seen, hit)
        return list(seen.values())

    async def _embed(self, text: str, *, label: str) -> list[float] | None:
        try:
            return await asyncio.wait_for(
                self._embedder.embed(text),
                timeout=_EMBED_TIMEOUT_S,
            )
        except Exception as e:
            logger.warning(
                "recall_memory: %s embed failed, fallback to keyword only: %s",
                label,
                e,
            )
            return None

    async def _gen_hypothesis(self, query: str, style: str) -> str | None:
        """生成一条假想记忆条目，style=event 生成带时间戳的事件，style=general 生成通用陈述。"""
        if style == "event":
            prompt = (
                "你是个人助手的记忆系统。根据用户提问，生成一条**带具体时间**的假想记忆条目，"
                "格式如 '[2026-03-08] 用户...'\n"
                "规则：第三人称、简洁事实陈述、只输出那一条文本\n\n"
                f"用户提问：{query}\n假想记忆条目："
            )
        else:
            prompt = (
                "你是个人助手的记忆系统。根据用户提问，生成一条假想记忆条目。\n"
                "规则：始终生成肯定式、第三人称（'用户…'）、简洁事实陈述、只输出那一条文本\n\n"
                f"用户提问：{query}\n假想记忆条目："
            )
        try:
            chat = cast(_ChatCall, getattr(self._provider, "chat"))
            resp = await asyncio.wait_for(
                chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    model=self._model,
                    max_tokens=_HYPOTHESIS_MAX_TOKENS,
                ),
                timeout=_HYPOTHESIS_TIMEOUT_S,
            )
            text = (resp.content or "").strip()
            return text if text else None
        except Exception as e:
            logger.debug("recall_memory: hypothesis generation failed (style=%s): %s", style, e)
            return None


def _remember_vector_hit(
    seen: dict[str, _MemoryHit],
    hit: _MemoryHit,
) -> None:
    hit_id = _hit_id(hit)
    hit_score = _hit_score(hit)
    seen_score = _hit_score(seen.get(hit_id, {}))
    if hit_id not in seen or hit_score > seen_score:
        seen[hit_id] = hit


def _hit_id(item: _MemoryHit) -> str:
    return str(item.get("id", "") or "")


def _hit_score(item: _MemoryHit, fallback_key: str = "score") -> float:
    raw = item.get(fallback_key)
    if raw is None and fallback_key != "score":
        raw = item.get("score")
    return float(raw) if isinstance(raw, int | float) else 0.0


def _now_local() -> datetime:
    return datetime.now(_LOCAL_TZ)


def _parse_day(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=_LOCAL_TZ)
    except ValueError:
        return None


def _parse_time_filter(value: str) -> tuple[datetime, datetime] | None:
    text = (value or "").strip()
    if not text:
        return None

    now = _now_local()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if text == "today":
        return today, today + timedelta(days=1)
    if text == "yesterday":
        start = today - timedelta(days=1)
        return start, today
    if text in _RECENT_PRESETS:
        return now - timedelta(days=_RECENT_PRESETS[text]), now

    if "~" in text:
        left, right = [part.strip() for part in text.split("~", 1)]
        start = _parse_day(left)
        end_day = _parse_day(right)
        if start is None or end_day is None:
            return None
        return start, end_day + timedelta(days=1)

    day = _parse_day(text)
    if day is None:
        return None
    return day, day + timedelta(days=1)


def _build_response(raw_items: list[_MemoryHit]) -> str:
    items: list[_MemoryHit] = []
    for item in raw_items:
        entry: _MemoryHit = {
            "id": item["id"],
            "memory_type": item["memory_type"],
            "summary": item["summary"],
            "happened_at": item.get("happened_at") or "",
            "score": round(_hit_score(item, fallback_key="rrf_score"), 4),
        }
        if item.get("source_ref"):
            entry["source_ref"] = item["source_ref"]
        items.append(entry)
    cited_item_ids = [str(item["id"]) for item in items if str(item.get("id", "")).strip()]
    return json.dumps(
        {
            "count": len(items),
            "items": items,
            "citation_required": True,
            "citation_format": "§cited:[id1,id2,...]§",
            "cited_item_ids": cited_item_ids,
            "citation_rule": (
                "若最终回复使用了本工具返回的任何记忆条目，"
                "必须在正文末尾输出 §cited:[实际使用的id列表]§"
            ),
        },
        ensure_ascii=False,
    )


_CJK_STOPWORDS = {
    "用户", "助手", "我们", "他们", "这个", "那个", "什么", "如何", "是否",
    "有没", "没有", "有过", "做过", "进行", "完成", "包括", "通过", "实现",
    "行为", "内容", "相关", "情况", "问题", "方式", "时候", "时间", "目前",
    "当前", "最近", "之前", "以前", "后来", "然后", "因为", "所以", "但是",
    "用户在", "用户对", "的行为吗", "进行了",
}


def _extract_terms(query: str) -> list[str]:
    """从 query 中提取关键词。

    策略：
    - ASCII/数字 token（>= 2 chars）直接保留（Phase、akashic、1-4 等）
    - CJK 块 <= 4 chars：整体保留（重构、架构、运行时、工具链 这类复合词）
    - CJK 块 > 4 chars：拆成 2-gram（避免过长短语无法匹配）
    - 去掉高频虚词
    """
    terms: list[str] = []
    ascii_tokens = re.findall(r"[a-zA-Z0-9_\-\.]{2,}", query)
    terms.extend(ascii_tokens)

    cjk_chunks = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]{2,}", query)
    for chunk in cjk_chunks:
        if len(chunk) <= 4:
            if chunk not in _CJK_STOPWORDS:
                terms.append(chunk)
        else:
            # 长块拆 2-gram，保留有意义的片段
            for i in range(len(chunk) - 1):
                bigram = chunk[i:i + 2]
                if bigram not in _CJK_STOPWORDS:
                    terms.append(bigram)

    seen: set[str] = set()
    result: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result[:20]


def _rrf_merge(
    vector_items: list[_MemoryHit],
    kw_items: list[_MemoryHit],
    *,
    top_n: int,
    k: int = _RRF_K,
) -> list[_MemoryHit]:
    """Reciprocal Rank Fusion：合并向量排名和关键字排名。"""
    # 构建各路排名 {id: rank}（rank 从 1 开始）
    vec_rank: dict[str, int] = {}
    for i, item in enumerate(sorted(vector_items, key=_hit_score, reverse=True)):
        item_id = _hit_id(item)
        if item_id:
            vec_rank[item_id] = i + 1
    kw_rank: dict[str, int] = {}
    for i, item in enumerate(kw_items):
        item_id = _hit_id(item)
        if item_id:
            kw_rank[item_id] = i + 1

    # 合并所有 id
    all_ids: set[str] = set(vec_rank) | set(kw_rank)
    # 建 id → item 映射（向量结果优先，因为包含完整字段）
    id_to_item: dict[str, _MemoryHit] = {}
    for item in kw_items:
        item_id = _hit_id(item)
        if item_id:
            id_to_item[item_id] = item
    for item in vector_items:
        item_id = _hit_id(item)
        if item_id:
            id_to_item[item_id] = item  # 覆盖 kw，vector 结果字段更全

    scored: list[tuple[str, float]] = []
    for item_id in all_ids:
        rrf = 0.0
        if item_id in vec_rank:
            rrf += 1.0 / (k + vec_rank[item_id])
        if item_id in kw_rank:
            rrf += 1.0 / (k + kw_rank[item_id])
        scored.append((item_id, rrf))

    scored.sort(key=lambda x: x[1], reverse=True)
    result: list[_MemoryHit] = []
    for item_id, rrf_score in scored[:top_n]:
        item = dict(id_to_item[item_id])
        item["rrf_score"] = rrf_score
        result.append(item)
    return result

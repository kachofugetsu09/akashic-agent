"""
recall_memory 工具：主动检索记忆数据库。

工具层只负责参数校验和引用协议输出；检索策略走 memory facade，
真正查库融合在 memory2.Retriever。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

from agent.tools.base import Tool
from core.memory.runtime_facade import ExplicitRetrievalRequest

if TYPE_CHECKING:
    from core.memory.runtime_facade import MemoryRuntimeFacade

_LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_MemoryHit = dict[str, object]
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
        facade: "MemoryRuntimeFacade",
    ) -> None:
        self._facade = facade

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
        else:
            start = time_window[0] if time_window is not None else None
            end = time_window[1] if time_window is not None else None

        result = await self._facade.retrieve_explicit(
            ExplicitRetrievalRequest(
                query=query,
                memory_type=memory_type,
                search_mode=mode,
                limit=limit,
                time_start=start,
                time_end=end,
            )
        )
        return _build_response(cast(list[_MemoryHit], result.hits))


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

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

from agent.tools.base import Tool
from core.memory.engine import (
    EvidenceRef,
    MemoryQuery,
    MemoryQueryFilters,
    MemoryQueryIntent,
    MemoryRecord,
    MemoryScope,
    MemoryToolSpec,
)

if TYPE_CHECKING:
    from core.memory.engine import MemoryRetrievalApi

_LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_RECENT_PRESETS = {
    "recent_3d": 3,
    "recent_7d": 7,
    "recent_30d": 30,
}

class RecallMemoryTool(Tool):
    name = "recall_memory"
    description = "由当前 memory engine 的 tool_profile 注入工具描述。"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(
        self,
        memory: "MemoryRetrievalApi",
        spec: MemoryToolSpec,
    ) -> None:
        self._memory = memory
        self._spec = spec
        self.description = self._spec.description
        self.parameters = self._spec.parameters

    async def execute(
        self,
        query: str,
        intent: str = "answer",
        memory_kind: str = "",
        time_filter: str = "",
        limit: int = 8,
        channel: str | None = None,
        chat_id: str | None = None,
        **extra: Any,
    ) -> str:
        text = (query or "").strip()
        if not text:
            return _render_records([], trace={})
        time_window = _parse_time_filter(time_filter)
        if time_filter and time_window is None:
            return json.dumps({"count": 0, "items": [], "error": "invalid_time_filter"}, ensure_ascii=False)
        result = await self._memory.query(
            MemoryQuery(
                text=text,
                intent=_normalize_intent(intent),
                scope=MemoryScope(
                    session_key=f"{channel}:{chat_id}" if channel and chat_id else "",
                    channel=channel or "",
                    chat_id=chat_id or "",
                ),
                filters=MemoryQueryFilters(
                    kinds=_memory_kinds(memory_kind),
                    time_start=time_window[0] if time_window else None,
                    time_end=time_window[1] if time_window else None,
                ),
                limit=max(1, min(int(limit), 200)),
                context=dict(extra),
                timestamp=_parse_current_timestamp(extra.get("current_timestamp")),
            )
        )
        return _render_records(result.records, trace=result.trace)


def _render_records(records: list[MemoryRecord], *, trace: dict[str, object]) -> str:
    items: list[dict[str, object]] = []
    for record in records:
        evidence = _render_evidence(record.evidence)
        source_ref = _first_source_ref(evidence)
        item: dict[str, object] = {
            "id": record.id,
            "memory_type": record.kind,
            "summary": record.summary,
            "score": round(record.score, 4),
            "evidence": evidence or [],
            "signals": record.signals,
        }
        if source_ref:
            item["source_ref"] = source_ref
        items.append(item)
    cited_item_ids = [str(item["id"]) for item in items if str(item.get("id", "")).strip()]
    return json.dumps(
        {
            "count": len(items),
            "items": items,
            "trace": trace,
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


def _render_evidence(evidence: list[EvidenceRef]) -> list[dict[str, object]]:
    return [
        {
            "kind": item.kind,
            "refs": item.refs,
            "resolver": item.resolver,
            "source_ref": item.source_ref,
            "metadata": item.metadata,
        }
        for item in evidence
    ]


def _first_source_ref(evidence: list[dict[str, object]]) -> str:
    for item in evidence:
        source_ref = str(item.get("source_ref") or "").strip()
        if source_ref:
            return source_ref
        refs = item.get("refs")
        if isinstance(refs, list):
            for ref in cast(list[object], refs):
                if not isinstance(ref, str):
                    continue
                text = ref.strip()
                if text:
                    return text
    return ""


def _normalize_intent(
    value: str,
) -> MemoryQueryIntent:
    intents: dict[str, MemoryQueryIntent] = {
        "context": "context",
        "answer": "answer",
        "timeline": "timeline",
        "interest": "interest",
        "procedure": "procedure",
    }
    return intents.get(value, "answer")


def _memory_kinds(memory_kind: str) -> tuple[str, ...]:
    value = memory_kind.strip()
    return (value,) if value else ()


def _parse_current_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return datetime.fromisoformat(value)


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

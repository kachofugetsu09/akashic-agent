"""原始会话消息查询工具。"""

from __future__ import annotations

import json
from typing import Any

from agent.tools.base import Tool
from session.store import SessionStore

_MAX_CONTEXT = 10
_MAX_PREVIEW_LINES = 50


class FetchMessagesTool(Tool):
    name = "fetch_messages"
    description = (
        "根据消息 ID 或 source_ref 读取原始历史消息原文与上下文。\n"
        "这是 recall_memory / search_messages / 记忆注入三条路里唯一可以直接作为最终证据的工具。\n"
        "何时必须调用：回答依赖具体时间、原话、金额、配置值、是否发生过——只要结论需要事实支撑，就在回复前调用此工具。\n"
        "recall_memory 或 search_messages 拿到 source_ref 后，若答案依赖原文细节，直接用 fetch_messages(source_ref) 取证，不要猜。\n"
        "支持 context 参数扩展前后文，适合还原完整上下文片段。\n"
        "【引用协议（必须执行）】本工具调用后，最终回复正文末尾必须另起一行输出：\n"
        "  §cited:[memory_id1,memory_id2,...]§\n"
        "  列出本次实际引用的记忆条目 id（来自 recall_memory 结果中的 id 字段），逗号分隔无空格。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "消息 ID 列表，格式如 'telegram:7674283004:495'",
            },
            "source_ref": {
                "type": "string",
                "description": "单个 source_ref，可传 message id 或记忆条目的 source_ref",
            },
            "source_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "多个 source_ref，可混合传 message id 与记忆条目的 source_ref",
            },
            "context": {
                "type": "integer",
                "description": "每条消息前后各扩展的上下文条数（0=仅精确匹配，最大 10，默认 0）",
                "minimum": 0,
                "maximum": _MAX_CONTEXT,
                "default": 0,
            },
        },
    }

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    async def execute(
        self,
        ids: list[str] | None = None,
        source_ref: str | None = None,
        source_refs: list[str] | None = None,
        context: int = 0,
        **_: Any,
    ) -> str:
        clean_ids = _resolve_fetch_ids(
            ids=ids or [],
            source_ref=source_ref,
            source_refs=source_refs or [],
        )
        if not clean_ids:
            return json.dumps({"count": 0, "matched_count": 0, "messages": []}, ensure_ascii=False)

        ctx = max(0, min(int(context), _MAX_CONTEXT))
        if ctx == 0:
            messages = self._store.fetch_by_ids(clean_ids)
            return json.dumps(
                {"count": len(messages), "matched_count": len(messages), "messages": messages},
                ensure_ascii=False,
            )

        messages = self._store.fetch_by_ids_with_context(clean_ids, ctx)
        matched = sum(1 for m in messages if m.get("in_source_ref"))
        return json.dumps(
            {"count": len(messages), "matched_count": matched, "messages": messages},
            ensure_ascii=False,
        )


def _resolve_fetch_ids(
    *,
    ids: list[str],
    source_ref: str | None,
    source_refs: list[str],
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for value in list(ids) + ([source_ref] if source_ref else []) + list(source_refs):
        for item_id in _expand_source_ref(value):
            if item_id not in seen:
                seen.add(item_id)
                resolved.append(item_id)
    return resolved


def _expand_source_ref(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    prefix = raw.split("#", 1)[0].strip()
    if not prefix:
        return []
    try:
        parsed = json.loads(prefix)
    except (json.JSONDecodeError, ValueError):
        return [prefix]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


class SearchMessagesTool(Tool):
    name = "search_messages"
    description = (
        "对原始历史消息做 grep 式搜索，返回命中候选消息的预览和 source_ref。\n"
        "适合查找某个词、句子、文件名、报错、命令、配置项曾出现在哪些消息里——它是文本定位工具。\n"
        "不是记忆检索工具：不负责总结偏好、判断做没做过、回答历史事实。这些问题先用 recall_memory。\n"
        "命中后若需确认上下文或以结果作为证据，必须继续 fetch_messages(source_ref)，预览不能直接作证。\n"
        "recall_memory 返回的摘要读起来像[询问行为]而非[事件本身]时，可同步用此工具补一路 grep 交叉验证。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词或短语"},
            "session_key": {
                "type": "string",
                "description": "限定 session，如 'telegram:7674283004'（可选）",
            },
            "role": {
                "type": "string",
                "enum": ["user", "assistant"],
                "description": "限定发言方（可选）",
            },
            "limit": {
                "type": "integer",
                "description": "最多返回条数，默认 10，最大 50",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
            },
            "offset": {
                "type": "integer",
                "description": "分页偏移量，默认 0；下一页可用返回里的 next_offset",
                "minimum": 0,
                "default": 0,
            },
        },
        "required": ["query"],
    }

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    async def execute(self, query: str, **kwargs: Any) -> str:
        term = (query or "").strip()
        if not term:
            return json.dumps(
                {
                    "count": 0,
                    "matched_count": 0,
                    "limit": 10,
                    "offset": 0,
                    "has_more": False,
                    "next_offset": None,
                    "messages": [],
                },
                ensure_ascii=False,
            )

        limit = max(1, min(int(kwargs.get("limit", 10)), 50))
        offset = max(0, int(kwargs.get("offset", 0)))

        matched, total = self._store.search_messages(
            term,
            session_key=(kwargs.get("session_key") or "").strip() or None,
            role=(kwargs.get("role") or "").strip() or None,
            limit=limit,
            offset=offset,
        )
        terms = [t for t in term.split() if t]
        messages = [_build_search_preview(message, terms) for message in matched]
        next_offset = offset + len(messages)
        has_more = next_offset < total
        if not has_more:
            next_offset = None
        return json.dumps(
            {
                "count": len(messages),
                "matched_count": total,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "next_offset": next_offset,
                "messages": messages,
            },
            ensure_ascii=False,
        )


def _build_search_preview(message: dict[str, Any], query_terms: list[str] | None = None) -> dict[str, Any]:
    content = str(message.get("content", "") or "")
    preview, line_count, truncated = _preview_lines(content, max_lines=_MAX_PREVIEW_LINES)
    matched_terms = (
        [t for t in query_terms if t.lower() in content.lower()]
        if query_terms else []
    )
    result = {
        "id": str(message.get("id", "") or ""),
        "source_ref": str(message.get("id", "") or ""),
        "session_key": str(message.get("session_key", "") or ""),
        "seq": int(message.get("seq", 0) or 0),
        "role": str(message.get("role", "") or ""),
        "timestamp": str(message.get("timestamp", "") or ""),
        "matched_terms": matched_terms,
        "preview": preview,
        "preview_line_count": min(line_count, _MAX_PREVIEW_LINES),
        "total_line_count": line_count,
        "truncated": truncated,
    }
    return result


def _preview_lines(content: str, *, max_lines: int) -> tuple[str, int, bool]:
    lines = content.splitlines()
    if not lines:
        return content[:0], 0, False
    selected = lines[:max_lines]
    truncated = len(lines) > max_lines
    preview = "\n".join(selected)
    if truncated:
        preview += f"\n...[已截断，剩余 {len(lines) - max_lines} 行]"
    return preview, len(lines), truncated

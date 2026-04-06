"""原始会话消息查询工具。"""

from __future__ import annotations

import json
from typing import Any

from agent.tools.base import Tool
from session.store import SessionStore

_MAX_CONTEXT = 5
_MAX_PREVIEW_LINES = 50


class FetchMessagesTool(Tool):
    name = "fetch_messages"
    description = (
        "按消息 ID 精确拉取原始对话内容。"
        "当 search_messages 返回了 source_ref，或记忆注入块中的条目附带 (src: ...) 标记时，必须优先用此工具获取原文，而非继续猜测。"
        "只要你准备基于某条历史消息下结论、引用细节、回答时间线、金额、是否发生过，就先调用 fetch_messages 拉原文再答。"
        "支持 context 参数扩展前后文，适合还原完整上下文片段。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "消息 ID 列表，格式如 'telegram:7674283004:495'",
            },
            "context": {
                "type": "integer",
                "description": "每条消息前后各扩展的上下文条数（0=仅精确匹配，最大 5，默认 0）",
                "minimum": 0,
                "maximum": _MAX_CONTEXT,
                "default": 0,
            },
        },
        "required": ["ids"],
    }

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    async def execute(self, ids: list[str], context: int = 0, **_: Any) -> str:
        clean_ids = [str(i).strip() for i in (ids or []) if str(i).strip()]
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


class SearchMessagesTool(Tool):
    name = "search_messages"
    description = (
        "按关键词搜索最相关的历史消息预览。"
        "它只返回分页后的消息摘要，不返回完整原文。"
        "每条结果都带 source_ref；只要搜索命中了候选消息，下一步就必须用 fetch_messages(source_ref) 回源查看原文。"
        "不要直接把 search_messages 的预览当成完整证据，更不要只靠预览就回答历史事实、时间、金额、是否发生过。"
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
        messages = [_build_search_preview(message) for message in matched]
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


def _build_search_preview(message: dict[str, Any]) -> dict[str, Any]:
    content = str(message.get("content", "") or "")
    preview, line_count, truncated = _preview_lines(content, max_lines=_MAX_PREVIEW_LINES)
    return {
        "id": str(message.get("id", "") or ""),
        "source_ref": str(message.get("id", "") or ""),
        "session_key": str(message.get("session_key", "") or ""),
        "seq": int(message.get("seq", 0) or 0),
        "role": str(message.get("role", "") or ""),
        "timestamp": str(message.get("timestamp", "") or ""),
        "preview": preview,
        "preview_line_count": min(line_count, _MAX_PREVIEW_LINES),
        "total_line_count": line_count,
        "truncated": truncated,
    }


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

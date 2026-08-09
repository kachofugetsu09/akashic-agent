from __future__ import annotations

import json
from typing import Any

from agent.model_runtime.tool_result_projection import (
    TOOL_RESULT_READ_DEFAULT_CHARS,
    TOOL_RESULT_READ_MAX_CHARS,
)
from agent.tools.base import Tool, get_current_tool_context
from session.store import SessionStore


class ReadToolResultTool(Tool):
    name = "read_tool_result"
    description = (
        "当历史工具结果显示 tool_result_ref 占位符时，"
        "用该 artifact_id 分页读取当前 session 的完整原文。"
        "只在当前任务确实需要旧结果细节时调用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "tool_result_ref 中的 artifact_id。",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "从原文的 Unicode 字符偏移开始读取。",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": TOOL_RESULT_READ_MAX_CHARS,
                "default": TOOL_RESULT_READ_DEFAULT_CHARS,
                "description": "本次最多返回的字符数。",
            },
        },
        "required": ["artifact_id"],
        "additionalProperties": False,
    }

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    async def execute(
        self,
        artifact_id: str,
        offset: int = 0,
        limit: int = TOOL_RESULT_READ_DEFAULT_CHARS,
        **_: Any,
    ) -> str:
        """Read one bounded artifact slice and durably record the read."""

        context = get_current_tool_context()
        if context is None or not context.origin_session_key or not context.turn_id:
            raise RuntimeError("read_tool_result 缺少 session/turn 执行身份")
        result = self._store.read_tool_result(
            session_key=context.origin_session_key,
            reader_turn_id=context.turn_id,
            artifact_id=artifact_id,
            offset=offset,
            limit=limit,
        )
        return json.dumps(result.to_payload(), ensure_ascii=False, separators=(",", ":"))

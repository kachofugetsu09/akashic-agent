"""
memorize 工具：用户主动写记忆
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from agent.tools.base import Tool
from core.memory.engine import MemoryScope, RememberRequest

if TYPE_CHECKING:
    from core.memory.engine import MemoryEngine

logger = logging.getLogger(__name__)


def _format_remember_result_text(item_id: str, write_status: str, summary: str) -> str:
    value = (item_id or "").strip()
    status = (write_status or "new").strip()
    return f"已记住（item_id={value}；status={status}）：{summary}"
class MemorizeTool(Tool):
    name = "memorize"
    description = (
        "将重要规则/流程/偏好永久写入记忆。\n"
        "仅在用户明确表达意图时调用（如：记住、以后、下次、你要）。\n"
        "若这条记忆来自你刚核实过的原始对话，可传 source_ref/source_refs 保留回源证据。\n"
        "禁止存储：第三方行为描述、用户个人印象、知识分享内容、已存储的偏好重复记录。\n"
        "【勿记录】：时效性事件（发布日期/赛季/已过期日程节点）、"
        "系统连接状态（管道/Token/服务可用性）、"
        "生理指标具体数值或推断（心率/血氧基线等，应通过 fitbit_health_snapshot 实时查询）、"
        "针对单次任务的专项操作规范。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "一句话描述要记住的内容",
            },
            "memory_type": {
                "type": "string",
                "enum": ["procedure", "preference", "event", "profile"],
                "description": "记忆类型",
            },
            "tool_requirement": {
                "type": "string",
                "description": "该规则要求必须调用的工具名（可选）",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "执行步骤（可选）",
            },
            "source_ref": {
                "type": "string",
                "description": "单个证据 source_ref；若已回看过原始对话，可传对应 source_ref 或 message id",
            },
            "source_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "多个证据 source_ref / message id；会合并保存，便于后续回源",
            },
        },
        "required": ["summary", "memory_type"],
    }

    def __init__(self, engine: "MemoryEngine") -> None:
        self._engine = engine

    async def execute(
        self,
        summary: str,
        memory_type: str,
        tool_requirement: str | None = None,
        steps: list[str] | None = None,
        source_ref: str | None = None,
        source_refs: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        **_: Any,
    ) -> str:
        resolved_source_ref = _resolve_memory_source_ref(
            source_ref=source_ref,
            source_refs=source_refs or [],
        )
        result = await self._engine.remember(
            RememberRequest(
                summary=summary,
                memory_type=memory_type,
                scope=MemoryScope(
                    session_key=f"{channel}:{chat_id}" if channel and chat_id else "",
                    channel=channel or "",
                    chat_id=chat_id or "",
                ),
                source_ref=resolved_source_ref,
                raw_extra={
                    "tool_requirement": tool_requirement,
                    "steps": steps or [],
                },
            )
        )
        logger.info("memorize: engine stored memory_type=%s", result.actual_type)
        return _format_remember_result_text(result.item_id, result.write_status, summary)


def _resolve_memory_source_ref(
    *,
    source_ref: str | None,
    source_refs: list[str],
) -> str:
    refs: list[str] = []
    seen: set[str] = set()
    for raw in ([source_ref] if source_ref else []) + list(source_refs):
        for value in _expand_source_ref(raw):
            if value not in seen:
                seen.add(value)
                refs.append(value)
    if not refs:
        return "memorize_tool"
    if len(refs) == 1:
        return refs[0]
    return json.dumps(refs, ensure_ascii=False)


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
        return [raw]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return [raw]

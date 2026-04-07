"""
memorize 工具：用户主动写记忆
"""

from __future__ import annotations

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
        "来源会自动绑定到当前用户这条消息，无需也不要手动传 source_ref。\n"
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
        current_user_source_ref: str | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        **_: Any,
    ) -> str:
        result = await self._engine.remember(
            RememberRequest(
                summary=summary,
                memory_type=memory_type,
                scope=MemoryScope(
                    session_key=f"{channel}:{chat_id}" if channel and chat_id else "",
                    channel=channel or "",
                    chat_id=chat_id or "",
                ),
                source_ref=str(current_user_source_ref or "").strip(),
                raw_extra={
                    "tool_requirement": tool_requirement,
                    "steps": steps or [],
                },
            )
        )
        logger.info("memorize: engine stored memory_type=%s", result.actual_type)
        return _format_remember_result_text(result.item_id, result.write_status, summary)

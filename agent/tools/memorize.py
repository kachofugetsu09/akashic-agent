"""
memorize 工具：用户主动写记忆
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from agent.tools.base import Tool
from core.memory.engine import MemoryScope, RememberRequest
from memory2.rule_schema import build_procedure_rule_schema

if TYPE_CHECKING:
    from core.memory.engine import PassiveMemoryEngine
    from core.memory.port import MemoryPort

logger = logging.getLogger(__name__)


def _format_remember_result_text(item_id: str, summary: str) -> str:
    value = (item_id or "").strip()
    if value.startswith(("new:", "reinforced:", "merged:")):
        return f"已记住（{value}）：{summary}"
    return f"已记住（item_id={value}）：{summary}"


def _coerce_memory_type(
    memory_type: str,
    tool_requirement: str | None,
    steps: list[str] | None,
) -> str:
    """procedure 若无 tool_requirement 也无 steps，极大概率是偏好被误分类，纠正为 preference。"""
    if memory_type != "procedure":
        return memory_type
    has_tool = bool(tool_requirement and tool_requirement.strip())
    has_steps = bool(steps and any(s.strip() for s in steps))
    if not has_tool and not has_steps:
        return "preference"
    return memory_type


class MemorizeTool(Tool):
    name = "memorize"
    description = (
        "将重要规则/流程/偏好永久写入记忆。\n"
        "仅在用户明确表达意图时调用（如：记住、以后、下次、你要）。\n"
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

    def __init__(self, memory: "MemoryPort", tagger=None) -> None:
        self._memory = memory
        self._tagger = tagger  # ProcedureTagger | None
        self._passive_engine: "PassiveMemoryEngine | None" = None

    def bind_passive_engine(self, engine: "PassiveMemoryEngine | None") -> None:
        self._passive_engine = engine

    async def execute(
        self,
        summary: str,
        memory_type: str,
        tool_requirement: str | None = None,
        steps: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        **_: Any,
    ) -> str:
        if self._passive_engine is not None:
            result = await self._passive_engine.remember(
                RememberRequest(
                    summary=summary,
                    memory_type=memory_type,
                    scope=MemoryScope(
                        session_key=f"{channel}:{chat_id}" if channel and chat_id else "",
                        channel=channel or "",
                        chat_id=chat_id or "",
                    ),
                    raw_extra={
                        "tool_requirement": tool_requirement,
                        "steps": steps or [],
                    },
                )
            )
            return _format_remember_result_text(result.item_id, summary)

        # 0. 类型纠正：无 tool_requirement 且无 steps 的 procedure → preference。
        memory_type = _coerce_memory_type(memory_type, tool_requirement, steps)
        # 1. 构造 extra，保留 tool_requirement / steps 字段。
        extra: dict = {
            "tool_requirement": tool_requirement,
            "steps": steps or [],
        }
        # 2. procedure 规则额外写入结构化 rule_schema，供后续冲突检测与 supersede 使用。
        if memory_type == "procedure":
            extra["rule_schema"] = build_procedure_rule_schema(
                summary=summary,
                tool_requirement=tool_requirement,
                steps=steps or [],
            )
        # 3. 补 tagger 产物并写入 memory port。
        if memory_type == "procedure" and self._tagger is not None:
            try:
                trigger_tags = await self._tagger.tag(summary)
                if trigger_tags is not None:
                    extra["trigger_tags"] = trigger_tags
                    logger.info(
                        "memorize: trigger_tags generated scope=%s",
                        trigger_tags.get("scope"),
                    )
            except Exception as e:
                logger.warning("memorize: trigger_tags generation failed: %s", e)
        result = await self._memory.save_item_with_supersede(
            summary=summary,
            memory_type=memory_type,
            extra=extra,
            source_ref="memorize_tool",
        )
        return f"已记住（{result}）：{summary}"

from __future__ import annotations

from typing import Any

from agent.background.subagent_manager import SubagentManager
from agent.policies.delegation import DelegationPolicy
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
import logging

logger = logging.getLogger(__name__)


class SpawnTool(Tool):
    """Create a background subagent task bound to the current session."""

    def __init__(
        self,
        manager: SubagentManager,
        tool_registry: ToolRegistry,
        policy: DelegationPolicy | None = None,
    ) -> None:
        self._manager = manager
        self._tool_registry = tool_registry
        self._policy = policy or DelegationPolicy()

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "将一个需要多步工具链的任务交给后台 subagent 独立执行。\n"
            "适合 spawn 的场景：\n"
            "- 任务需要串联多个工具（如：先搜索、再读文件、再写结果）\n"
            "- 任务耗时较长，放在主会话会长时间阻塞\n"
            "- 任务上下文独立，结果回传即可，不需要主会话中间介入\n"
            "- 多个子任务可并行推进\n"
            "不适合 spawn 的场景：\n"
            "- 单条 shell 命令或单次工具调用——直接调用对应工具即可\n"
            "- 简单问答或查询——直接回答\n"
            "- 任务需要和用户来回确认\n"
            "调用后立即向用户说明已开始处理，不要等待结果；后台完成后系统会把结果带回。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "交给后台 subagent 的完整任务描述",
                },
                "label": {
                    "type": "string",
                    "description": "可选的短标签，用于显示当前后台任务名称",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **_: Any) -> str:
        ctx = self._tool_registry.get_context()
        channel = str(ctx.get("channel", "") or "").strip()
        chat_id = str(ctx.get("chat_id", "") or "").strip()
        if not channel or not chat_id:
            return "错误：当前会话上下文缺失，无法创建后台任务"
        decision = self._policy.decide(task=task, label=label)
        logger.info(
            "[spawn] decision should_spawn=%s reason=%s confidence=%s source=%s label=%r explicit_call=true",
            decision.should_spawn,
            decision.meta.reason_code,
            decision.meta.confidence,
            decision.meta.source,
            decision.label,
        )
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=channel,
            origin_chat_id=chat_id,
            decision=decision,
        )

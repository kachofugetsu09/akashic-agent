from __future__ import annotations

from typing import Any

from agent.subagent_manager import SubagentManager
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry


class SpawnTool(Tool):
    """Create a background subagent task bound to the current session."""

    def __init__(self, manager: SubagentManager, tool_registry: ToolRegistry) -> None:
        self._manager = manager
        self._tool_registry = tool_registry

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "将一个复杂或耗时较长的任务放到后台 subagent 中执行。"
            "适合会阻塞当前对话、需要较多工具链、或不适合污染主会话上下文的任务。"
            "调用后不要等待结果；应立即向用户说明你已开始处理，完成后会继续回复。"
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
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=channel,
            origin_chat_id=chat_id,
        )

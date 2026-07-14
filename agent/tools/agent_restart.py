from __future__ import annotations

import json
from typing import Any

from agent.control.context import current_turn_id
from agent.restart import RestartCoordinator
from agent.tools.base import Tool
from core.error_context import current_session_key


class AgentRestartTool(Tool):
    name = "agent_restart"
    description = (
        "安全重启当前 Akashic Agent。仅在核心代码或主配置必须重新加载时使用；"
        "MCP 与插件可热重载时不要调用。执行后会等待本轮回复持久化并送达。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "需要完整重启 Agent 的具体原因",
                "minLength": 1,
                "maxLength": 300,
            }
        },
        "required": ["reason"],
        "additionalProperties": False,
    }

    def __init__(self, coordinator: RestartCoordinator) -> None:
        self._coordinator = coordinator

    async def execute(
        self,
        reason: str,
        channel: str = "",
        chat_id: str = "",
        **_: Any,
    ) -> str:
        request = self._coordinator.arm(
            turn_id=current_turn_id.get(),
            session_key=current_session_key.get() or "",
            channel=channel,
            chat_id=chat_id,
            reason=reason,
        )
        return json.dumps(
            {
                "status": "scheduled",
                "requestId": request.id,
                "message": "将在本轮回复持久化并送达后重启。",
            },
            ensure_ascii=False,
        )

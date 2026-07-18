from __future__ import annotations

from agent.tools.base import Tool, ToolResult


class RequestUserConfirmationTool(Tool):
    """显式标记本轮需要用户确认后才能继续。"""

    name = "request_user_confirmation"
    description = (
        "当任务必须等待用户做出明确选择、授权或确认时调用。"
        "调用后，在本轮最终回复中清楚列出要确认的事项；"
        "普通提问、补充信息或修辞问句不要调用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
                "description": "需要用户明确确认的具体事项。",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    async def execute(self, prompt: str, **_: object) -> ToolResult:
        return ToolResult(
            text=f"已标记等待用户确认：{prompt}",
            mobile_attention="confirmation",
        )

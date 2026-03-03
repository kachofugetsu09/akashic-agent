"""
NotifyOwnerTool — 只能发给 owner 的消息推送工具

channel 和 chat_id 在构造时从配置固定，不对 LLM 暴露，
autonomous agent 无法向第三方发送消息。
"""

from __future__ import annotations

import logging
from typing import Any

from agent.tools.base import Tool
from agent.tools.message_push import MessagePushTool

logger = logging.getLogger(__name__)


class NotifyOwnerTool(Tool):
    """向 owner 发送通知，目标渠道硬编码，LLM 只能提供消息内容。"""

    name = "notify_owner"
    description = (
        "向用户发送通知消息。任务完成后调用，发送结果摘要或简短感想。"
        "只需提供消息内容，渠道和目标由系统自动确定。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "要发送给用户的消息内容",
            },
        },
        "required": ["message"],
    }

    def __init__(
        self, push_tool: MessagePushTool, channel: str, chat_id: str
    ) -> None:
        self._push = push_tool
        self._channel = channel
        self._chat_id = chat_id

    async def execute(self, message: str, **_: Any) -> str:
        if not self._channel or not self._chat_id:
            return "未配置 default_channel / default_chat_id，跳过发送"
        if not message or not message.strip():
            return "消息内容为空，跳过发送"
        try:
            result = await self._push.execute(
                channel=self._channel,
                chat_id=self._chat_id,
                message=message.strip(),
            )
            logger.info("[notify_owner] 发送完成: %s", result)
            return result
        except Exception as e:
            logger.warning("[notify_owner] 发送失败: %s", e)
            return f"发送失败: {e}"

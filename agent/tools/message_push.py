"""
统一消息推送工具，替代原有的 telegram_push / qq_push。
agent 通过 channel + chat_id 向任意已注册渠道发送消息。
"""
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent.tools.base import Tool

logger = logging.getLogger(__name__)


class MessagePushTool(Tool):
    name = "message_push"
    description = (
        "向指定渠道的用户或群组主动发送消息。"
        "需要提供渠道名（如 telegram、qq）和目标 chat_id。"
        "chat_id 在对话上下文中可获取，也可由用户直接告知。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "目标渠道名，如 telegram、qq",
            },
            "chat_id": {
                "type": "string",
                "description": "目标会话 ID，私聊为用户 ID，群聊为群号",
            },
            "message": {
                "type": "string",
                "description": "要发送的消息内容",
            },
        },
        "required": ["channel", "chat_id", "message"],
    }

    def __init__(self) -> None:
        self._senders: dict[str, Callable[[str, str], Awaitable[None]]] = {}

    def register_channel(self, channel: str, sender: Callable[[str, str], Awaitable[None]]) -> None:
        """注册一个渠道的发送函数。sender(chat_id, message) -> Awaitable[None]"""
        self._senders[channel] = sender
        logger.debug(f"message_push: 注册渠道 {channel!r}")

    async def execute(self, **kwargs: Any) -> str:
        channel: str = kwargs["channel"]
        chat_id: str = str(kwargs["chat_id"])
        message: str = kwargs["message"]

        sender = self._senders.get(channel)
        if sender is None:
            available = list(self._senders.keys()) or ["（无）"]
            return f"渠道 {channel!r} 未注册，可用渠道：{available}"

        try:
            await sender(chat_id, message)
            preview = message[:60] + "..." if len(message) > 60 else message
            logger.info(f"[message_push] {channel}:{chat_id} ← {preview!r}")
            return f"已发送到 {channel}:{chat_id}"
        except Exception as e:
            logger.error(f"[message_push] 发送失败 {channel}:{chat_id}: {e}")
            return f"发送失败：{e}"

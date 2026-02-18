"""
Telegram 主动推送工具
agent 可调用此工具向 Telegram 用户发送消息，无需等待对方先发消息（需已有历史会话）。
"""
import logging
from typing import Any

from agent.tools.base import Tool

logger = logging.getLogger(__name__)


class TelegramPushTool(Tool):
    """向指定 Telegram 用户主动推送消息"""

    name = "telegram_push"
    description = (
        "向 Telegram 用户主动发送消息。"
        "用户必须曾经给 bot 发过消息（bot 才能知道其 chat_id）。"
        "可通过用户名（不含 @）或数字 chat_id 指定目标。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "description": "目标用户名，不含 @，例如 huashen258",
            },
            "message": {
                "type": "string",
                "description": "要发送的消息内容",
            },
        },
        "required": ["username", "message"],
    }

    def __init__(self, bot, user_map: dict[str, str]) -> None:
        """
        bot:      telegram.Bot 实例（来自 TelegramChannel.bot）
        user_map: TelegramChannel.user_map 的引用（username.lower() → chat_id）
        """
        self._bot = bot
        self._user_map = user_map  # 共享引用，自动感知新用户

    async def execute(self, **kwargs: Any) -> str:
        username: str = kwargs["username"].lstrip("@").lower()
        message: str = kwargs["message"]

        chat_id = self._user_map.get(username)
        if not chat_id:
            return (
                f"未找到用户 @{username} 的会话记录。"
                f"该用户需要先给 bot 发一条消息，bot 才能主动联系他。"
                f"当前已知用户：{list(self._user_map.keys()) or '（无）'}"
            )

        try:
            await self._bot.send_message(chat_id=int(chat_id), text=message)
            logger.info(f"[telegram_push] 已推送消息给 @{username}  chat_id={chat_id}")
            return f"消息已成功发送给 @{username}"
        except Exception as e:
            logger.error(f"[telegram_push] 发送失败 @{username}: {e}")
            return f"发送失败：{e}"

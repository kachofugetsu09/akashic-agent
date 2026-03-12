"""
统一消息推送工具，agent 通过 channel + chat_id 向任意已注册渠道发送消息、文件或图片。
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent.tools.base import Tool

logger = logging.getLogger(__name__)


class MessagePushTool(Tool):
    name = "message_push"
    description = (
        "向指定渠道的用户主动发送消息、文件或图片。"
        "需要提供渠道名（如 telegram、qq）和目标 chat_id。"
        "message/file/image 三者至少提供一个。"
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
                "description": "目标会话 ID",
            },
            "message": {
                "type": "string",
                "description": "要发送的文本内容（可与 file/image 同时提供）",
            },
            "file": {
                "type": "string",
                "description": "要发送的文件本地路径，例如 /tmp/report.pdf",
            },
            "image": {
                "type": "string",
                "description": "要发送的图片本地路径或 URL",
            },
        },
        "required": ["channel", "chat_id"],
    }

    def __init__(self) -> None:
        # channel -> {type: sender_fn}
        self._senders: dict[str, dict[str, Callable[..., Awaitable[None]]]] = {}

    def register_channel(
        self,
        channel: str,
        text: Callable[[str, str], Awaitable[None]] | None = None,
        file: Callable[[str, str, str | None], Awaitable[None]] | None = None,
        image: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        """注册渠道的各类 sender。
        - text(chat_id, message)
        - file(chat_id, file_path, name=None)
        - image(chat_id, image_path_or_url)
        """
        self._senders[channel] = {}
        if text:
            self._senders[channel]["text"] = text
        if file:
            self._senders[channel]["file"] = file
        if image:
            self._senders[channel]["image"] = image
        logger.debug(
            f"message_push: 注册渠道 {channel!r}  支持: {list(self._senders[channel])}"
        )

    async def execute(self, **kwargs: Any) -> str:
        channel: str = kwargs["channel"]
        chat_id: str = str(kwargs["chat_id"])
        message: str | None = kwargs.get("message")
        file: str | None = kwargs.get("file")
        image: str | None = kwargs.get("image")

        if not message and not file and not image:
            return "错误：message、file、image 至少提供一个"

        senders = self._senders.get(channel)
        if senders is None:
            return f"渠道 {channel!r} 未注册，可用渠道：{list(self._senders) or ['（无）']}"

        results = []
        try:
            if message and "text" in senders:
                await senders["text"](chat_id, message)
                preview = message[:60] + "..." if len(message) > 60 else message
                logger.info(f"[message_push] {channel}:{chat_id} ← text: {preview!r}")
                results.append("文本已发送")

            if file:
                if "file" not in senders:
                    results.append(f"渠道 {channel!r} 不支持发送文件")
                else:
                    import os

                    name = os.path.basename(file)
                    await senders["file"](chat_id, file, name)
                    logger.info(f"[message_push] {channel}:{chat_id} ← file: {file!r}")
                    results.append(f"文件 {name!r} 已发送")

            if image:
                if "image" not in senders:
                    results.append(f"渠道 {channel!r} 不支持发送图片")
                else:
                    await senders["image"](chat_id, image)
                    logger.info(
                        f"[message_push] {channel}:{chat_id} ← image: {image!r}"
                    )
                    results.append("图片已发送")

        except Exception as e:
            logger.error(f"[message_push] 发送失败 {channel}:{chat_id}: {e}")
            return f"发送失败：{e}"

        return "；".join(results) if results else f"渠道 {channel!r} 没有可用的 sender"

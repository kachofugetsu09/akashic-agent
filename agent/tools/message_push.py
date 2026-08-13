"""
统一消息推送工具，agent 通过显式 target_channel + target_chat_id 向任意已注册渠道发送消息、文件或图片。
"""

import logging
from dataclasses import replace
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, cast
from uuid import uuid4

from agent.tools.base import Tool
from bus.events import (
    AttachmentKind,
    ChannelAttachment,
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
)
from bus.queue import ChatLane

logger = logging.getLogger(__name__)


class ChannelRegistration:
    def __init__(self, tool: "MessagePushTool", channel: str, token: object) -> None:
        self._tool = tool
        self._channel = channel
        self._token = token
        self._active = True

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        self._tool.unregister_channel(self._channel, self._token)


class MessagePushTool(Tool):
    name = "message_push"
    description = (
        "向指定渠道的用户主动发送消息、文件或图片。"
        "需要提供目标渠道名（如 telegram、qq）和目标 chat_id。"
        "message/file/image 三者至少提供一个。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "target_channel": {
                "type": "string",
                "description": "目标渠道名，如 telegram、qq",
            },
            "target_chat_id": {
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
        "required": ["target_channel", "target_chat_id"],
    }

    def __init__(self, chat_lane: ChatLane | None = None) -> None:
        self._adapters: dict[
            str, Callable[[ChannelMessage], Awaitable[DeliveryReceipt]]
        ] = {}
        self._registration_tokens: dict[str, object] = {}
        self._chat_lane = chat_lane

    def register_channel(
        self,
        channel: str,
        deliver: Callable[[ChannelMessage], Awaitable[DeliveryReceipt]],
    ) -> ChannelRegistration:
        """注册完整逻辑消息的唯一渠道 adapter。"""

        if channel in self._adapters:
            raise RuntimeError(f"message_push 渠道名称重复: {channel}")
        self._adapters[channel] = deliver
        token = object()
        self._registration_tokens[channel] = token
        logger.debug("message_push: 注册渠道 %r", channel)
        return ChannelRegistration(self, channel, token)

    def unregister_channel(self, channel: str, token: object) -> None:
        if self._registration_tokens.get(channel) is not token:
            return
        _ = self._registration_tokens.pop(channel, None)
        _ = self._adapters.pop(channel, None)

    async def execute(self, **kwargs: Any) -> str:
        target_channel = kwargs["target_channel"]
        target_chat_id = str(kwargs["target_chat_id"])
        message: str | None = kwargs.get("message")
        file: str | None = kwargs.get("file")
        image: str | None = kwargs.get("image")
        commit_role = str(kwargs.get("_commit_role") or "").strip()
        raw_outbound_metadata = kwargs.get("_outbound_metadata", {})
        if not isinstance(raw_outbound_metadata, dict):
            raise TypeError("message_push _outbound_metadata 必须是字符串键对象")
        metadata_object = cast(dict[object, object], raw_outbound_metadata)
        if not all(isinstance(key, str) for key in metadata_object):
            raise TypeError("message_push _outbound_metadata 必须是字符串键对象")
        outbound_metadata = cast(dict[str, object], metadata_object)

        if not message and not file and not image:
            return "错误：message、file、image 至少提供一个"
        attachments: list[ChannelAttachment] = []
        if file:
            attachments.append(
                ChannelAttachment(AttachmentKind.FILE, file, Path(file).name)
            )
        if image:
            attachments.append(ChannelAttachment(AttachmentKind.IMAGE, image))
        receipt = await self.dispatch(
            ChannelMessage(
                channel=target_channel,
                chat_id=target_chat_id,
                content=message or "",
                attachments=tuple(attachments),
                metadata=outbound_metadata,
            ),
            commit_role=commit_role,
        )
        if receipt.status is DeliveryStatus.SUCCESS:
            return "消息已发送"
        if receipt.status is DeliveryStatus.PARTIAL:
            return f"消息部分送达：{receipt.detail or '渠道未提交全部内容'}"
        return f"发送失败：{receipt.detail or '渠道未提交消息'}"

    async def dispatch(
        self,
        message: ChannelMessage,
        *,
        commit_role: str = "",
    ) -> DeliveryReceipt:
        """通过单一 adapter 提交完整消息，并保留 chat lane 顺序。"""

        if commit_role != "passive" and message.control_turn_id is None:
            message = replace(message, control_turn_id=f"turn:{uuid4().hex}")
        adapter = self._adapters.get(message.channel)
        if adapter is None:
            return DeliveryReceipt(
                DeliveryStatus.FAILED,
                detail=(
                    f"渠道 {message.channel!r} 未注册，可用渠道："
                    f"{list(self._adapters) or ['（无）']}"
                ),
            )

        async def _deliver() -> DeliveryReceipt:
            receipt = await adapter(message)
            if not isinstance(receipt, DeliveryReceipt):
                raise TypeError("message_push channel adapter 必须返回 DeliveryReceipt")
            return receipt

        if self._chat_lane is not None:
            if commit_role == "passive":
                return await self._chat_lane.run_passive(
                    message.channel,
                    message.chat_id,
                    _deliver,
                )
            return await self._chat_lane.run_non_passive(
                message.channel,
                message.chat_id,
                _deliver,
            )
        return await _deliver()

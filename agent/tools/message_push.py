"""
统一消息推送工具，agent 通过显式 target_channel + target_chat_id 向任意已注册渠道发送消息、文件或图片。
"""

import json
from dataclasses import replace
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, cast
from uuid import uuid4

from agent.tools.base import Tool
from agent.plugin_composition.channels import (
    ChannelDeliveryReceipt,
)
from bus.events import (
    AttachmentKind,
    ChannelAttachment,
    ChannelMessage,
)
from bus.queue import ChatLane

V3ChannelDispatcher = Callable[
    [ChannelMessage, bool],
    Awaitable[ChannelDeliveryReceipt],
]


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
        # ``chat_lane`` remains an accepted constructor argument for callers that
        # construct the shared tool before Core wiring; ordering is now owned by
        # the committed Channel dispatcher.
        _ = chat_lane
        self._v3_dispatcher: V3ChannelDispatcher | None = None

    def bind_v3_channel_dispatcher(self, dispatcher: V3ChannelDispatcher) -> None:
        """Bind Core's exact stable Channel dispatch boundary once."""

        if not callable(dispatcher):
            raise TypeError("v3 channel dispatcher 必须可调用")
        if self._v3_dispatcher is not None:
            raise RuntimeError("v3 channel dispatcher 已绑定")
        self._v3_dispatcher = dispatcher

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
        outbound_metadata = dict(cast(dict[str, object], metadata_object))
        outbound_metadata.setdefault("source", "message_push")

        if not message and not file and not image:
            return "错误：message、file、image 至少提供一个"
        attachments: list[ChannelAttachment] = []
        if file:
            attachments.append(
                ChannelAttachment(AttachmentKind.FILE, file, Path(file).name)
            )
        if image:
            attachments.append(ChannelAttachment(AttachmentKind.IMAGE, image))
        receipt = await self._dispatch_result(
            ChannelMessage(
                channel=target_channel,
                chat_id=target_chat_id,
                content=message or "",
                attachments=tuple(attachments),
                metadata=outbound_metadata,
            ),
            commit_role=commit_role,
        )
        return json.dumps(
            {
                "delivery_id": receipt.delivery_id,
                "status": receipt.status.value,
                "retryable": False,
                "provider_ids": list(receipt.provider_ids),
                "error": receipt.error,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    async def dispatch(
        self,
        message: ChannelMessage,
        *,
        commit_role: str = "",
    ) -> ChannelDeliveryReceipt:
        """通过 committed Channel catalog 提交一条完整逻辑消息。"""

        return await self._dispatch_result(message, commit_role=commit_role)

    async def _dispatch_result(
        self,
        message: ChannelMessage,
        *,
        commit_role: str = "",
    ) -> ChannelDeliveryReceipt:
        """Dispatch through the required committed Channel catalog and fail loudly otherwise."""

        if commit_role != "passive" and message.control_turn_id is None:
            message = replace(message, control_turn_id=f"turn:{uuid4().hex}")
        dispatcher = self._v3_dispatcher
        if dispatcher is None:
            raise RuntimeError("message_push committed Channel dispatcher 未绑定")
        receipt = await dispatcher(message, commit_role == "passive")
        if not isinstance(receipt, ChannelDeliveryReceipt):
            raise TypeError("message_push committed Channel dispatcher 必须返回 ChannelDeliveryReceipt")
        return receipt

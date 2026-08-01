from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from bus.events import (
    AttachmentKind,
    ChannelAttachment,
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
    OutboundMessage,
)
from bus.queue import MessageBus


@dataclass
class OutboundDispatch:
    channel: str
    chat_id: str
    content: str
    thinking: str | None = None
    metadata: dict[str, object] = field(default_factory=dict[str, object])
    media: list[str] = field(default_factory=list[str])
    session_message_id: str | None = None


class OutboundPort(Protocol):
    async def dispatch(self, outbound: OutboundDispatch) -> DeliveryReceipt: ...


class BusOutboundPort:
    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus

    async def dispatch(self, outbound: OutboundDispatch) -> DeliveryReceipt:
        await self._bus.publish_outbound(
            OutboundMessage(
                channel=outbound.channel,
                chat_id=outbound.chat_id,
                content=outbound.content,
                thinking=outbound.thinking,
                metadata=dict(outbound.metadata),
                media=list(outbound.media),
                session_message_id=outbound.session_message_id,
            )
        )
        return DeliveryReceipt(
            DeliveryStatus.SUCCESS,
            canonical_media=tuple(outbound.media),
        )


class PushToolOutboundPort:
    def __init__(self, push_tool: Any) -> None:
        self._push = push_tool

    async def dispatch(self, outbound: OutboundDispatch) -> DeliveryReceipt:
        message = outbound.content.strip()
        channel = outbound.channel.strip()
        chat_id = outbound.chat_id.strip()
        media = [item.strip() for item in outbound.media if item.strip()]
        if (not message and not media) or not channel or not chat_id:
            return DeliveryReceipt(
                DeliveryStatus.FAILED,
                detail="出站消息缺少渠道、会话或内容",
            )
        return await self._push.dispatch(
            ChannelMessage(
                channel=channel,
                chat_id=chat_id,
                content=message,
                attachments=tuple(
                    ChannelAttachment(AttachmentKind.IMAGE, item) for item in media
                ),
                metadata=dict(outbound.metadata),
                session_message_id=outbound.session_message_id,
            )
        )

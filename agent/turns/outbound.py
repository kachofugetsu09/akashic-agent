from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from bus.events import OutboundMessage
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
    async def dispatch(self, outbound: OutboundDispatch) -> bool: ...


class BusOutboundPort:
    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
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
        return True


class PushToolOutboundPort:
    def __init__(self, push_tool: Any) -> None:
        self._push = push_tool

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        message = outbound.content.strip()
        channel = outbound.channel.strip()
        chat_id = outbound.chat_id.strip()
        media = [item.strip() for item in outbound.media if item.strip()]
        if (not message and not media) or not channel or not chat_id:
            return False
        result = await self._push.execute(
            channel=channel,
            chat_id=chat_id,
            message=message,
            image=media[0] if media else None,
            _outbound_metadata=dict(outbound.metadata),
        )
        for image in media[1:]:
            result = await self._push.execute(
                channel=channel,
                chat_id=chat_id,
                image=image,
            )
        return "已发送" in str(result)

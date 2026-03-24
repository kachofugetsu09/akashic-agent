from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Protocol

from bus.events import OutboundMessage


@dataclass
class OutboundDispatch:
    channel: str
    chat_id: str
    content: str
    thinking: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OutboundPort(Protocol):
    async def dispatch(self, outbound: OutboundDispatch) -> bool: ...


class BusOutboundPort:
    def __init__(self, bus: Any) -> None:
        self._bus = bus

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        maybe = self._bus.publish_outbound(
            OutboundMessage(
                channel=outbound.channel,
                chat_id=outbound.chat_id,
                content=outbound.content,
                thinking=outbound.thinking,
                metadata=dict(outbound.metadata or {}),
            )
        )
        if inspect.isawaitable(maybe):
            await maybe
        return True


class PushToolOutboundPort:
    def __init__(self, push_tool: Any) -> None:
        self._push = push_tool

    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        message = str(outbound.content or "").strip()
        channel = str(outbound.channel or "").strip()
        chat_id = str(outbound.chat_id or "").strip()
        if not message or not channel or not chat_id:
            return False
        try:
            result = await self._push.execute(
                channel=channel,
                chat_id=chat_id,
                message=message,
            )
        except Exception:
            return False
        return "已发送" in str(result)

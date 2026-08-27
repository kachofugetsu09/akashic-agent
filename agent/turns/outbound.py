from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent.plugin_composition.channels import (
    AttachmentRef,
    ChannelDeliveryReceipt,
    DeliveryStatus as ChannelDeliveryStatus,
)

from bus.events import (
    AttachmentKind,
    ChannelAttachment,
    ChannelMessage,
    TurnTerminalStatus,
)


@dataclass
class OutboundDispatch:
    channel: str
    chat_id: str
    content: str
    thinking: str | None = None
    reply_to: str | None = None
    metadata: dict[str, object] = field(default_factory=dict[str, object])
    media: list[str] = field(default_factory=list[str])
    attachment_refs: tuple[AttachmentRef, ...] = ()
    session_message_id: str | None = None
    control_turn_id: str | None = None
    execution_attempt_id: str | None = None
    terminal_status: TurnTerminalStatus | None = None


class OutboundPort(Protocol):
    async def dispatch(self, outbound: OutboundDispatch) -> ChannelDeliveryReceipt: ...


class PushToolOutboundPort:
    def __init__(
        self,
        push_tool: Any,
        *,
        commit_role: Literal["", "passive"] = "",
    ) -> None:
        self._push = push_tool
        self._commit_role = commit_role

    async def dispatch(self, outbound: OutboundDispatch) -> ChannelDeliveryReceipt:
        message = outbound.content.strip()
        channel = outbound.channel.strip()
        chat_id = outbound.chat_id.strip()
        media = [item.strip() for item in outbound.media if item.strip()]
        if (not message and not media) or not channel or not chat_id:
            return ChannelDeliveryReceipt(
                delivery_id="invalid-outbound",
                status=ChannelDeliveryStatus.REJECTED,
                error="出站消息缺少渠道、会话或内容",
            )
        receipt = await self._push.dispatch(
            ChannelMessage(
                channel=channel,
                chat_id=chat_id,
                content=message,
                attachments=tuple(
                    ChannelAttachment(AttachmentKind.IMAGE, item) for item in media
                ),
                attachment_refs=outbound.attachment_refs,
                metadata=dict(outbound.metadata),
                reply_to=outbound.reply_to,
                session_message_id=outbound.session_message_id,
                control_turn_id=outbound.control_turn_id,
                execution_attempt_id=outbound.execution_attempt_id,
                terminal_status=outbound.terminal_status,
            ),
            commit_role=self._commit_role,
        )
        if not isinstance(receipt, ChannelDeliveryReceipt):
            raise TypeError("PushToolOutboundPort 必须返回 ChannelDeliveryReceipt")
        return receipt

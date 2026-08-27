from __future__ import annotations

import asyncio

import pytest

from agent.plugin_composition.channels import (
    ChannelDeliveryReceipt,
    DeliveryStatus as ChannelDeliveryStatus,
)
from agent.turns.outbound import OutboundDispatch, PushToolOutboundPort
from agent.tools.message_push import MessagePushTool
from bus.events import ChannelMessage, TurnTerminalStatus
from bus.queue import ChatLane


@pytest.mark.asyncio
async def test_push_tool_outbound_port_forwards_turn_identity_verbatim() -> None:
    delivered: list[ChannelMessage] = []

    class _PushTool:
        async def dispatch(
            self,
            message: ChannelMessage,
            *,
            commit_role: str = "",
        ) -> ChannelDeliveryReceipt:
            delivered.append(message)
            assert commit_role == ""
            return ChannelDeliveryReceipt(
                delivery_id="delivery-1",
                status=ChannelDeliveryStatus.DELIVERED,
            )

    port = PushToolOutboundPort(_PushTool())

    _ = await port.dispatch(
        OutboundDispatch(
            channel="telegram",
            chat_id="123",
            content="hello",
            reply_to="message-1",
            metadata={"source": "passive"},
            media=["/tmp/image.png"],
            session_message_id="telegram:123:5",
            control_turn_id="interaction:authoritative",
            execution_attempt_id="turn:attempt",
            terminal_status=TurnTerminalStatus.COMPLETED,
        )
    )

    assert len(delivered) == 1
    message = delivered[0]
    assert message.control_turn_id == "interaction:authoritative"
    assert message.execution_attempt_id == "turn:attempt"
    assert message.terminal_status is TurnTerminalStatus.COMPLETED
    assert message.reply_to == "message-1"
    assert message.session_message_id == "telegram:123:5"
    assert message.metadata == {"source": "passive"}
    assert message.content == "hello"


@pytest.mark.asyncio
async def test_message_push_assigns_one_independent_turn_per_dispatch() -> None:
    delivered: list[ChannelMessage] = []
    push = MessagePushTool()

    async def deliver(
        message: ChannelMessage,
        _passive: bool,
    ) -> ChannelDeliveryReceipt:
        delivered.append(message)
        return ChannelDeliveryReceipt(
            delivery_id=f"delivery-{len(delivered)}",
            status=ChannelDeliveryStatus.DELIVERED,
        )

    push.bind_v3_channel_dispatcher(deliver)
    for content in ("one", "two"):
        receipt = await push.dispatch(ChannelMessage("mobile", "chat", content))
        assert receipt.status is ChannelDeliveryStatus.DELIVERED

    turn_ids = [message.control_turn_id for message in delivered]
    assert all(
        turn_id is not None and turn_id.startswith("turn:")
        for turn_id in turn_ids
    )
    assert len(set(turn_ids)) == 2


@pytest.mark.asyncio
async def test_passive_outbound_uses_passive_commit_role_without_self_wait() -> None:
    lane = ChatLane()
    delivered: list[tuple[ChannelMessage, str]] = []
    push = MessagePushTool(chat_lane=lane)

    async def deliver(
        message: ChannelMessage,
        passive: bool,
    ) -> ChannelDeliveryReceipt:
        assert passive is True

        async def send() -> None:
            delivered.append((message, "send"))

        await lane.run_passive(message.channel, message.chat_id, send)
        return ChannelDeliveryReceipt(
            delivery_id="delivery-passive",
            status=ChannelDeliveryStatus.DELIVERED,
        )

    push.bind_v3_channel_dispatcher(deliver)
    port = PushToolOutboundPort(push, commit_role="passive")

    await lane.mark_passive_pending("mobile", "chat")
    try:
        receipt = await asyncio.wait_for(
            port.dispatch(
                OutboundDispatch(
                    channel="mobile",
                    chat_id="chat",
                    content="passive reply",
                    reply_to="message-1",
                    session_message_id="mobile:chat:2",
                    control_turn_id="turn:passive",
                    media=["/tmp/image.png"],
                )
            ),
            timeout=0.5,
        )
    finally:
        await lane.mark_passive_done("mobile", "chat")

    assert receipt.status is ChannelDeliveryStatus.DELIVERED
    assert delivered[0][0].reply_to == "message-1"
    assert delivered[0][0].session_message_id == "mobile:chat:2"
    assert delivered[0][0].control_turn_id == "turn:passive"

from __future__ import annotations

import pytest

from agent.turns.outbound import BusOutboundPort, OutboundDispatch, PushToolOutboundPort
from bus.events import ChannelMessage, DeliveryStatus, OutboundMessage
from bus.queue import MessageBus


@pytest.mark.asyncio
async def test_bus_outbound_port_publishes_typed_message() -> None:
    bus = MessageBus()
    port = BusOutboundPort(bus)

    sent = await port.dispatch(
        OutboundDispatch(
            channel="telegram",
            chat_id="123",
            content="hello",
            thinking="reasoning",
            metadata={"source": "passive"},
            media=["/tmp/image.png"],
            control_turn_id="turn:final",
        )
    )
    message = await bus._outbound.get()

    assert sent.status is DeliveryStatus.SUCCESS
    assert message == OutboundMessage(
        channel="telegram",
        chat_id="123",
        content="hello",
        thinking="reasoning",
        metadata={"source": "passive"},
        media=["/tmp/image.png"],
        control_turn_id="turn:final",
    )


@pytest.mark.asyncio
async def test_bus_outbound_port_forwards_control_turn_id_verbatim() -> None:
    bus = MessageBus()
    port = BusOutboundPort(bus)

    _ = await port.dispatch(
        OutboundDispatch(
            channel="telegram",
            chat_id="123",
            content="hello",
            control_turn_id="turn:authoritative",
        )
    )
    message = await bus._outbound.get()

    assert message.control_turn_id == "turn:authoritative"
    assert message.session_message_id is None


@pytest.mark.asyncio
async def test_bus_outbound_port_keeps_control_turn_id_none_when_absent() -> None:
    bus = MessageBus()
    port = BusOutboundPort(bus)

    _ = await port.dispatch(
        OutboundDispatch(
            channel="telegram",
            chat_id="123",
            content="hello",
        )
    )
    message = await bus._outbound.get()

    assert message.control_turn_id is None


@pytest.mark.asyncio
async def test_push_tool_outbound_port_forwards_control_turn_id_verbatim() -> None:
    delivered: list[ChannelMessage] = []

    class _PushTool:
        async def dispatch(self, message: ChannelMessage) -> object:
            delivered.append(message)
            return None

    port = PushToolOutboundPort(_PushTool())

    _ = await port.dispatch(
        OutboundDispatch(
            channel="telegram",
            chat_id="123",
            content="hello",
            metadata={"source": "passive"},
            media=["/tmp/image.png"],
            session_message_id="telegram:123:5",
            control_turn_id="turn:authoritative",
        )
    )

    assert len(delivered) == 1
    message = delivered[0]
    assert message.control_turn_id == "turn:authoritative"
    assert message.session_message_id == "telegram:123:5"
    assert message.metadata == {"source": "passive"}
    assert message.content == "hello"

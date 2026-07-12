from __future__ import annotations

import pytest

from agent.turns.outbound import BusOutboundPort, OutboundDispatch
from bus.events import OutboundMessage
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
        )
    )
    message = await bus._outbound.get()

    assert sent is True
    assert message == OutboundMessage(
        channel="telegram",
        chat_id="123",
        content="hello",
        thinking="reasoning",
        metadata={"source": "passive"},
        media=["/tmp/image.png"],
    )

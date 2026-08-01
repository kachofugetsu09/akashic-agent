from __future__ import annotations

import asyncio

import pytest

from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus, MessageBusCapacityError


@pytest.mark.asyncio
async def test_inbound_capacity_rejection_does_not_leave_chat_lane_pending() -> None:
    bus = MessageBus(inbound_capacity=1, inbound_bytes=4096)
    first = InboundMessage("cli", "user", "one", "first")
    await bus.publish_inbound(first)

    with pytest.raises(MessageBusCapacityError, match="resource-exhausted"):
        await bus.publish_inbound(InboundMessage("cli", "user", "two", "second"))

    assert bus.inbound_size == 1
    assert bus.chat_lane._states[("cli", "one")].passive_turns == 1
    assert ("cli", "two") not in bus.chat_lane._states

    consumed = await bus.consume_inbound()
    await bus.complete_inbound(consumed)
    assert bus.inbound_bytes == 0
    assert bus.chat_lane._states == {}


@pytest.mark.asyncio
async def test_outbound_capacity_rejection_rolls_back_chat_lane_send_count() -> None:
    bus = MessageBus(outbound_capacity=1, outbound_bytes=4096)
    await bus.publish_outbound(OutboundMessage("cli", "one", "first"))

    with pytest.raises(MessageBusCapacityError, match="resource-exhausted"):
        await bus.publish_outbound(OutboundMessage("cli", "two", "second"))

    state = bus.chat_lane._states[("cli", "one")]
    assert state.passive_sends == 1
    assert ("cli", "two") not in bus.chat_lane._states

    bus._running = True
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    await asyncio.sleep(0)
    bus.stop()
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert bus.outbound_bytes == 0


@pytest.mark.asyncio
async def test_cancelled_admission_releases_reservation_before_chat_lane_wait() -> None:
    bus = MessageBus(inbound_capacity=1)
    blocked = asyncio.Event()

    async def wait_for_admission(_channel: str, _chat_id: str) -> None:
        await blocked.wait()

    bus.chat_lane.mark_passive_pending = wait_for_admission  # type: ignore[method-assign]
    publish = asyncio.create_task(
        bus.publish_inbound(InboundMessage("cli", "user", "one", "first"))
    )
    await asyncio.sleep(0)
    assert bus.inbound_reserved_bytes > 0
    publish.cancel()
    with pytest.raises(asyncio.CancelledError):
        await publish
    assert bus.inbound_reserved_bytes == 0
    assert bus.inbound_size == 0

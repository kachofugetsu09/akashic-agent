from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.looping.core import AgentLoop
from agent.looping.ports import LLMConfig
from bus.events import InboundMessage, OutboundMessage


@pytest.mark.asyncio
async def test_run_cleans_active_state_before_inbound_completion_failure() -> None:
    item = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="1",
        content="hello",
    )
    bus = SimpleNamespace(
        consume_inbound=AsyncMock(return_value=item),
        publish_outbound=AsyncMock(),
        complete_inbound=AsyncMock(side_effect=RuntimeError("ack failed")),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop._llm_config = LLMConfig()
    loop.bus = bus
    loop._active_tasks = {}
    loop._active_turn_states = {}
    loop._process_with_runtime_admission = AsyncMock(
        return_value=OutboundMessage(channel="cli", chat_id="1", content="ok")
    )

    with pytest.raises(RuntimeError, match="ack failed"):
        await loop.run()

    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}


@pytest.mark.asyncio
async def test_run_propagates_runtime_cancellation_after_ack() -> None:
    item = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="1",
        content="hello",
    )
    consumed = False
    started = asyncio.Event()

    async def consume_inbound() -> InboundMessage:
        nonlocal consumed
        if consumed:
            raise AssertionError("运行器取消后不应继续消费消息")
        consumed = True
        return item

    async def process(_item: InboundMessage) -> OutboundMessage:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    complete_inbound = AsyncMock()
    loop = AgentLoop.__new__(AgentLoop)
    loop._llm_config = LLMConfig()
    loop.bus = SimpleNamespace(
        consume_inbound=consume_inbound,
        publish_outbound=AsyncMock(),
        complete_inbound=complete_inbound,
    )
    loop._active_tasks = {}
    loop._active_turn_states = {}
    loop._process_with_runtime_admission = process

    run_task = asyncio.create_task(loop.run())
    await started.wait()
    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=0.5)

    complete_inbound.assert_awaited_once_with(item)
    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}
    assert loop._running is False


@pytest.mark.asyncio
async def test_run_waits_for_ack_before_propagating_runtime_cancellation() -> None:
    item = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="1",
        content="hello",
    )
    consumed = False
    started = asyncio.Event()
    ack_started = asyncio.Event()
    release_ack = asyncio.Event()

    async def consume_inbound() -> InboundMessage:
        nonlocal consumed
        if consumed:
            raise AssertionError("运行器取消后不应继续消费消息")
        consumed = True
        return item

    async def process(_item: InboundMessage) -> OutboundMessage:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def complete_inbound(_item: InboundMessage) -> None:
        ack_started.set()
        await release_ack.wait()

    loop = AgentLoop.__new__(AgentLoop)
    loop._llm_config = LLMConfig()
    loop.bus = SimpleNamespace(
        consume_inbound=consume_inbound,
        publish_outbound=AsyncMock(),
        complete_inbound=complete_inbound,
    )
    loop._active_tasks = {}
    loop._active_turn_states = {}
    loop._process_with_runtime_admission = process

    run_task = asyncio.create_task(loop.run())
    await started.wait()
    run_task.cancel()
    await ack_started.wait()
    assert run_task.done() is False

    release_ack.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=0.5)

    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}


@pytest.mark.asyncio
async def test_stop_cancels_active_turn_and_acknowledges_inbound() -> None:
    item = InboundMessage(
        channel="cli",
        sender="user",
        chat_id="1",
        content="hello",
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def process(_item: InboundMessage) -> OutboundMessage:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    bus = SimpleNamespace(
        consume_inbound=AsyncMock(return_value=item),
        publish_outbound=AsyncMock(),
        complete_inbound=AsyncMock(),
    )
    loop = AgentLoop.__new__(AgentLoop)
    loop._llm_config = LLMConfig()
    loop.bus = bus
    loop._active_tasks = {}
    loop._active_turn_states = {}
    loop._process_with_runtime_admission = process

    run_task = asyncio.create_task(loop.run())
    await started.wait()
    loop.stop()
    await asyncio.wait_for(run_task, timeout=0.5)

    assert cancelled.is_set()
    bus.complete_inbound.assert_awaited_once_with(item)
    assert loop._active_tasks == {}
    assert loop._active_turn_states == {}

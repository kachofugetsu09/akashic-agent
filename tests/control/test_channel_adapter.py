from __future__ import annotations

from pathlib import Path
import asyncio
from typing import Any, cast

import pytest

from agent.control.models import TurnRequest, TurnStatus
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from bootstrap.passive_worker import PassiveMessageWorker
from bus.events import InboundMessage, OutboundMessage
from session.store import SessionStore


class _Bus:
    def __init__(self) -> None:
        self.outbound: list[OutboundMessage] = []
        self.completed: list[InboundMessage] = []
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.completions: asyncio.Queue[InboundMessage] = asyncio.Queue()

    async def consume_inbound(self) -> InboundMessage:
        return await self.inbound.get()

    async def publish_outbound(self, message: OutboundMessage) -> None:
        self.outbound.append(message)

    async def complete_inbound(self, message: InboundMessage) -> None:
        self.completed.append(message)
        self.completions.put_nowait(message)


@pytest.mark.asyncio
async def test_channel_adapter_uses_same_conversation_runtime(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest) -> str:
        assert request.metadata["channel"] == "telegram"
        return f"channel:{request.input}"

    runtime = ConversationRuntime(store, execute)
    bus = _Bus()
    worker = PassiveMessageWorker(cast(Any, bus), runtime, cast(Any, object()))
    inbound = InboundMessage("telegram", "user", "42", "hello")
    await worker._run_message(inbound)

    assert [message.content for message in bus.outbound] == ["channel:hello"]
    assert bus.completed == [inbound]
    turns = store.list_turns("telegram:42")
    assert len(turns) == 1
    assert turns[0].final_response == "channel:hello"
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_worker_queues_different_threads_without_blocking_consumer(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    release = asyncio.Event()
    first_started = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if request.thread_id == "telegram:one":
            first_started.set()
            await release.wait()
        return request.input

    runtime = ConversationRuntime(store, execute)
    bus = _Bus()
    worker = PassiveMessageWorker(cast(Any, bus), runtime, cast(Any, object()))
    worker_task = asyncio.create_task(worker.run())
    bus.inbound.put_nowait(InboundMessage("telegram", "user", "one", "first"))
    bus.inbound.put_nowait(InboundMessage("telegram", "user", "two", "second"))
    await asyncio.wait_for(first_started.wait(), 1)

    async def second_is_queued() -> None:
        while not store.list_turns("telegram:two"):
            await asyncio.sleep(0)

    await asyncio.wait_for(second_is_queued(), 1)
    assert store.list_turns("telegram:two")[0].status is TurnStatus.QUEUED
    release.set()
    _ = await asyncio.wait_for(bus.completions.get(), 1)
    _ = await asyncio.wait_for(bus.completions.get(), 1)
    worker.stop()
    await worker_task
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_worker_serializes_same_thread_and_continues_after_failure(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    calls: list[str] = []

    async def execute(request: TurnRequest) -> str:
        calls.append(request.input)
        if request.input == "bad":
            raise RuntimeError("broken turn")
        return request.input

    runtime = ConversationRuntime(store, execute)
    bus = _Bus()
    worker = PassiveMessageWorker(cast(Any, bus), runtime, cast(Any, object()))
    worker_task = asyncio.create_task(worker.run())
    bus.inbound.put_nowait(InboundMessage("telegram", "user", "same", "bad"))
    bus.inbound.put_nowait(InboundMessage("telegram", "user", "same", "good"))
    _ = await asyncio.wait_for(bus.completions.get(), 1)
    _ = await asyncio.wait_for(bus.completions.get(), 1)

    assert calls == ["bad", "good"]
    assert [message.content for message in bus.outbound] == [
        "处理消息时出错，请稍后再试。",
        "good",
    ]
    assert [turn.status for turn in reversed(store.list_turns("telegram:same"))] == [
        TurnStatus.FAILED,
        TurnStatus.COMPLETED,
    ]
    worker.stop()
    await worker_task
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_channel_adapter_preserves_full_outbound_projection(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        return ControlExecutionResult(
            "answer",
            assistant_data={
                "thinking": "reasoning",
                "replyTo": "message-1",
                "media": ["image.png"],
                "metadata": {"render": "card"},
            },
        )

    runtime = ConversationRuntime(store, execute)
    bus = _Bus()
    worker = PassiveMessageWorker(cast(Any, bus), runtime, cast(Any, object()))
    inbound = InboundMessage("telegram", "user", "42", "hello")
    await worker._run_message(inbound)

    assert bus.outbound == [
        OutboundMessage(
            channel="telegram",
            chat_id="42",
            content="answer",
            thinking="reasoning",
            reply_to="message-1",
            media=["image.png"],
            metadata={"render": "card"},
        )
    ]
    await runtime.shutdown()
    store.close()

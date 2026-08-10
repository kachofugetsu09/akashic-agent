from __future__ import annotations

from pathlib import Path
import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.control.models import TurnRequest, TurnStatus
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from bootstrap.passive_worker import PassiveMessageWorker
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from session.manager import SessionManager
from session.store import SessionStore


class _Bus:
    def __init__(self) -> None:
        self.outbound: list[OutboundMessage] = []
        self.completed: list[InboundMessage] = []
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.completions: asyncio.Queue[InboundMessage] = asyncio.Queue()

    async def consume_inbound(self) -> InboundMessage:
        return await self.inbound.get()

    async def recover_durable_inbounds(self) -> None:
        return None

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
        assert request.metadata["inboundMetadata"] == {"reply_to_message_id": "m1"}
        return f"channel:{request.input}"

    runtime = ConversationRuntime(store, execute)
    bus = _Bus()
    worker = PassiveMessageWorker(cast(Any, bus), runtime, cast(Any, object()))
    inbound = InboundMessage(
        "telegram",
        "user",
        "42",
        "hello",
        metadata={"reply_to_message_id": "m1"},
    )
    await worker._run_message(inbound)

    assert [message.content for message in bus.outbound] == ["channel:hello"]
    assert bus.completed == [inbound]
    turns = store.list_turns("telegram:42")
    assert len(turns) == 1
    assert turns[0].final_response == "channel:hello"
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_channel_adapter_releases_session_admission_after_completion(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:one"
    manager.save(manager.get_or_create(session_key))
    _, admission_id = manager.admit_existing(session_key)

    async def execute(request: TurnRequest) -> str:
        return request.input

    runtime = ConversationRuntime(manager.control_store, execute)
    bus = _Bus()
    worker = PassiveMessageWorker(
        cast(Any, bus),
        runtime,
        cast(Any, SimpleNamespace(session_manager=manager)),
    )
    inbound = InboundMessage(
        "mobile",
        "device",
        "one",
        "hello",
        metadata={"session_key_override": session_key},
        session_admission_id=admission_id,
    )

    await worker._run_message(inbound)

    dashboard_store = SessionStore(manager.db_path)
    assert dashboard_store.delete_session(session_key, cascade=True)
    dashboard_store.close()
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_recovered_mobile_handoff_with_canonical_user_skips_new_turn(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:existing"
    session = manager.get_or_create(session_key)
    session.add_message("user", "hello", client_message_id="client:1")
    manager.save(session)
    bus = MessageBus()
    bus.bind_durable_inbound_store(manager.control_store)
    inbound = InboundMessage(
        "mobile",
        "device:1",
        "existing",
        "hello",
        metadata={"session_key_override": session_key, "client_message_id": "client:1"},
    )
    await bus.publish_inbound(inbound)
    recovered = await bus.consume_inbound()
    worker = PassiveMessageWorker(
        bus,
        cast(Any, object()),
        cast(Any, SimpleNamespace(session_manager=manager)),
    )

    assert isinstance(recovered, InboundMessage)
    await worker._run_message(recovered)

    assert manager.control_store.list_turns(session_key) == []
    assert manager.control_store.list_inbound_handoffs() == []
    manager.close()


@pytest.mark.asyncio
async def test_recovered_mobile_handoff_in_interrupted_attempt_is_not_reenqueued(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:interrupted"
    reached = asyncio.Event()

    async def execute(_request: TurnRequest) -> str:
        reached.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    runtime = ConversationRuntime(manager.control_store, execute)
    handle = await runtime.start_turn(
        TurnRequest(
            session_key,
            "u1",
            {
                "inboundMetadata": {"client_message_id": "client:interrupted"},
            },
        )
    )
    await reached.wait()
    assert (await handle.interrupt()).status is TurnStatus.INTERRUPTED

    bus = _Bus()
    worker = PassiveMessageWorker(
        cast(Any, bus),
        runtime,
        cast(Any, SimpleNamespace(session_manager=manager)),
    )
    recovered = InboundMessage(
        "mobile",
        "device:1",
        "interrupted",
        "u1",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client:interrupted",
        },
        handoff_id="handoff:interrupted",
    )

    await worker._run_message(recovered)

    assert bus.completed == [recovered]
    assert len(manager.control_store.list_turns(session_key)) == 1
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_worker_executes_different_threads_without_blocking_consumer(
    tmp_path: Path,
) -> None:
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

    async def second_is_completed() -> None:
        while (
            not (turns := store.list_turns("telegram:two"))
            or turns[0].status is not TurnStatus.COMPLETED
        ):
            await asyncio.sleep(0)

    await asyncio.wait_for(second_is_completed(), 1)
    assert store.list_turns("telegram:one")[0].status is TurnStatus.IN_PROGRESS
    release.set()
    _ = await asyncio.wait_for(bus.completions.get(), 1)
    _ = await asyncio.wait_for(bus.completions.get(), 1)
    worker.stop()
    await worker_task
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_worker_waits_for_terminal_before_admitting_next_message(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    release = asyncio.Event()
    first_started = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if request.input == "u1":
            first_started.set()
            await release.wait()
        return request.input

    runtime = ConversationRuntime(store, execute)
    bus = _Bus()
    worker = PassiveMessageWorker(cast(Any, bus), runtime, cast(Any, object()))
    worker_task = asyncio.create_task(worker.run())
    bus.inbound.put_nowait(InboundMessage("telegram", "user", "same", "u1"))
    bus.inbound.put_nowait(InboundMessage("telegram", "user", "same", "u2"))

    await asyncio.wait_for(first_started.wait(), 1)
    assert len(store.list_turns("telegram:same")) == 1
    assert bus.completed == []
    release.set()
    _ = await asyncio.wait_for(bus.completions.get(), 1)
    _ = await asyncio.wait_for(bus.completions.get(), 1)

    assert [message.content for message in bus.outbound] == ["u1", "u2"]
    turns = store.list_turns("telegram:same")
    assert len(turns) == 2
    assert all(turn.status is TurnStatus.COMPLETED for turn in turns)
    worker.stop()
    await worker_task
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_channel_adapter_preserves_full_outbound_projection(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        return ControlExecutionResult(
            "answer",
            assistant_data={
                "thinking": "reasoning",
                "replyTo": "message-1",
                "media": ["image.png"],
                "metadata": {"render": "card"},
                "sessionMessageId": "telegram:42:2",
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
            session_message_id="telegram:42:2",
        )
    ]
    assert bus.outbound[0].session_message_id == "telegram:42:2"
    await runtime.shutdown()
    store.close()

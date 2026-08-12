from __future__ import annotations

from pathlib import Path
import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

from agent.control.models import TurnRequest, TurnStatus
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from bootstrap.passive_worker import PassiveMessageWorker
from bus.events import InboundMessage, OutboundMessage, TurnTerminalStatus
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


async def _wait_for(predicate: Callable[[], bool], *, timeout: float = 3.0) -> None:
    """轮询等待断言条件成立，测试内部辅助函数。"""

    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(_poll(), timeout=timeout)


def _real_worker(
    manager: SessionManager,
    runtime: ConversationRuntime,
) -> tuple[MessageBus, PassiveMessageWorker]:
    """构造绑定同一 control_store 的真实 MessageBus 与 worker。"""

    bus = MessageBus()
    bus.bind_durable_inbound_store(manager.control_store)
    worker = PassiveMessageWorker(
        bus,
        runtime,
        cast(Any, SimpleNamespace(session_manager=manager)),
    )
    return bus, worker


async def _consume_message(bus: MessageBus) -> InboundMessage:
    """消费并证明该测试路径拿到的是渠道消息而非 spawn completion。"""

    item = await bus.consume_inbound()
    assert isinstance(item, InboundMessage)
    return item


def _mobile_item(
    chat_id: str,
    content: str,
    client_message_id: str,
) -> InboundMessage:
    return InboundMessage(
        "mobile",
        "device:1",
        chat_id,
        content,
        metadata={"client_message_id": client_message_id},
    )


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


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


@pytest.mark.asyncio
async def test_recovered_mobile_handoff_without_turn_creates_one_turn_and_delivers(
    tmp_path: Path,
) -> None:
    session_key = "mobile:existing"
    manager = SessionManager(tmp_path / "workspace")
    session = manager.get_or_create(session_key)
    session.add_message("user", "hello", client_message_id="client:1")
    manager.save(session)

    async def execute(request: TurnRequest) -> str:
        return f"echo:{request.input}"

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def on_outbound(msg: OutboundMessage) -> None:
        delivered.append(msg)

    bus.subscribe_outbound("mobile", on_outbound)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    inbound = _mobile_item("existing", "hello", "client:1")
    await bus.publish_inbound(inbound)
    recovered = await _consume_message(bus)
    await worker._run_message(recovered)

    # 1. canonical user 已有但无匹配 turn：仍正常 start_turn 一次并投递。
    turns = manager.control_store.list_turns(session_key)
    assert len(turns) == 1
    assert turns[0].status is TurnStatus.COMPLETED
    assert manager.control_store.list_inbound_handoffs() == []
    assert [msg.content for msg in delivered] == ["echo:hello"]
    assert delivered[0].control_turn_id == turns[0].id
    await _cancel_task(dispatch)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_recovered_mobile_handoff_with_completed_turn_redelivers_and_acks(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:terminal"
    manager.save(manager.get_or_create(session_key))

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        return ControlExecutionResult(
            "persisted-answer",
            assistant_data={
                "sessionMessageId": "mobile:terminal:5",
                "metadata": {"client_message_id": "client:previous-attempt"},
            },
        )

    runtime = ConversationRuntime(manager.control_store, execute)
    # 进程1：turn 已 terminal，但 handoff 未 ACK（崩溃窗口）。
    handle = await runtime.start_turn(
        TurnRequest(
            session_key,
            "hello",
            {"inboundMetadata": {"client_message_id": "client:t"}},
        )
    )
    assert (await handle.result()).status is TurnStatus.COMPLETED

    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def on_outbound(msg: OutboundMessage) -> None:
        delivered.append(msg)

    bus.subscribe_outbound("mobile", on_outbound)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    inbound = _mobile_item("terminal", "hello", "client:t")
    await bus.publish_inbound(inbound)
    recovered = await _consume_message(bus)
    with caplog.at_level(logging.INFO, logger="bootstrap.passive_worker"):
        await worker._run_message(recovered)

    # 1. terminal turn 匹配：不创建第二 turn，用权威终态投影重投递后才 ACK。
    turns = manager.control_store.list_turns(session_key)
    assert len(turns) == 1
    assert [msg.content for msg in delivered] == ["persisted-answer"]
    assert delivered[0].control_turn_id == turns[0].id
    assert delivered[0].metadata["client_message_id"] == "client:t"
    assert delivered[0].session_message_id == "mobile:terminal:5"
    assert manager.control_store.list_inbound_handoffs() == []
    recovery_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("turn_id") == turns[0].id
        and str(record.akashic_fields.get("event", "")).startswith(
            "tl:worker.terminal."
        )
    ]
    assert [record.akashic_fields["event"] for record in recovery_records] == [
        "tl:worker.terminal.start",
        "tl:worker.terminal.done",
    ]
    assert all(
        record.akashic_fields["counts"] == "mode=recovery"
        for record in recovery_records
    )
    await _cancel_task(dispatch)
    await runtime.shutdown()
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
            {"inboundMetadata": {"client_message_id": "client:interrupted"}},
        )
    )
    await reached.wait()
    assert (await handle.interrupt()).status is TurnStatus.INTERRUPTED

    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def commit_mobile_terminal(message: OutboundMessage) -> None:
        delivered.append(message)

    _ = bus.subscribe_outbound("mobile", commit_mobile_terminal)
    dispatcher = asyncio.create_task(bus.dispatch_outbound())
    inbound = _mobile_item("interrupted", "u1", "client:interrupted")
    await bus.publish_inbound(inbound)
    recovered = await _consume_message(bus)
    await worker._run_message(recovered)

    # 1. 恢复态 interrupted 也经 typed durable terminal barrier 后才删 handoff。
    assert len(manager.control_store.list_turns(session_key)) == 1
    assert len(delivered) == 1
    assert delivered[0].terminal_status is TurnTerminalStatus.INTERRUPTED
    assert delivered[0].metadata["client_message_id"] == "client:interrupted"
    assert manager.control_store.list_inbound_handoffs() == []
    assert bus._inbound_accepted == {}
    bus.stop()
    await dispatcher
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_capacity_busy_waits_then_creates_single_turn_and_delivers(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_a, session_b = "mobile:one", "mobile:two"
    manager.save(manager.get_or_create(session_a))
    manager.save(manager.get_or_create(session_b))
    first_started = asyncio.Event()
    release = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if request.thread_id == session_a:
            first_started.set()
            await release.wait()
        return f"echo:{request.input}"

    runtime = ConversationRuntime(manager.control_store, execute, max_active_turns=1)
    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def on_outbound(msg: OutboundMessage) -> None:
        delivered.append(msg)

    bus.subscribe_outbound("mobile", on_outbound)
    worker_task = asyncio.create_task(worker.run())
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    await bus.publish_inbound(_mobile_item("one", "u1", "client:a"))
    await asyncio.wait_for(first_started.wait(), timeout=2)
    await bus.publish_inbound(_mobile_item("two", "u2", "client:b"))

    # 1. 容量占满：B 等待期间 row 保留、不建 turn、无 outbound。
    await _wait_for(lambda: len(manager.control_store.list_inbound_handoffs()) == 2)
    await _wait_for(
        lambda: manager.control_store.list_turns(session_b, limit=10) == []
        and manager.control_store._conn.execute(
            "SELECT 1 FROM session_admissions WHERE session_key = ?",
            (session_b,),
        ).fetchone()
        is None
    )
    assert len(manager.control_store.list_turns(session_a, limit=10)) == 1
    assert delivered == []

    # 2. 释放容量：B 只建一次 turn 并最终投递。
    release.set()
    await _wait_for(lambda: manager.control_store.list_inbound_handoffs() == [])
    assert len(manager.control_store.list_turns(session_b, limit=10)) == 1
    assert manager.control_store.list_turns(session_b, limit=10)[0].status is (
        TurnStatus.COMPLETED
    )
    assert [msg.content for msg in delivered] == ["echo:u1", "echo:u2"]
    assert len({msg.control_turn_id for msg in delivered}) == 2
    await _cancel_task(worker_task)
    await _cancel_task(dispatch)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_capacity_bytes_includes_request_waits_without_busy_polling(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_a, session_b = "mobile:big", "mobile:small"
    manager.save(manager.get_or_create(session_a))
    manager.save(manager.get_or_create(session_b))
    first_started = asyncio.Event()
    release = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if request.thread_id == session_a:
            first_started.set()
            await release.wait()
        return f"echo:{request.input}"

    # A(900 字符) 实际计费 1209B + B(200 字符) 515B = 1724B > max 1500B：
    # 请求必须计入容量判断，否则会误判可立即通过并忙轮询 start_turn。
    runtime = ConversationRuntime(
        manager.control_store, execute, max_active_bytes=1500, max_active_turns=16
    )
    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def on_outbound(msg: OutboundMessage) -> None:
        delivered.append(msg)

    bus.subscribe_outbound("mobile", on_outbound)
    worker_task = asyncio.create_task(worker.run())
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    await bus.publish_inbound(_mobile_item("big", "x" * 900, "client:big"))
    await asyncio.wait_for(first_started.wait(), timeout=2)

    attempts = {"count": 0}
    original_start = runtime.start_turn

    async def counting_start(request: TurnRequest) -> Any:
        attempts["count"] += 1
        return await original_start(request)

    runtime.start_turn = counting_start  # type: ignore[method-assign]
    await bus.publish_inbound(_mobile_item("small", "y" * 200, "client:small"))

    # 1. B 的第一次 start_turn 被拒后进入事件等待：不建 turn、无 outbound。
    await _wait_for(lambda: attempts["count"] == 1)
    await asyncio.sleep(0.2)
    assert attempts["count"] == 1
    assert manager.control_store.list_turns(session_b, limit=10) == []
    assert delivered == []

    # 2. 释放容量后只重试一次并只建一次 turn，无忙轮询。
    release.set()
    await _wait_for(lambda: manager.control_store.list_inbound_handoffs() == [])
    assert attempts["count"] == 2
    turns_b = manager.control_store.list_turns(session_b, limit=10)
    assert len(turns_b) == 1
    assert turns_b[0].status is TurnStatus.COMPLETED
    assert [msg.content for msg in delivered] == [
        f"echo:{'x' * 900}",
        f"echo:{'y' * 200}",
    ]
    await _cancel_task(worker_task)
    await _cancel_task(dispatch)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_worker_cancelled_while_waiting_capacity_keeps_handoff(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_a, session_b = "mobile:one", "mobile:two"
    manager.save(manager.get_or_create(session_a))
    manager.save(manager.get_or_create(session_b))
    first_started = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if request.thread_id == session_a:
            first_started.set()
            await asyncio.Event().wait()
        return request.input

    runtime = ConversationRuntime(manager.control_store, execute, max_active_turns=1)
    bus, worker = _real_worker(manager, runtime)
    worker_task = asyncio.create_task(worker.run())
    await bus.publish_inbound(_mobile_item("one", "u1", "client:a"))
    await asyncio.wait_for(first_started.wait(), timeout=2)
    await bus.publish_inbound(_mobile_item("two", "u2", "client:b"))
    await _wait_for(lambda: len(manager.control_store.list_inbound_handoffs()) == 2)

    # 1. 等待容量期间取消：row 保留、不建 turn、无 outbound。
    await _cancel_task(worker_task)
    assert len(manager.control_store.list_inbound_handoffs()) == 2
    assert manager.control_store.list_turns(session_b, limit=10) == []
    assert bus.outbound_size == 0

    # 2. admission 释放后同一会话可再次取得。
    _, readmission = manager.admit_existing(session_b)
    manager.release_admission(readmission)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_runtime_closed_while_waiting_capacity_keeps_handoff(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_a, session_b = "mobile:one", "mobile:two"
    manager.save(manager.get_or_create(session_a))
    manager.save(manager.get_or_create(session_b))
    first_started = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if request.thread_id == session_a:
            first_started.set()
            await asyncio.Event().wait()
        return request.input

    runtime = ConversationRuntime(manager.control_store, execute, max_active_turns=1)
    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def commit_mobile_terminal(message: OutboundMessage) -> None:
        delivered.append(message)

    _ = bus.subscribe_outbound("mobile", commit_mobile_terminal)
    dispatcher = asyncio.create_task(bus.dispatch_outbound())
    worker_task = asyncio.create_task(worker.run())
    await bus.publish_inbound(_mobile_item("one", "u1", "client:a"))
    await asyncio.wait_for(first_started.wait(), timeout=2)
    await bus.publish_inbound(_mobile_item("two", "u2", "client:b"))
    await _wait_for(lambda: len(manager.control_store.list_inbound_handoffs()) == 2)

    # 1. 关闭唤醒等待并暴露 RuntimeClosedError：B 不建 turn、row 保留。
    await runtime.shutdown()
    await _wait_for(
        lambda: {
            row["dedupe_key"] for row in manager.control_store.list_inbound_handoffs()
        }
        == {"mobile:two:client:b"}
    )
    worker.stop()
    await worker_task
    assert manager.control_store.list_turns(session_b, limit=10) == []

    # 2. A 只有在 typed CANCELLED terminal callback 成功后才删 handoff。
    assert len(delivered) == 1
    assert delivered[0].terminal_status is TurnTerminalStatus.CANCELLED
    assert delivered[0].metadata["client_message_id"] == "client:a"

    # 3. B 未建 turn，不发送“请重发”假 final；durable row 留给新 runtime。
    assert bus.outbound_size == 0
    bus.stop()
    await dispatcher
    manager.close()


@pytest.mark.asyncio
async def test_restart_cancel_resumes_waiting_mobile_handoff_in_same_process(
    tmp_path: Path,
) -> None:
    """restart 取消恢复准入后，同一 worker 原地继续已 accepted 的 handoff。"""

    manager = SessionManager(tmp_path / "workspace")
    caller_key = "control:restart-owner"
    waiting_key = "mobile:restart-waiting"
    manager.save(manager.get_or_create(caller_key))
    manager.save(manager.get_or_create(waiting_key))
    caller_started = asyncio.Event()
    release_caller = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if request.thread_id == caller_key:
            caller_started.set()
            await release_caller.wait()
        return f"echo:{request.input}"

    runtime = ConversationRuntime(manager.control_store, execute)
    caller = await runtime.start_turn(TurnRequest(caller_key, "restart", {}))
    await asyncio.wait_for(caller_started.wait(), timeout=2)
    runtime.quiesce_for_restart(caller.id)

    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def commit_mobile_terminal(message: OutboundMessage) -> None:
        delivered.append(message)

    bus.subscribe_outbound("mobile", commit_mobile_terminal)
    dispatcher = asyncio.create_task(bus.dispatch_outbound())
    worker_task = asyncio.create_task(worker.run())
    await bus.publish_inbound(
        _mobile_item("restart-waiting", "hello", "client:restart-waiting")
    )
    await _wait_for(lambda: len(bus._inbound_accepted) == 1)
    await asyncio.sleep(0.05)
    assert manager.control_store.list_turns(waiting_key, limit=10) == []
    assert bus.inbound_size == 0

    # 1. 恢复栅栏后不依赖新消息或进程重启，原 accepted owner 直接建立 turn。
    runtime.resume_after_restart_cancel(caller.id)
    await _wait_for(lambda: manager.control_store.list_inbound_handoffs() == [])
    waiting_turns = manager.control_store.list_turns(waiting_key, limit=10)
    assert len(waiting_turns) == 1
    assert waiting_turns[0].status is TurnStatus.COMPLETED
    assert len(delivered) == 1
    assert delivered[0].control_turn_id == waiting_turns[0].id
    assert bus._inbound_accepted == {}

    release_caller.set()
    assert (await caller.result()).status is TurnStatus.COMPLETED
    worker.stop()
    await worker_task
    bus.stop()
    await dispatcher
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_create_turn_oserror_keeps_handoff_and_releases_admission(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:oserror"
    manager.save(manager.get_or_create(session_key))

    async def execute(_request: TurnRequest) -> str:
        return "unreachable"

    runtime = ConversationRuntime(manager.control_store, execute)

    def fail_create(_record: object) -> object:
        raise OSError("disk full")

    manager.control_store.create_turn = fail_create  # type: ignore[method-assign]
    bus, worker = _real_worker(manager, runtime)
    inbound = _mobile_item("oserror", "hello", "client:o")
    await bus.publish_inbound(inbound)
    consumed = await _consume_message(bus)
    with pytest.raises(OSError, match="disk full"):
        await worker._run_message(consumed)

    # 1. create_turn 失败：原异常可观察，row 保留、无 turn。
    assert len(manager.control_store.list_inbound_handoffs()) == 1
    assert manager.control_store.list_turns(session_key, limit=10) == []
    assert id(consumed) in bus._inbound_accepted

    # 2. admission 恰一次释放，可再次取得。
    _, readmission = manager.admit_existing(session_key)
    manager.release_admission(readmission)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_terminal_handoff_retained_until_dispatcher_delivers(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:p1"
    manager.save(manager.get_or_create(session_key))

    async def execute(request: TurnRequest) -> str:
        return f"echo:{request.input}"

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    inbound = _mobile_item("p1", "hello", "client:p1")
    await bus.publish_inbound(inbound)
    consumed = await _consume_message(bus)
    result_task = await worker._admit_message(consumed)
    assert result_task is not None
    await _wait_for(
        lambda: bool(manager.control_store.list_turns(session_key))
        and manager.control_store.list_turns(session_key)[0].status
        is TurnStatus.COMPLETED
    )

    # 1. dispatcher 未启动：SessionDB terminal 已落，但 handoff row 仍在。
    assert len(manager.control_store.list_inbound_handoffs()) == 1
    assert bus.outbound_size == 1
    delivered: list[OutboundMessage] = []

    async def on_outbound(msg: OutboundMessage) -> None:
        delivered.append(msg)

    bus.subscribe_outbound("mobile", on_outbound)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    await asyncio.wait_for(result_task, timeout=2)

    # 2. 实际送达后 worker 才完成 handoff。
    assert manager.control_store.list_inbound_handoffs() == []
    assert [msg.content for msg in delivered] == ["echo:hello"]
    await _cancel_task(dispatch)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_handoff_deleted_only_after_callback_durable_commit(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:order"
    manager.save(manager.get_or_create(session_key))
    events: list[str] = []

    async def execute(request: TurnRequest) -> str:
        return f"echo:{request.input}"

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    callback_entered = asyncio.Event()
    release_callback = asyncio.Event()

    async def on_outbound(_msg: OutboundMessage) -> None:
        callback_entered.set()
        events.append("callback_entered")
        await release_callback.wait()
        events.append("callback_committed")

    bus.subscribe_outbound("mobile", on_outbound)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    inbound = _mobile_item("order", "hello", "client:o")
    await bus.publish_inbound(inbound)
    consumed = await _consume_message(bus)
    result_task = await worker._admit_message(consumed)
    assert result_task is not None
    await _wait_for(
        lambda: bool(manager.control_store.list_turns(session_key))
        and manager.control_store.list_turns(session_key)[0].status
        is TurnStatus.COMPLETED
    )
    await asyncio.wait_for(callback_entered.wait(), timeout=2)

    # 1. SessionDB terminal 已落、callback 已进入但未提交：handoff 仍在。
    assert events == ["callback_entered"]
    assert len(manager.control_store.list_inbound_handoffs()) == 1
    assert not result_task.done()
    release_callback.set()
    await asyncio.wait_for(result_task, timeout=2)

    # 2. 顺序：callback 提交成功后才删除 handoff。
    assert events == ["callback_entered", "callback_committed"]
    assert manager.control_store.list_inbound_handoffs() == []
    await _cancel_task(dispatch)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_handoff_retained_when_callback_fails_twice(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:fail2"
    manager.save(manager.get_or_create(session_key))
    attempts = {"count": 0}

    async def execute(request: TurnRequest) -> str:
        return f"echo:{request.input}"

    async def on_outbound(_msg: OutboundMessage) -> None:
        attempts["count"] += 1
        raise RuntimeError("channel down")

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    bus.subscribe_outbound("mobile", on_outbound)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    inbound = _mobile_item("fail2", "hello", "client:f")
    await bus.publish_inbound(inbound)
    consumed = await _consume_message(bus)
    with pytest.raises(RuntimeError, match="handoff retained"):
        await worker._run_message(consumed)

    # 1. callback 两次失败：fail-loud，row 与 owner 保留，不伪成功删除。
    assert attempts["count"] >= 2
    assert len(manager.control_store.list_inbound_handoffs()) == 1
    assert id(consumed) in bus._inbound_accepted

    # 2. 失败路径 finally 恰一次释放 session admission，同一会话可再次 admit。
    _, readmission = manager.admit_existing(session_key)
    manager.release_admission(readmission)
    await _cancel_task(dispatch)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_result_task_cancel_releases_admission_keeps_handoff(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:rcancel"
    manager.save(manager.get_or_create(session_key))
    entered = asyncio.Event()

    async def execute(_request: TurnRequest) -> str:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    inbound = _mobile_item("rcancel", "hello", "client:r")
    await bus.publish_inbound(inbound)
    consumed = await _consume_message(bus)
    result_task = await worker._admit_message(consumed)
    assert result_task is not None
    await asyncio.wait_for(entered.wait(), timeout=2)

    # 1. worker.stop 语义等价：result task 在等待 turn 期间被取消。
    await _cancel_task(result_task)

    # 2. row 保留、无 owner 清除，session admission 恰一次释放可再次取得。
    assert len(manager.control_store.list_inbound_handoffs()) == 1
    assert id(consumed) in bus._inbound_accepted
    _, readmission = manager.admit_existing(session_key)
    manager.release_admission(readmission)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_handoff_retained_when_dispatcher_cancelled(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:dc"
    manager.save(manager.get_or_create(session_key))
    entered = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        return f"echo:{request.input}"

    async def on_outbound(_msg: OutboundMessage) -> None:
        entered.set()
        await asyncio.Event().wait()

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    bus.subscribe_outbound("mobile", on_outbound)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    inbound = _mobile_item("dc", "hello", "client:d")
    await bus.publish_inbound(inbound)
    consumed = await _consume_message(bus)
    result_task = await worker._admit_message(consumed)
    assert result_task is not None
    await asyncio.wait_for(entered.wait(), timeout=2)

    # 1. dispatch 取消把 receipt 收束为未送达：worker fail-loud，row 保留。
    await _cancel_task(dispatch)
    with pytest.raises(RuntimeError, match="handoff retained"):
        await result_task
    assert len(manager.control_store.list_inbound_handoffs()) == 1
    assert id(consumed) in bus._inbound_accepted
    assert bus._pending_outbound_receipts == set()
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_failed_outbound_carries_authoritative_turn_id_across_threads(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_a, session_b = "mobile:afail", "mobile:bsuccess"
    manager.save(manager.get_or_create(session_a))
    manager.save(manager.get_or_create(session_b))

    async def execute(request: TurnRequest) -> str:
        if request.thread_id == session_a:
            raise RuntimeError("model crash")
        return f"echo:{request.input}"

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def on_outbound(msg: OutboundMessage) -> None:
        delivered.append(msg)

    bus.subscribe_outbound("mobile", on_outbound)
    worker_task = asyncio.create_task(worker.run())
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    await bus.publish_inbound(_mobile_item("afail", "a", "client:a"))
    await bus.publish_inbound(_mobile_item("bsuccess", "b", "client:b"))
    await _wait_for(lambda: manager.control_store.list_inbound_handoffs() == [])

    turns_a = manager.control_store.list_turns(session_a, limit=10)
    turns_b = manager.control_store.list_turns(session_b, limit=10)
    assert len(turns_a) == 1 and turns_a[0].status is TurnStatus.FAILED
    assert len(turns_b) == 1 and turns_b[0].status is TurnStatus.COMPLETED
    failed = next(msg for msg in delivered if msg.content.startswith("处理消息时出错"))
    echoed = next(msg for msg in delivered if msg.content == "echo:b")

    # 1. FAILED 与 COMPLETED 都显式携带各自权威 turn id，A/B 不漂移。
    assert failed.control_turn_id == turns_a[0].id
    assert echoed.control_turn_id == turns_b[0].id
    assert turns_a[0].id != turns_b[0].id
    await _cancel_task(worker_task)
    await _cancel_task(dispatch)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_handoff_retained_without_subscriber(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:nosub"
    manager.save(manager.get_or_create(session_key))

    async def execute(request: TurnRequest) -> str:
        return f"echo:{request.input}"

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    inbound = _mobile_item("nosub", "hello", "client:n")
    await bus.publish_inbound(inbound)
    consumed = await _consume_message(bus)
    with pytest.raises(RuntimeError, match="handoff retained"):
        await worker._run_message(consumed)

    # 1. 无 subscriber 不算 delivered：row 与 owner 保留。
    assert len(manager.control_store.list_inbound_handoffs()) == 1
    assert id(consumed) in bus._inbound_accepted
    await _cancel_task(dispatch)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_restart_recovery_redelivers_terminals_and_creates_missing_turn_once(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    done_key, fail_key, noturn_key, interrupt_key = (
        "mobile:restart-done",
        "mobile:restart-fail",
        "mobile:restart-noturn",
        "mobile:restart-interrupt",
    )
    interrupt_entered = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if request.input == "fail":
            raise RuntimeError("boom")
        if request.input == "interrupt":
            interrupt_entered.set()
            await asyncio.Event().wait()
        return f"echo:{request.input}"

    # 进程1：真实 store+bus+runtime+worker，terminal 已落但 handoff 全部保留。
    manager1 = SessionManager(workspace)
    for key in (done_key, fail_key, noturn_key, interrupt_key):
        manager1.save(manager1.get_or_create(key))
    runtime1 = ConversationRuntime(manager1.control_store, execute)
    bus1, worker1 = _real_worker(manager1, runtime1)
    worker1_task = asyncio.create_task(worker1.run())
    await bus1.publish_inbound(_mobile_item("restart-done", "done", "client:done"))
    await bus1.publish_inbound(_mobile_item("restart-fail", "fail", "client:fail"))

    def done_and_fail_terminal() -> bool:
        done_turn = manager1.control_store.find_turn_by_client_message_id(
            done_key, "client:done"
        )
        fail_turn = manager1.control_store.find_turn_by_client_message_id(
            fail_key, "client:fail"
        )
        return (
            done_turn is not None
            and done_turn.status is TurnStatus.COMPLETED
            and fail_turn is not None
            and fail_turn.status is TurnStatus.FAILED
        )

    await _wait_for(done_and_fail_terminal)

    # 进程1崩溃窗口：worker 停止（receipt 未送达），noturn/interrupt 只留 row。
    worker1.stop()
    await worker1_task

    # interrupted turn 用 executor 入口 Event 证明已进入执行后才中断，
    # 避免 queued 取消竞态把终态变成 CANCELLED。
    handle = await runtime1.start_turn(
        TurnRequest(
            interrupt_key,
            "interrupt",
            {"inboundMetadata": {"client_message_id": "client:interrupt"}},
        )
    )
    await asyncio.wait_for(interrupt_entered.wait(), timeout=2)
    assert (await handle.interrupt()).status is TurnStatus.INTERRUPTED

    # 四种 handoff 使用不同 session/lane：互不阻塞 receipt，也不在
    # aclose 之后再 publish（aclose 排空后不再接受新的出站项）。
    await bus1.publish_inbound(
        _mobile_item("restart-noturn", "noturn", "client:noturn")
    )
    await bus1.publish_inbound(
        _mobile_item("restart-interrupt", "interrupt", "client:interrupt")
    )
    assert len(manager1.control_store.list_inbound_handoffs()) == 4
    await bus1.aclose()
    await runtime1.shutdown()
    manager1.close()

    # 进程2：同 DB 的新 store+bus+runtime+worker 恢复。
    manager2 = SessionManager(workspace)
    assert (
        manager2.control_store.find_turn_by_client_message_id(
            done_key, "client:done"
        ).status
        is TurnStatus.COMPLETED
    )

    async def execute2(request: TurnRequest) -> str:
        return f"echo:{request.input}"

    runtime2 = ConversationRuntime(manager2.control_store, execute2)
    bus2, worker2 = _real_worker(manager2, runtime2)
    delivered2: list[OutboundMessage] = []

    async def on_outbound(msg: OutboundMessage) -> None:
        delivered2.append(msg)

    bus2.subscribe_outbound("mobile", on_outbound)
    await bus2.recover_durable_inbounds()
    await bus2.recover_durable_inbounds()
    assert bus2.inbound_size == 4
    assert len(bus2._inbound_accepted) == 4
    worker2_task = asyncio.create_task(worker2.run())
    dispatch2 = asyncio.create_task(bus2.dispatch_outbound())
    await _wait_for(lambda: manager2.control_store.list_inbound_handoffs() == [])

    done = manager2.control_store.find_turn_by_client_message_id(
        done_key, "client:done"
    )
    failed = manager2.control_store.find_turn_by_client_message_id(
        fail_key, "client:fail"
    )
    noturn = manager2.control_store.find_turn_by_client_message_id(
        noturn_key, "client:noturn"
    )
    interrupted = manager2.control_store.find_turn_by_client_message_id(
        interrupt_key, "client:interrupt"
    )
    assert done is not None and done.status is TurnStatus.COMPLETED
    assert failed is not None and failed.status is TurnStatus.FAILED
    # 1. no-turn 只创建一次；completed/failed 不重开 turn；interrupted 不重启 turn。
    assert noturn is not None and noturn.status is TurnStatus.COMPLETED
    assert interrupted is not None and interrupted.status is TurnStatus.INTERRUPTED
    assert (
        sum(
            len(manager2.control_store.list_turns(key, limit=200))
            for key in (done_key, fail_key, noturn_key, interrupt_key)
        )
        == 4
    )

    # 2. 四种终态都用权威 turn id 重投递；interrupted 使用 typed projection。
    contents = [msg.content for msg in delivered2]
    assert "echo:done" in contents
    assert "echo:noturn" in contents
    assert any(content.startswith("处理消息时出错") for content in contents)
    redelivered_failed = next(
        msg for msg in delivered2 if msg.content.startswith("处理消息时出错")
    )
    assert redelivered_failed.control_turn_id == failed.id
    echoed_done = next(msg for msg in delivered2 if msg.content == "echo:done")
    assert echoed_done.control_turn_id == done.id
    echoed_noturn = next(msg for msg in delivered2 if msg.content == "echo:noturn")
    assert echoed_noturn.control_turn_id == noturn.id
    interrupted_outbound = next(
        msg for msg in delivered2 if msg.control_turn_id == interrupted.id
    )
    assert interrupted_outbound.terminal_status is TurnTerminalStatus.INTERRUPTED
    assert interrupted_outbound.metadata["client_message_id"] == "client:interrupt"
    assert bus2._inbound_accepted == {}
    await _cancel_task(worker2_task)
    await _cancel_task(dispatch2)
    await runtime2.shutdown()
    manager2.close()


@pytest.mark.asyncio
async def test_failed_outbound_carries_verified_client_message_id(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:fcmid"
    manager.save(manager.get_or_create(session_key))

    async def execute(_request: TurnRequest) -> str:
        raise RuntimeError("boom")

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def on_outbound(msg: OutboundMessage) -> None:
        delivered.append(msg)

    bus.subscribe_outbound("mobile", on_outbound)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    inbound = _mobile_item("fcmid", "hello", "client:fcmid")
    await bus.publish_inbound(inbound)
    consumed = await _consume_message(bus)
    await worker._run_message(consumed)

    # 1. FAILED 终态从已验证 userMessage item 贯通 client_message_id。
    turn = manager.control_store.find_turn_by_client_message_id(
        session_key, "client:fcmid"
    )
    assert turn is not None and turn.status is TurnStatus.FAILED
    failed = next(msg for msg in delivered if msg.content.startswith("处理消息时出错"))
    assert failed.metadata["client_message_id"] == "client:fcmid"
    assert failed.control_turn_id == turn.id
    assert manager.control_store.list_inbound_handoffs() == []
    await _cancel_task(dispatch)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_restart_redelivery_failed_carries_verified_client_message_id(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    session_key = "mobile:rdfail"
    manager1 = SessionManager(workspace)
    manager1.save(manager1.get_or_create(session_key))

    async def execute(_request: TurnRequest) -> str:
        raise RuntimeError("boom")

    # 进程1：FAILED turn 已落库，但无 subscriber 送达，handoff 保留（崩溃窗口）。
    runtime1 = ConversationRuntime(manager1.control_store, execute)
    bus1, worker1 = _real_worker(manager1, runtime1)
    dispatch1 = asyncio.create_task(bus1.dispatch_outbound())
    inbound1 = _mobile_item("rdfail", "hello", "client:rdfail")
    await bus1.publish_inbound(inbound1)
    consumed1 = await _consume_message(bus1)
    with pytest.raises(RuntimeError, match="handoff retained"):
        await worker1._run_message(consumed1)
    assert len(manager1.control_store.list_inbound_handoffs()) == 1
    await _cancel_task(dispatch1)
    await bus1.aclose()
    await runtime1.shutdown()
    manager1.close()

    # 进程2：同 DB 恢复，重投递 FAILED 终态并贯通已验证 client_message_id。
    manager2 = SessionManager(workspace)
    runtime2 = ConversationRuntime(manager2.control_store, execute)
    bus2, worker2 = _real_worker(manager2, runtime2)
    delivered2: list[OutboundMessage] = []

    async def on_outbound(msg: OutboundMessage) -> None:
        delivered2.append(msg)

    bus2.subscribe_outbound("mobile", on_outbound)
    dispatch2 = asyncio.create_task(bus2.dispatch_outbound())
    await bus2.recover_durable_inbounds()
    recovered = await _consume_message(bus2)
    await worker2._run_message(recovered)

    turn = manager2.control_store.find_turn_by_client_message_id(
        session_key, "client:rdfail"
    )
    assert turn is not None and turn.status is TurnStatus.FAILED
    failed = next(msg for msg in delivered2 if msg.content.startswith("处理消息时出错"))
    assert failed.metadata["client_message_id"] == "client:rdfail"
    assert failed.control_turn_id == turn.id
    assert manager2.control_store.list_inbound_handoffs() == []
    await _cancel_task(dispatch2)
    await runtime2.shutdown()
    manager2.close()


@pytest.mark.asyncio
async def test_worker_terminal_error_milestone_carries_result_identity(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:emid"
    manager.save(manager.get_or_create(session_key))

    async def execute(request: TurnRequest) -> str:
        return f"echo:{request.input}"

    async def on_outbound(_msg: OutboundMessage) -> None:
        raise RuntimeError("channel down")

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    monkeypatch.setattr("bus.queue._OUTBOUND_RETRY_DELAY", 0.0)
    monkeypatch.setattr(
        "bootstrap.passive_worker._TERMINAL_DELIVERY_RETRY_DELAYS",
        (0.0, 0.0),
    )
    bus.subscribe_outbound("mobile", on_outbound)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    inbound = _mobile_item("emid", "hello", "client:emid")
    await bus.publish_inbound(inbound)
    consumed = await _consume_message(bus)
    with caplog.at_level(logging.INFO, logger="bootstrap.passive_worker"):
        with pytest.raises(RuntimeError, match="handoff retained"):
            await worker._run_message(consumed)

    # 1. result 已取得：error 里程碑保留 result.id 与已验证 client_message_id。
    turn = manager.control_store.find_turn_by_client_message_id(
        session_key, "client:emid"
    )
    assert turn is not None and turn.status is TurnStatus.COMPLETED
    error_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event")
        == "tl:worker.terminal.error"
    ]
    assert len(error_records) == 1
    assert error_records[0].akashic_fields["turn_id"] == turn.id
    assert error_records[0].akashic_fields["client_message_id"] == "client:emid"
    assert error_records[0].akashic_fields["outcome"] == "error"
    assert error_records[0].akashic_fields["duration_ms"] is not None
    assert len(manager.control_store.list_inbound_handoffs()) == 1
    await _cancel_task(dispatch)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_worker_terminal_cleanup_failure_emits_only_error_terminal(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """durable terminal 已送达但 handoff 首次删除失败时，span 只以 error 收口。"""

    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:cleanup-fail-once"
    manager.save(manager.get_or_create(session_key))

    async def execute(request: TurnRequest) -> str:
        return f"echo:{request.input}"

    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def commit_mobile_terminal(message: OutboundMessage) -> None:
        delivered.append(message)

    original_complete = SessionStore.complete_inbound_handoff
    cleanup_attempts = 0

    def fail_once(store: SessionStore, handoff_id: str) -> None:
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        if cleanup_attempts == 1:
            raise OSError("simulated handoff fsync failure")
        original_complete(store, handoff_id)

    monkeypatch.setattr(SessionStore, "complete_inbound_handoff", fail_once)
    monkeypatch.setattr("bus.queue._INBOUND_CLEANUP_RETRY_INITIAL_DELAY", 0.0)
    bus.subscribe_outbound("mobile", commit_mobile_terminal)
    dispatcher = asyncio.create_task(bus.dispatch_outbound())
    inbound = _mobile_item(
        "cleanup-fail-once",
        "hello",
        "client:cleanup-fail-once",
    )
    await bus.publish_inbound(inbound)
    consumed = await _consume_message(bus)
    with caplog.at_level(logging.INFO, logger="bootstrap.passive_worker"):
        with pytest.raises(OSError, match="simulated handoff fsync failure"):
            await worker._run_message(consumed)

    # 1. 后台 cleanup-only retry 最终删除 row，但本次 span 不得先 done 再 error。
    await _wait_for(lambda: manager.control_store.list_inbound_handoffs() == [])
    turn = manager.control_store.find_turn_by_client_message_id(
        session_key,
        "client:cleanup-fail-once",
    )
    assert turn is not None and turn.status is TurnStatus.COMPLETED
    assert cleanup_attempts == 2
    assert len(delivered) == 1
    terminal_events = [
        record.akashic_fields["event"]
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("turn_id") == turn.id
        and str(record.akashic_fields.get("event", "")).startswith(
            "tl:worker.terminal."
        )
    ]
    assert terminal_events == [
        "tl:worker.terminal.start",
        "tl:worker.terminal.error",
    ]
    await _cancel_task(dispatcher)
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_worker_run_retries_terminal_and_excludes_executor_from_duration(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 worker/bus/runtime/store 链在临时送达失败后重投同一 terminal。"""

    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:retry-chain"
    manager.save(manager.get_or_create(session_key))
    executor_delay = 0.2

    async def execute(request: TurnRequest) -> str:
        await asyncio.sleep(executor_delay)
        return f"echo:{request.input}"

    monkeypatch.setattr("bus.queue._OUTBOUND_RETRY_DELAY", 0.0)
    monkeypatch.setattr(
        "bootstrap.passive_worker._TERMINAL_DELIVERY_RETRY_DELAYS",
        (0.0, 0.0),
    )
    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    attempts = 0
    delivered: list[OutboundMessage] = []

    async def commit_mobile_terminal(message: OutboundMessage) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise RuntimeError("temporary mobile inbox failure")
        delivered.append(message)

    _ = bus.subscribe_outbound("mobile", commit_mobile_terminal)
    dispatcher = asyncio.create_task(bus.dispatch_outbound())
    worker_task = asyncio.create_task(worker.run())
    with caplog.at_level(logging.INFO):
        await bus.publish_inbound(
            _mobile_item("retry-chain", "hello", "client:retry-chain")
        )
        await _wait_for(
            lambda: manager.control_store.list_inbound_handoffs() == [],
        )

    # 1. 首个 worker attempt 内两次渠道发送都失败，第二个 attempt 成功。
    assert attempts == 3
    assert len(delivered) == 1
    turn = manager.control_store.find_turn_by_client_message_id(
        session_key,
        "client:retry-chain",
    )
    assert turn is not None and turn.status is TurnStatus.COMPLETED
    assert delivered[0].control_turn_id == turn.id
    assert delivered[0].terminal_status is TurnTerminalStatus.COMPLETED
    assert worker._result_tasks == set()

    # 2. terminal duration 从 handle.result 后开始，不包含 executor 的 200ms。
    done_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event")
        == "tl:worker.terminal.done"
        and record.akashic_fields.get("turn_id") == turn.id
    ]
    assert len(done_records) == 1
    duration_ms = done_records[0].akashic_fields["duration_ms"]
    assert isinstance(duration_ms, float)
    assert duration_ms < executor_delay * 1_000
    assert not any(
        "Task exception was never retrieved" in record.getMessage()
        for record in caplog.records
    )

    worker.stop()
    await worker_task
    bus.stop()
    await dispatcher
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_worker_lane_retries_terminal_after_bounded_attempts_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一次有界投递耗尽后 lane 继续重投同一 terminal，不重跑 Provider。"""

    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:retry-after-exhaustion"
    manager.save(manager.get_or_create(session_key))
    executor_calls = 0

    async def execute(request: TurnRequest) -> str:
        nonlocal executor_calls
        executor_calls += 1
        return f"echo:{request.input}"

    monkeypatch.setattr("bus.queue._OUTBOUND_RETRY_DELAY", 0.0)
    monkeypatch.setattr(
        "bootstrap.passive_worker._TERMINAL_DELIVERY_RETRY_DELAYS",
        (0.0, 0.0),
    )
    monkeypatch.setattr(
        "bootstrap.passive_worker._TERMINAL_LANE_RETRY_DELAY",
        0.0,
    )
    runtime = ConversationRuntime(manager.control_store, execute)
    bus, worker = _real_worker(manager, runtime)
    physical_attempts = 0
    delivered: list[OutboundMessage] = []

    async def recover_after_first_cycle(message: OutboundMessage) -> None:
        nonlocal physical_attempts
        physical_attempts += 1
        if physical_attempts <= 6:
            raise OSError("mobile inbox temporarily unavailable")
        delivered.append(message)

    bus.subscribe_outbound("mobile", recover_after_first_cycle)
    dispatcher = asyncio.create_task(bus.dispatch_outbound())
    worker_task = asyncio.create_task(worker.run())
    await bus.publish_inbound(
        _mobile_item(
            "retry-after-exhaustion",
            "hello",
            "client:retry-after-exhaustion",
        )
    )
    await _wait_for(lambda: manager.control_store.list_inbound_handoffs() == [])

    # 1. 首轮 3×MessageBus(每次2次物理发送)耗尽；lane 第二轮第1次成功。
    assert physical_attempts == 7
    assert executor_calls == 1
    assert len(delivered) == 1
    turn = manager.control_store.find_turn_by_client_message_id(
        session_key,
        "client:retry-after-exhaustion",
    )
    assert turn is not None and turn.status is TurnStatus.COMPLETED
    assert delivered[0].control_turn_id == turn.id
    assert bus._inbound_accepted == {}
    assert worker_task.done() is False

    worker.stop()
    await worker_task
    bus.stop()
    await dispatcher
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_never_fit_input_persists_failed_terminal_before_handoff_ack(
    tmp_path: Path,
) -> None:
    """永久超限输入由 runtime 建立 failed turn，并经 Mobile barrier 收口。"""

    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:never-fit"
    manager.save(manager.get_or_create(session_key))
    executed: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> str:
        executed.append(request)
        return f"echo:{request.input}"

    runtime = ConversationRuntime(
        manager.control_store,
        execute,
        max_active_bytes=1_024,
    )
    bus, worker = _real_worker(manager, runtime)
    delivered: list[OutboundMessage] = []

    async def commit_mobile_terminal(message: OutboundMessage) -> None:
        delivered.append(message)

    _ = bus.subscribe_outbound("mobile", commit_mobile_terminal)
    dispatcher = asyncio.create_task(bus.dispatch_outbound())
    worker_task = asyncio.create_task(worker.run())
    await bus.publish_inbound(
        _mobile_item("never-fit", "x" * 2_000, "client:never-fit")
    )
    await _wait_for(lambda: manager.control_store.list_inbound_handoffs() == [])

    # 1. Provider 未执行；SessionStore 拥有真实 failed turn 与稳定 client identity。
    assert executed == []
    turn = manager.control_store.find_turn_by_client_message_id(
        session_key,
        "client:never-fit",
    )
    assert turn is not None and turn.status is TurnStatus.FAILED
    assert turn.error is not None and turn.error.type == "resource-exhausted"
    assert len(delivered) == 1
    assert delivered[0].control_turn_id == turn.id
    assert delivered[0].terminal_status is TurnTerminalStatus.FAILED
    assert delivered[0].metadata["client_message_id"] == "client:never-fit"

    # 2. 永久拒绝关闭该 logical interaction；后续小输入不携带被拒绝正文续接。
    await bus.publish_inbound(_mobile_item("never-fit", "ok", "client:after-never-fit"))
    await _wait_for(lambda: manager.control_store.list_inbound_handoffs() == [])
    assert [request.input for request in executed] == ["ok"]
    turns = manager.control_store.list_turns(session_key, limit=10)
    assert len(turns) == 2
    followup = turns[0]
    assert followup.status is TurnStatus.COMPLETED
    assert "continuedFromTurnId" not in followup.metadata
    assert followup.metadata["priorInputCount"] == 0
    assert delivered[-1].control_turn_id == followup.id

    worker.stop()
    await worker_task
    bus.stop()
    await dispatcher
    await runtime.shutdown()
    manager.close()

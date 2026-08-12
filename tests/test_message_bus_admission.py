from __future__ import annotations

import asyncio
import gc
import logging
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.control.models import TurnRequest, TurnStatus
from agent.control.runtime import ConversationRuntime
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from session.manager import SessionManager
from session.store import SessionStore


@pytest.mark.asyncio
async def test_worker_error_before_turn_owner_keeps_handoff_and_releases_admission(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:cleanup"
    manager.save(manager.get_or_create(session_key))
    _, admission_id = manager.admit_existing(session_key)
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        "mobile",
        "device:1",
        "cleanup",
        "hello",
        metadata={"session_key_override": session_key, "client_message_id": "client-1"},
        session_admission_id=admission_id,
    )
    await bus.publish_inbound(item)
    consumed = await bus.consume_inbound()
    assert consumed is item
    item_ref = weakref.ref(item)

    class _Runtime:
        async def wait_thread_available(self, _session_key: str) -> None:
            return None

        async def start_turn(self, _request: object) -> object:
            raise RuntimeError("worker stopped before turn submission")

    from bootstrap.passive_worker import PassiveMessageWorker

    worker = PassiveMessageWorker(
        bus,
        _Runtime(),  # type: ignore[arg-type]
        SimpleNamespace(session_manager=manager),  # type: ignore[arg-type]
    )
    lane = asyncio.Queue()
    lane.put_nowait(consumed)
    worker._lane_queues[session_key] = lane
    lane_task = asyncio.create_task(worker._run_lane(session_key, lane))
    await lane_task

    # 1. start_turn 建立 turn owner 前失败：不 complete_inbound，row 与 owner 保留。
    assert manager.control_store.list_turns(session_key) == []
    assert len(store.list_inbound_handoffs()) == 1
    owner_key = id(item)
    assert owner_key in bus._inbound_accepted
    del consumed
    del item
    gc.collect()
    assert item_ref() is not None

    # 2. session admission 恰一次释放，同一会话可再次取得。
    assert (
        store._conn.execute(
            "SELECT 1 FROM session_admissions WHERE admission_id = ?",
            (admission_id,),
        ).fetchone()
        is None
    )
    _, reacquired = manager.admit_existing(session_key)
    manager.release_admission(reacquired)

    # 3. 没有删除授权：不启动 cleanup retry，row 继续由 durable owner 持有。
    assert bus._inbound_cleanup_tasks == {}
    await bus.aclose()
    assert len(store.list_inbound_handoffs()) == 1
    assert owner_key in bus._inbound_accepted
    manager.close()
    store.close()


@pytest.mark.asyncio
async def test_persistent_cleanup_failure_is_bounded_and_shutdown_cancels_retry(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        "mobile",
        "device:1",
        "persistent",
        "hello",
        metadata={"client_message_id": "client-persistent"},
    )
    await bus.publish_inbound(item)
    consumed = await bus.consume_inbound()
    attempts = 0

    def always_fail(_handoff_id: str) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("persistent delete failure")

    store.complete_inbound_handoff = always_fail  # type: ignore[method-assign]
    with pytest.raises(OSError, match="persistent delete failure"):
        await bus.complete_inbound(consumed)
    await asyncio.sleep(0.35)
    assert 2 <= attempts <= 4
    assert len(bus._inbound_accepted) == 1
    assert len(bus._inbound_cleanup_tasks) == 1
    await bus.aclose()
    assert bus._inbound_cleanup_tasks == {}
    assert (
        store._conn.execute(
            "SELECT 1 FROM inbound_handoffs WHERE handoff_id = ?",
            (consumed.handoff_id,),
        ).fetchone()
        is not None
    )
    store.close()


@pytest.mark.asyncio
async def test_cleanup_finalize_failure_is_fatal_and_observable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        "mobile",
        "device:1",
        "fatal",
        "hello",
        metadata={"client_message_id": "client-fatal"},
    )
    await bus.publish_inbound(item)
    consumed = await bus.consume_inbound()
    original_complete = store.complete_inbound_handoff
    failed = True

    def fail_once(handoff_id: str) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise OSError("temporary delete failure")
        original_complete(handoff_id)

    store.complete_inbound_handoff = fail_once  # type: ignore[method-assign]

    async def fatal_finalize(_owner_key: int, _owner: object) -> None:
        raise RuntimeError("owner mismatch")

    bus._finalize_inbound_owner = fatal_finalize  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError, match="temporary delete failure"):
            await bus.complete_inbound(consumed)
        await asyncio.sleep(0.2)
    assert bus._inbound_cleanup_error is not None
    assert bus._inbound_cleanup_tasks == {}
    assert "event=runtime_fatal" in caplog.text
    assert "owner=message_bus.inbound_cleanup" in caplog.text
    with pytest.raises(RuntimeError, match="cleanup owner failed"):
        await bus.aclose()
    store.close()


@pytest.mark.asyncio
async def test_awaited_outbound_receipt_true_only_after_subscriber_commit() -> None:
    bus = MessageBus()
    committed: list[str] = []

    async def callback(msg: OutboundMessage) -> None:
        committed.append(msg.content)

    bus.subscribe_outbound("mobile", callback)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    delivered = await asyncio.wait_for(
        bus.publish_outbound_awaited(
            OutboundMessage("mobile", "one", "final", control_turn_id="turn:1")
        ),
        timeout=1,
    )
    assert delivered is True
    assert committed == ["final"]
    bus.stop()
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert bus._pending_outbound_receipts == set()


@pytest.mark.asyncio
async def test_awaited_outbound_receipt_false_without_subscriber() -> None:
    bus = MessageBus()
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    delivered = await asyncio.wait_for(
        bus.publish_outbound_awaited(OutboundMessage("mobile", "one", "final")),
        timeout=1,
    )
    assert delivered is False
    bus.stop()
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert bus._pending_outbound_receipts == set()


@pytest.mark.asyncio
async def test_awaited_outbound_receipt_false_after_two_callback_failures() -> None:
    bus = MessageBus()
    attempts = {"count": 0}

    async def callback(_msg: OutboundMessage) -> None:
        attempts["count"] += 1
        raise RuntimeError("channel unavailable")

    bus.subscribe_outbound("mobile", callback)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    delivered = await asyncio.wait_for(
        bus.publish_outbound_awaited(OutboundMessage("mobile", "one", "final")),
        timeout=5,
    )
    assert delivered is False
    assert attempts["count"] >= 2
    bus.stop()
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch


@pytest.mark.asyncio
async def test_awaited_outbound_two_failures_never_sends_fallback() -> None:
    bus = MessageBus()
    attempts: list[str] = []

    async def callback(msg: OutboundMessage) -> None:
        attempts.append(msg.content)
        raise RuntimeError("channel down")

    bus.subscribe_outbound("mobile", callback)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    delivered = await asyncio.wait_for(
        bus.publish_outbound_awaited(OutboundMessage("mobile", "one", "final")),
        timeout=5,
    )
    assert delivered is False
    # awaited 封套：恰 2 次原始 terminal 内容，严禁降级文案占用终态。
    assert attempts == ["final", "final"]
    bus.stop()
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert bus._pending_outbound_receipts == set()


@pytest.mark.asyncio
async def test_fire_and_forget_two_failures_still_sends_fallback() -> None:
    bus = MessageBus()
    attempts: list[str] = []

    async def callback(msg: OutboundMessage) -> None:
        attempts.append(msg.content)
        raise RuntimeError("channel down")

    bus.subscribe_outbound("mobile", callback)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    await bus.publish_outbound(OutboundMessage("mobile", "one", "final"))

    async def reached_three() -> None:
        while len(attempts) < 3:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(reached_three(), timeout=5)
    bus.stop()
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    # fire-and-forget 保持既有 fallback：两次失败后发送降级文案。
    assert attempts == ["final", "final", "（消息发送失败，请稍后重试）"]


@pytest.mark.asyncio
async def test_aclose_drains_queued_outbound_and_releases_chat_lane_state() -> None:
    bus = MessageBus()
    delivered: list[str] = []

    async def callback(msg: OutboundMessage) -> None:
        delivered.append(msg.content)

    bus.subscribe_outbound("mobile", callback)
    pending_a = asyncio.create_task(
        bus.publish_outbound_awaited(OutboundMessage("mobile", "one", "a"))
    )
    pending_b = asyncio.create_task(
        bus.publish_outbound_awaited(OutboundMessage("mobile", "two", "b"))
    )
    await bus.publish_outbound(OutboundMessage("mobile", "two", "c"))
    await asyncio.sleep(0)
    assert bus.outbound_size == 3
    assert len(bus._pending_outbound_receipts) == 2

    # 1. aclose 排空未 dispatch 项：receipt 收束为 False、lane pending 回滚。
    await bus.aclose()
    assert bus.outbound_size == 0
    assert bus._pending_outbound_receipts == set()
    assert bus.chat_lane._states == {}
    assert (await pending_a) is False
    assert (await pending_b) is False
    assert delivered == []


@pytest.mark.asyncio
async def test_live_reserve_and_recovery_race_never_duplicates_handoff(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(tmp_path / "workspace")
    session_key = "mobile:race"
    manager.save(manager.get_or_create(session_key))

    async def execute(request: TurnRequest) -> str:
        return f"echo:{request.input}"

    runtime = ConversationRuntime(manager.control_store, execute)
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    delivered: list[OutboundMessage] = []

    async def on_outbound(msg: OutboundMessage) -> None:
        delivered.append(msg)

    bus.subscribe_outbound("mobile", on_outbound)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    item = InboundMessage(
        "mobile",
        "device:1",
        "race",
        "hello",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client:race",
        },
    )
    # 门控：live publish 在 durable reserve 落库后、accepted owner 登记前暂停。
    original_pending = bus._chat_lane.mark_passive_pending
    reserve_committed = asyncio.Event()
    resume_live = asyncio.Event()

    async def gated_pending(channel: str, chat_id: str) -> None:
        reserve_committed.set()
        await resume_live.wait()
        await original_pending(channel, chat_id)

    bus._chat_lane.mark_passive_pending = gated_pending  # type: ignore[method-assign]
    live = asyncio.create_task(bus.publish_inbound(item))
    await asyncio.wait_for(reserve_committed.wait(), timeout=2)

    # 1. 同窗口启动恢复：必须等待 durable lock，不能看到半登记的 live row。
    recovery = asyncio.create_task(bus.recover_durable_inbounds())
    await asyncio.sleep(0.05)
    assert not recovery.done()
    assert len(store.list_inbound_handoffs()) == 1

    resume_live.set()
    await asyncio.wait_for(live, timeout=2)
    await asyncio.wait_for(recovery, timeout=2)

    # 2. 同一 handoff 只有一 queue item / 一 accepted owner。
    assert bus.inbound_size == 1
    assert len(bus._inbound_accepted) == 1

    # 3. 处理一次只建一个 turn，无重复 client_message_id 冲突。
    from bootstrap.passive_worker import PassiveMessageWorker

    worker = PassiveMessageWorker(
        bus,
        runtime,
        cast(Any, SimpleNamespace(session_manager=manager)),
    )
    consumed = await bus.consume_inbound()
    assert isinstance(consumed, InboundMessage)
    await worker._run_message(consumed)
    turns = manager.control_store.list_turns(session_key)
    assert len(turns) == 1
    assert turns[0].status is TurnStatus.COMPLETED
    assert manager.control_store.list_inbound_handoffs() == []
    assert [msg.content for msg in delivered] == ["echo:hello"]
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    await runtime.shutdown()
    manager.close()
    store.close()


@pytest.mark.asyncio
async def test_awaited_outbound_receipt_settled_false_on_aclose_without_leak() -> None:
    bus = MessageBus()
    pending = asyncio.create_task(
        bus.publish_outbound_awaited(OutboundMessage("mobile", "one", "final"))
    )
    await asyncio.sleep(0)
    assert len(bus._pending_outbound_receipts) == 1
    await bus.aclose()
    assert (await pending) is False
    assert bus._pending_outbound_receipts == set()


@pytest.mark.asyncio
async def test_awaited_outbound_receipt_settled_false_on_dispatch_cancel() -> None:
    bus = MessageBus()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def callback(_msg: OutboundMessage) -> None:
        entered.set()
        await release.wait()

    bus.subscribe_outbound("mobile", callback)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    pending = asyncio.create_task(
        bus.publish_outbound_awaited(OutboundMessage("mobile", "one", "final"))
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert (await pending) is False
    assert bus._pending_outbound_receipts == set()
    release.set()

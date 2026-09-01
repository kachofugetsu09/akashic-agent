from __future__ import annotations

import asyncio
import gc
import logging
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.control.models import TurnRequest, TurnStatus
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
    ChannelInboundMessage,
    ChannelDeliveryReceipt,
    DeliveryStatus,
    InboundEnvelope,
    InboundOwner,
    InboundState,
    JsonValue,
    OutboundEnvelope,
    RawInbound,
)
from bus.events import InboundMessage
from bus.queue import MessageBus
from session.manager import SessionManager
from session.store import SessionAdmissionConflictError, SessionStore


def _v3_outbound() -> tuple[OutboundEnvelope, _OutboundBinding]:
    envelope = OutboundEnvelope(
        logical_delivery_id="delivery-1",
        delivery_id="delivery-1",
        attempt_sequence=1,
        snapshot_id="snapshot-1",
        generation_id="generation-1",
        binding_token="binding-1",
        channel="feishu",
        recipient="chat-1",
        body="hello",
        metadata={"kind": "final"},
    )
    binding = _OutboundBinding()
    return envelope, binding


class _InboundLease:
    def __init__(
        self,
        close_gate: asyncio.Event | None = None,
        *,
        channel: str = "feishu",
    ) -> None:
        self.snapshot_id = "snapshot-1"
        self.generation_id = "generation-1"
        self.binding_token = "binding-1"
        self.channel_name = channel
        self.snapshot_lease = SimpleNamespace(
            active=True,
            snapshot=SimpleNamespace(snapshot_id=self.snapshot_id),
            validation_candidate_plugin_ids=frozenset(),
        )
        self.closed = 0
        self.closed_event = asyncio.Event()
        self.close_gate = close_gate
        self.close_started = asyncio.Event()

    @property
    def active(self) -> bool:
        return self.closed == 0

    async def aclose(self) -> None:
        self.close_started.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.closed += 1
        self.closed_event.set()


def _v3_inbound(
    close_gate: asyncio.Event | None = None,
    *,
    message_id: str = "message-1",
    attachments: tuple[AttachmentRef, ...] = (),
    metadata: dict[str, JsonValue] | None = None,
    channel: str = "feishu",
) -> tuple[InboundEnvelope, _InboundLease]:
    lease = _InboundLease(close_gate, channel=channel)
    envelope = InboundEnvelope(
        message_id=message_id,
        snapshot_id=lease.snapshot_id,
        generation_id=lease.generation_id,
        binding_token=lease.binding_token,
        message=ChannelInboundMessage(
            channel=channel,
            sender="user-1",
            chat_id="chat-1",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
            attachments=attachments,
        ),
        lease=lease,
    )
    return envelope, lease


@dataclass(frozen=True)
class _OutboundBinding:
    snapshot_id: str = "snapshot-1"
    generation_id: str = "generation-1"
    binding_token: str = "binding-1"
    channel_name: str = "feishu"
    active: bool = True


@pytest.mark.asyncio
async def test_worker_error_before_turn_owner_keeps_handoff_and_releases_admission(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:cleanup"
    manager.save(manager.get_or_create(session_key))
    _, admission_id = manager.admit_existing(session_key)
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        "akashic",
        "device:1",
        "cleanup",
        "hello",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-1",
            "mobile_v3_handoff": True,
        },
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
async def test_mobile_attachment_acquire_failure_releases_session_admission(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:missing-attachment"
    manager.save(manager.get_or_create(session_key))
    _, admission_id = manager.admit_existing(session_key)
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        "akashic",
        "device:1",
        "missing-attachment",
        "hello",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-missing-attachment",
            "attachment_ids": ["artifact-missing"],
            "mobile_v3_handoff": True,
        },
        session_admission_id=admission_id,
    )
    await bus.publish_inbound(item)
    consumed = await bus.consume_inbound()

    class _Runtime:
        async def start_turn(self, _request: object) -> object:
            raise AssertionError("attachment acquisition failure must precede turn start")

    from bootstrap.passive_worker import PassiveMessageWorker

    worker = PassiveMessageWorker(
        bus,
        _Runtime(),  # type: ignore[arg-type]
        SimpleNamespace(session_manager=manager),  # type: ignore[arg-type]
    )
    lane = asyncio.Queue()
    lane.put_nowait(consumed)
    worker._lane_queues[session_key] = lane
    await worker._run_lane(session_key, lane)

    assert len(store.list_inbound_handoffs()) == 1
    assert (
        store._conn.execute(
            "SELECT 1 FROM session_admissions WHERE admission_id = ?",
            (admission_id,),
        ).fetchone()
        is None
    )
    assert item.session_admission_id is None
    await bus.aclose()
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
        "akashic",
        "device:1",
        "persistent",
        "hello",
        metadata={
            "client_message_id": "client-persistent",
            "mobile_v3_handoff": True,
        },
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
        "akashic",
        "device:1",
        "fatal",
        "hello",
        metadata={
            "client_message_id": "client-fatal",
            "mobile_v3_handoff": True,
        },
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
async def test_removed_legacy_outbound_paths_fail_loud() -> None:
    bus = MessageBus()

    with pytest.raises(RuntimeError, match="legacy publish_outbound 已删除"):
        await bus.publish_outbound(object())
    with pytest.raises(RuntimeError, match="legacy publish_outbound_awaited 已删除"):
        await bus.publish_outbound_awaited(object())
    assert bus.outbound_size == 0


@pytest.mark.asyncio
async def test_v3_channel_outbound_returns_exact_provider_receipt_once() -> None:
    bus = MessageBus()
    envelope, binding = _v3_outbound()
    calls = 0

    async def deliver(
        received: OutboundEnvelope,
        owner: Any,
    ) -> ChannelDeliveryReceipt:
        nonlocal calls
        calls += 1
        assert received is envelope
        assert owner is binding
        return ChannelDeliveryReceipt(
            delivery_id=received.delivery_id,
            status=DeliveryStatus.DELIVERED,
            provider_ids=("provider-1",),
        )

    bus.bind_channel_outbound_dispatcher(deliver)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    receipt = await bus.publish_channel_outbound_awaited(envelope, binding)
    assert receipt.status is DeliveryStatus.DELIVERED
    assert receipt.provider_ids == ("provider-1",)
    assert calls == 1
    bus.stop()
    await dispatch


@pytest.mark.asyncio
async def test_v3_channel_pre_provider_commit_fences_adapter_effect() -> None:
    bus = MessageBus()
    envelope, binding = _v3_outbound()
    provider_calls = 0

    async def deliver(
        _received: OutboundEnvelope,
        _owner: Any,
    ) -> ChannelDeliveryReceipt:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("failed pre-provider commit must fence adapter effect")

    def reject_commit() -> None:
        raise RuntimeError("ledger commit failed")

    bus.bind_channel_outbound_dispatcher(deliver)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    receipt = await bus.publish_channel_outbound_awaited(
        envelope,
        binding,
        passive=False,
        before_provider=reject_commit,
    )

    assert receipt.status is DeliveryStatus.REJECTED
    assert receipt.error == "ledger commit failed"
    assert provider_calls == 0
    bus.stop()
    await dispatch


@pytest.mark.asyncio
async def test_v3_direct_channel_outbound_waits_for_passive_turn() -> None:
    bus = MessageBus()
    envelope, binding = _v3_outbound()
    entered = asyncio.Event()

    async def deliver(
        received: OutboundEnvelope,
        _owner: Any,
    ) -> ChannelDeliveryReceipt:
        entered.set()
        return ChannelDeliveryReceipt(
            received.delivery_id,
            DeliveryStatus.DELIVERED,
        )

    bus.bind_channel_outbound_dispatcher(deliver)
    await bus.chat_lane.mark_passive_pending(
        envelope.channel,
        envelope.recipient,
    )
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    pending = asyncio.create_task(
        bus.publish_channel_outbound_awaited(
            envelope,
            binding,
            passive=False,
        )
    )
    await asyncio.sleep(0)
    assert not entered.is_set()
    assert not pending.done()

    await bus.chat_lane.mark_passive_done(
        envelope.channel,
        envelope.recipient,
    )
    receipt = await asyncio.wait_for(pending, timeout=1)
    assert receipt.status is DeliveryStatus.DELIVERED
    assert entered.is_set()
    bus.stop()
    await dispatch


@pytest.mark.asyncio
async def test_v3_direct_channel_wait_is_rejected_by_terminal_bus_close() -> None:
    bus = MessageBus()
    envelope, binding = _v3_outbound()
    calls = 0

    async def deliver(
        _received: OutboundEnvelope,
        _owner: Any,
    ) -> ChannelDeliveryReceipt:
        nonlocal calls
        calls += 1
        raise AssertionError("关闭前仍在 lane 等待的 direct push 不得调用 provider")

    bus.bind_channel_outbound_dispatcher(deliver)
    await bus.chat_lane.mark_passive_pending(
        envelope.channel,
        envelope.recipient,
    )
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    pending = asyncio.create_task(
        bus.publish_channel_outbound_awaited(
            envelope,
            binding,
            passive=False,
        )
    )
    while bus.outbound_size:
        await asyncio.sleep(0)

    await asyncio.wait_for(bus.aclose(), timeout=1)
    receipt = await asyncio.wait_for(pending, timeout=1)

    assert receipt.status is DeliveryStatus.REJECTED
    assert calls == 0
    assert dispatch.done()
    await bus.chat_lane.mark_passive_done(
        envelope.channel,
        envelope.recipient,
    )


@pytest.mark.asyncio
async def test_v3_passive_channel_outbound_does_not_wait_for_its_own_turn() -> None:
    bus = MessageBus()
    envelope, binding = _v3_outbound()
    calls = 0

    async def deliver(
        received: OutboundEnvelope,
        _owner: Any,
    ) -> ChannelDeliveryReceipt:
        nonlocal calls
        calls += 1
        return ChannelDeliveryReceipt(
            received.delivery_id,
            DeliveryStatus.DELIVERED,
        )

    bus.bind_channel_outbound_dispatcher(deliver)
    await bus.chat_lane.mark_passive_pending(
        envelope.channel,
        envelope.recipient,
    )
    dispatch = asyncio.create_task(bus.dispatch_outbound())

    receipt = await asyncio.wait_for(
        bus.publish_channel_outbound_awaited(
            envelope,
            binding,
            passive=True,
        ),
        timeout=1,
    )

    assert receipt.status is DeliveryStatus.DELIVERED
    assert calls == 1
    await bus.chat_lane.mark_passive_done(
        envelope.channel,
        envelope.recipient,
    )
    bus.stop()
    await dispatch


@pytest.mark.asyncio
async def test_v3_channel_outbound_exception_is_unknown_without_retry() -> None:
    bus = MessageBus()
    envelope, binding = _v3_outbound()
    calls = 0

    async def fail_after_effect(
        _received: OutboundEnvelope,
        _owner: Any,
    ) -> ChannelDeliveryReceipt:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider receipt lost")

    bus.bind_channel_outbound_dispatcher(fail_after_effect)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    receipt = await bus.publish_channel_outbound_awaited(envelope, binding)
    assert receipt.status is DeliveryStatus.UNKNOWN
    assert receipt.error == "provider receipt lost"
    assert calls == 1
    bus.stop()
    await dispatch


@pytest.mark.asyncio
async def test_v3_channel_outbound_cancel_waits_for_provider_settlement() -> None:
    bus = MessageBus()
    envelope, binding = _v3_outbound()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def blocked_delivery(
        received: OutboundEnvelope,
        _owner: Any,
    ) -> ChannelDeliveryReceipt:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return ChannelDeliveryReceipt(
            received.delivery_id,
            DeliveryStatus.DELIVERED,
        )

    bus.bind_channel_outbound_dispatcher(blocked_delivery)
    dispatch = asyncio.create_task(bus.dispatch_outbound())
    pending = asyncio.create_task(
        bus.publish_channel_outbound_awaited(envelope, binding)
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert calls == 1
    assert bus._pending_channel_receipts == set()
    bus.stop()
    await dispatch


@pytest.mark.asyncio
async def test_v3_channel_queued_receipt_rejected_on_bus_close() -> None:
    bus = MessageBus()
    envelope, binding = _v3_outbound()
    pending = asyncio.create_task(
        bus.publish_channel_outbound_awaited(envelope, binding)
    )
    await asyncio.sleep(0)
    assert len(bus._pending_channel_receipts) == 1
    await bus.aclose()
    receipt = await pending
    assert receipt.status is DeliveryStatus.REJECTED
    assert bus._pending_channel_receipts == set()


@pytest.mark.asyncio
async def test_v3_channel_publish_after_bus_close_is_immediately_rejected() -> None:
    bus = MessageBus()
    envelope, binding = _v3_outbound()
    calls = 0

    async def deliver(
        _received: OutboundEnvelope,
        _owner: Any,
    ) -> ChannelDeliveryReceipt:
        nonlocal calls
        calls += 1
        raise AssertionError("closed bus must not dispatch")

    bus.bind_channel_outbound_dispatcher(deliver)
    await bus.aclose()

    receipt = await bus.publish_channel_outbound_awaited(envelope, binding)

    assert receipt.status is DeliveryStatus.REJECTED
    assert calls == 0
    assert bus.outbound_size == 0
    await bus.dispatch_outbound()
    assert calls == 0


@pytest.mark.asyncio
async def test_v3_channel_publish_blocked_at_lane_is_rejected_by_concurrent_close() -> None:
    bus = MessageBus()
    envelope, binding = _v3_outbound()
    key, state = bus._chat_lane._acquire_state(
        envelope.channel,
        envelope.recipient,
    )
    try:
        await state.condition.acquire()
        pending = asyncio.create_task(
            bus.publish_channel_outbound_awaited(envelope, binding)
        )
        await asyncio.sleep(0)
        closing = asyncio.create_task(bus.aclose())
        await asyncio.sleep(0)
        assert not pending.done()
        state.condition.release()
        await closing
        receipt = await pending
    finally:
        if state.condition.locked():
            state.condition.release()
        bus._chat_lane._release_state(key, state)

    assert receipt.status is DeliveryStatus.REJECTED
    assert bus.outbound_size == 0


@pytest.mark.asyncio
async def test_v3_channel_inbound_transfers_bus_lane_loop_and_closes_once() -> None:
    bus = MessageBus()
    envelope, lease = _v3_inbound()

    await bus.publish_channel_inbound(envelope)
    assert (envelope.owner, envelope.state) == (
        InboundOwner.BUS,
        InboundState.BUS_QUEUED,
    )
    consumed = await bus.consume_inbound()
    assert consumed is envelope
    assert (envelope.owner, envelope.state) == (
        InboundOwner.LANE,
        InboundState.LANE_QUEUED,
    )
    envelope.handoff(InboundOwner.LANE, InboundOwner.LOOP)
    await bus.complete_inbound(envelope)
    assert (envelope.owner, envelope.state) == (
        InboundOwner.CLOSED,
        InboundState.TERMINAL,
    )
    assert lease.closed == 1


@pytest.mark.asyncio
async def test_v3_mobile_inbound_reserves_before_bus_queue_and_deletes_after_terminal(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:chat-1"
    manager.save(manager.get_or_create(session_key))
    store = manager.control_store
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    bus.bind_mobile_session_admission_owner(manager)
    envelope, lease = _v3_inbound(
        channel="akashic",
        message_id="client-message-1",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-message-1",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-1",
        },
    )

    await bus.publish_channel_inbound(envelope)
    assert [row["handoff_id"] for row in store.list_inbound_handoffs()] == [
        "handoff-1"
    ]
    with pytest.raises(SessionAdmissionConflictError):
        manager.delete_session_with_audit(session_key)
    assert await bus.consume_inbound() is envelope
    envelope.handoff(InboundOwner.LANE, InboundOwner.LOOP)
    await bus.complete_inbound(envelope)

    assert store.list_inbound_handoffs() == []
    assert lease.closed == 1
    manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_handoff_recovers_through_current_exact_binding(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    manager = SessionManager(workspace)
    session_key = "akashic:chat-2"
    manager.save(manager.get_or_create(session_key))
    store = manager.control_store
    original = MessageBus()
    original.bind_durable_inbound_store(store)
    original.bind_mobile_session_admission_owner(manager)
    envelope, _ = _v3_inbound(
        channel="akashic",
        message_id="client-message-2",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-message-2",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-2",
        },
    )
    await original.publish_channel_inbound(envelope)
    manager.close()

    restarted_manager = SessionManager(workspace)
    restarted_manager.clear_stale_admissions()
    store = restarted_manager.control_store
    restarted = MessageBus()
    restarted.bind_durable_inbound_store(store)
    restarted.bind_mobile_session_admission_owner(restarted_manager)
    recovered_leases: list[_InboundLease] = []

    async def recover(raw: object) -> bool:
        assert isinstance(raw, RawInbound)
        lease = _InboundLease(channel="akashic")
        recovered_leases.append(lease)
        recovered = InboundEnvelope(
            message_id=raw.message_id,
            snapshot_id=lease.snapshot_id,
            generation_id=lease.generation_id,
            binding_token=lease.binding_token,
            message=raw.message,
            lease=lease,
        )
        await restarted.publish_channel_inbound(recovered)
        return True

    restarted.bind_mobile_channel_inbound_recoverer(recover)
    await restarted.recover_durable_inbounds()
    recovered = await restarted.consume_inbound()
    assert isinstance(recovered, InboundEnvelope)
    assert recovered.message_id == "client-message-2"
    recovered.handoff(InboundOwner.LANE, InboundOwner.LOOP)
    await restarted.complete_inbound(recovered)

    assert store.list_inbound_handoffs() == []
    assert recovered_leases[0].closed == 1
    restarted_manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_restart_missing_session_keeps_visible_handoff(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    manager = SessionManager(workspace)
    session_key = "akashic:deleted-before-recovery"
    manager.save(manager.get_or_create(session_key))
    original = MessageBus()
    original.bind_durable_inbound_store(manager.control_store)
    original.bind_mobile_session_admission_owner(manager)
    envelope, _ = _v3_inbound(
        channel="akashic",
        message_id="client-deleted-before-recovery",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-deleted-before-recovery",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-deleted-before-recovery",
        },
    )
    await original.publish_channel_inbound(envelope)
    manager.close()

    restarted_manager = SessionManager(workspace)
    restarted_manager.clear_stale_admissions()
    assert restarted_manager.delete_session(session_key) is True
    restarted = MessageBus()
    restarted.bind_durable_inbound_store(restarted_manager.control_store)
    restarted.bind_mobile_session_admission_owner(restarted_manager)

    async def recover(raw: RawInbound) -> bool:
        lease = _InboundLease(channel="akashic")
        recovered = InboundEnvelope(
            message_id=raw.message_id,
            snapshot_id=lease.snapshot_id,
            generation_id=lease.generation_id,
            binding_token=lease.binding_token,
            message=raw.message,
            lease=lease,
        )
        await restarted.publish_channel_inbound(recovered)
        return True

    restarted.bind_mobile_channel_inbound_recoverer(recover)
    with pytest.raises(KeyError, match="session 不存在"):
        await restarted.recover_durable_inbounds()

    assert [
        row["handoff_id"]
        for row in restarted_manager.control_store.list_inbound_handoffs()
    ] == ["handoff-deleted-before-recovery"]
    restarted_manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_bus_close_retains_durable_handoff_for_next_boot(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:chat-3"
    manager.save(manager.get_or_create(session_key))
    store = manager.control_store
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    bus.bind_mobile_session_admission_owner(manager)
    expected_ref = AttachmentRef(
        artifact_id="artifact-3",
        kind=AttachmentKind.IMAGE,
        filename="photo.png",
        media_type="image/png",
        size_bytes=123,
        sha256="a" * 64,
    )
    envelope, lease = _v3_inbound(
        channel="akashic",
        message_id="client-message-3",
        attachments=(expected_ref,),
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-message-3",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-3",
        },
    )

    await bus.publish_channel_inbound(envelope)
    assert bus.pending_mobile_attachment_refs(
        session_key=session_key,
        client_message_id="client-message-3",
    ) == (expected_ref,)
    await bus.aclose()

    assert envelope.state is InboundState.TERMINAL
    assert lease.closed == 1
    assert [row["handoff_id"] for row in store.list_inbound_handoffs()] == [
        "handoff-3"
    ]
    manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_same_process_recovery_does_not_duplicate_live_owner(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:same-process"
    manager.save(manager.get_or_create(session_key))
    bus = MessageBus()
    bus.bind_durable_inbound_store(manager.control_store)
    bus.bind_mobile_session_admission_owner(manager)
    recovered = 0

    async def recover(_raw: RawInbound) -> bool:
        nonlocal recovered
        recovered += 1
        return True

    bus.bind_mobile_channel_inbound_recoverer(recover)
    envelope, _ = _v3_inbound(
        channel="akashic",
        message_id="client-same-process",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-same-process",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-same-process",
        },
    )
    await bus.publish_channel_inbound(envelope)

    await bus.recover_durable_inbounds()

    assert recovered == 0
    assert bus.inbound_size == 1
    await bus.aclose()
    manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_reserve_loses_session_before_lock_without_orphan_row(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:delete-before-reserve"
    manager.save(manager.get_or_create(session_key))
    bus = MessageBus()
    bus.bind_durable_inbound_store(manager.control_store)
    bus.bind_mobile_session_admission_owner(manager)
    envelope, _ = _v3_inbound(
        channel="akashic",
        message_id="client-delete-before-reserve",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-delete-before-reserve",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-delete-before-reserve",
        },
    )
    raw = RawInbound(
        message_id=envelope.message_id,
        provider_identity=envelope.sender,
        recipient=envelope.chat_id,
        message=envelope.message,
    )
    await bus._durable_handoff_lock.acquire()
    reserving = asyncio.create_task(bus.reserve_mobile_channel_handoff(raw))
    await asyncio.sleep(0)
    assert not reserving.done()
    assert manager.delete_session(session_key) is True
    bus._durable_handoff_lock.release()

    with pytest.raises(KeyError, match="session 不存在"):
        await reserving
    assert manager.control_store.list_inbound_handoffs() == []
    assert (
        manager.control_store._conn.execute(
            "SELECT 1 FROM session_admissions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        is None
    )
    await bus.aclose()
    manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_reserve_waiting_on_lock_is_rejected_by_bus_close(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:close-before-reserve-lock"
    manager.save(manager.get_or_create(session_key))
    store = manager.control_store
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    bus.bind_mobile_session_admission_owner(manager)
    envelope, _ = _v3_inbound(
        channel="akashic",
        message_id="client-close-before-reserve-lock",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-close-before-reserve-lock",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-close-before-reserve-lock",
        },
    )
    raw = RawInbound(
        message_id=envelope.message_id,
        provider_identity=envelope.sender,
        recipient=envelope.chat_id,
        message=envelope.message,
    )
    await bus._durable_handoff_lock.acquire()
    reserving = asyncio.create_task(bus.reserve_mobile_channel_handoff(raw))
    await asyncio.sleep(0)
    closing = asyncio.create_task(bus.aclose())
    await asyncio.sleep(0)
    assert bus._outbound_closed is True
    assert not reserving.done()
    assert not closing.done()
    bus._durable_handoff_lock.release()

    with pytest.raises(RuntimeError, match="message bus 已关闭"):
        await reserving
    await closing
    assert store.list_inbound_handoffs() == []
    assert bus._mobile_v3_admissions == {}
    assert (
        store._conn.execute(
            "SELECT 1 FROM session_admissions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        is None
    )
    manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_mark_pending_race_with_close_cannot_queue_after_shutdown(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:close-during-mark-pending"
    manager.save(manager.get_or_create(session_key))
    store = manager.control_store
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    bus.bind_mobile_session_admission_owner(manager)
    envelope, lease = _v3_inbound(
        channel="akashic",
        message_id="client-close-during-mark-pending",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-close-during-mark-pending",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-close-during-mark-pending",
        },
    )
    key, state = bus._chat_lane._acquire_state("akashic", "chat-1")
    try:
        await state.condition.acquire()
        publishing = asyncio.create_task(bus.publish_channel_inbound(envelope))
        await asyncio.sleep(0)
        closing = asyncio.create_task(bus.aclose())
        await asyncio.sleep(0)
        assert bus._outbound_closed is True
        assert not publishing.done()
        assert not closing.done()
        state.condition.release()

        with pytest.raises(RuntimeError, match="message bus 已关闭"):
            await publishing
        await closing
    finally:
        if state.condition.locked():
            state.condition.release()
        bus._chat_lane._release_state(key, state)

    assert envelope.state is InboundState.TERMINAL
    assert lease.closed == 1
    assert bus.inbound_size == 0
    assert bus._mobile_v3_handoffs == {}
    assert bus._mobile_v3_admissions == {}
    assert bus._recovery_claimed == set()
    assert bus._chat_lane._states == {}
    assert [row["handoff_id"] for row in store.list_inbound_handoffs()] == [
        "handoff-close-during-mark-pending"
    ]
    assert (
        store._conn.execute(
            "SELECT 1 FROM session_admissions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        is None
    )
    manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_delete_retry_retains_exact_and_session_owners(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:delete-retry"
    manager.save(manager.get_or_create(session_key))
    store = manager.control_store
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    bus.bind_mobile_session_admission_owner(manager)
    envelope, lease = _v3_inbound(
        channel="akashic",
        message_id="client-delete-retry",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-delete-retry",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-delete-retry",
        },
    )
    await bus.publish_channel_inbound(envelope)
    assert await bus.consume_inbound() is envelope
    envelope.handoff(InboundOwner.LANE, InboundOwner.LOOP)
    original_complete = store.complete_inbound_handoff
    attempts = 0

    def fail_once(handoff_id: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("delete busy")
        original_complete(handoff_id)

    store.complete_inbound_handoff = fail_once  # type: ignore[method-assign]
    with pytest.raises(OSError, match="delete busy"):
        await bus.complete_inbound(envelope)

    assert lease.closed == 0
    with pytest.raises(SessionAdmissionConflictError):
        manager.delete_session_with_audit(session_key)
    await asyncio.wait_for(lease.closed_event.wait(), timeout=2)
    cleanup_tasks = tuple(bus._inbound_cleanup_tasks.values())
    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks)
    assert store.list_inbound_handoffs() == []
    assert attempts == 2
    assert (
        store._conn.execute(
            "SELECT 1 FROM session_admissions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        is None
    )
    await bus.aclose()
    manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_delete_failure_then_bus_close_keeps_durable_row(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:delete-close"
    manager.save(manager.get_or_create(session_key))
    store = manager.control_store
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    bus.bind_mobile_session_admission_owner(manager)
    envelope, lease = _v3_inbound(
        channel="akashic",
        message_id="client-delete-close",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-delete-close",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-delete-close",
        },
    )
    await bus.publish_channel_inbound(envelope)
    assert await bus.consume_inbound() is envelope
    envelope.handoff(InboundOwner.LANE, InboundOwner.LOOP)

    def always_fail(_handoff_id: str) -> None:
        raise OSError("delete unavailable")

    store.complete_inbound_handoff = always_fail  # type: ignore[method-assign]
    with pytest.raises(OSError, match="delete unavailable"):
        await bus.complete_inbound(envelope)
    await bus.aclose()

    assert lease.closed == 1
    assert [row["handoff_id"] for row in store.list_inbound_handoffs()] == [
        "handoff-delete-close"
    ]
    assert (
        store._conn.execute(
            "SELECT 1 FROM session_admissions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        is None
    )
    manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_completion_cancellation_waits_for_exact_cleanup(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:cancel-cleanup"
    manager.save(manager.get_or_create(session_key))
    store = manager.control_store
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    bus.bind_mobile_session_admission_owner(manager)
    close_gate = asyncio.Event()
    envelope, lease = _v3_inbound(
        close_gate,
        channel="akashic",
        message_id="client-cancel-cleanup",
        metadata={
            "session_key_override": session_key,
            "client_message_id": "client-cancel-cleanup",
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-cancel-cleanup",
        },
    )
    await bus.publish_channel_inbound(envelope)
    assert await bus.consume_inbound() is envelope
    envelope.handoff(InboundOwner.LANE, InboundOwner.LOOP)
    completing = asyncio.create_task(bus.complete_inbound(envelope))
    await lease.close_started.wait()
    completing.cancel()
    await asyncio.sleep(0)
    assert not completing.done()
    close_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await completing

    assert store.list_inbound_handoffs() == []
    assert lease.closed == 1
    assert (
        store._conn.execute(
            "SELECT 1 FROM session_admissions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        is None
    )
    await bus.aclose()
    manager.close()


@pytest.mark.asyncio
async def test_v3_channel_inbound_bus_close_releases_queued_exact_lease() -> None:
    bus = MessageBus()
    envelope, lease = _v3_inbound()
    await bus.publish_channel_inbound(envelope)

    await bus.aclose()

    assert envelope.state is InboundState.TERMINAL
    assert lease.closed == 1
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_v3_channel_inbound_blocked_at_lane_is_closed_by_concurrent_bus_close() -> None:
    bus = MessageBus()
    envelope, lease = _v3_inbound()
    key, state = bus._chat_lane._acquire_state(
        envelope.channel,
        envelope.chat_id,
    )
    try:
        await state.condition.acquire()
        pending = asyncio.create_task(bus.publish_channel_inbound(envelope))
        await asyncio.sleep(0)
        await bus.aclose()
        state.condition.release()
        with pytest.raises(RuntimeError, match="已关闭"):
            await pending
    finally:
        if state.condition.locked():
            state.condition.release()
        bus._chat_lane._release_state(key, state)

    assert envelope.state is InboundState.TERMINAL
    assert lease.closed == 1
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_v3_channel_bus_close_cancellation_drains_every_queued_lease() -> None:
    bus = MessageBus()
    close_gate = asyncio.Event()
    first, first_lease = _v3_inbound(close_gate, message_id="message-1")
    second, second_lease = _v3_inbound(message_id="message-2")
    await bus.publish_channel_inbound(first)
    await bus.publish_channel_inbound(second)

    closing = asyncio.create_task(bus.aclose())
    await first_lease.close_started.wait()
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    close_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert first.state is InboundState.TERMINAL
    assert second.state is InboundState.TERMINAL
    assert first_lease.closed == second_lease.closed == 1
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_v3_channel_inbound_release_cancellation_clears_lane_before_return() -> None:
    bus = MessageBus()
    close_gate = asyncio.Event()
    envelope, lease = _v3_inbound(close_gate)
    await bus.publish_channel_inbound(envelope)
    assert await bus.consume_inbound() is envelope

    releasing = asyncio.create_task(
        bus.release_channel_inbound(envelope, InboundOwner.LANE)
    )
    await lease.close_started.wait()
    releasing.cancel()
    await asyncio.sleep(0)
    assert not releasing.done()
    close_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await releasing

    assert envelope.state is InboundState.TERMINAL
    assert lease.closed == 1
    assert bus._chat_lane._states == {}

    completed = False

    async def mark_completed() -> None:
        nonlocal completed
        completed = True

    await asyncio.wait_for(
        bus._chat_lane.run_non_passive(
            envelope.channel,
            envelope.chat_id,
            mark_completed,
        ),
        timeout=1,
    )
    assert completed


@pytest.mark.asyncio
async def test_v3_channel_worker_preserves_exact_binding_through_terminal_delivery(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    manager.save(manager.get_or_create("feishu:chat-1"))
    store = manager.control_store
    seen_request: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        from agent.plugins.channel_generation_host import (
            get_current_channel_turn_binding,
        )
        from agent.plugins.snapshot import get_current_runtime_snapshot

        assert get_current_runtime_snapshot() is not None
        assert get_current_runtime_snapshot().snapshot_id == "snapshot-1"
        channel_binding = get_current_channel_turn_binding()
        assert channel_binding is lease
        seen_request.append(request)
        return ControlExecutionResult(response=f"echo:{request.input}")

    runtime = ConversationRuntime(store, execute)
    bus = MessageBus()
    delivered: list[tuple[OutboundEnvelope, object]] = []

    async def dispatch(
        envelope: OutboundEnvelope,
        binding: object,
    ) -> ChannelDeliveryReceipt:
        delivered.append((envelope, binding))
        return ChannelDeliveryReceipt(
            envelope.delivery_id,
            DeliveryStatus.DELIVERED,
            ("provider-1",),
        )

    bus.bind_channel_outbound_dispatcher(dispatch)
    from bootstrap.passive_worker import PassiveMessageWorker

    worker = PassiveMessageWorker(
        bus,
        runtime,
        cast(Any, SimpleNamespace(session_manager=manager)),
    )
    worker_task = asyncio.create_task(worker.run())
    dispatch_task = asyncio.create_task(bus.dispatch_outbound())
    envelope, lease = _v3_inbound()

    await bus.publish_channel_inbound(envelope)
    await asyncio.wait_for(lease.closed_event.wait(), timeout=2)
    while envelope.state is not InboundState.TERMINAL:
        await asyncio.sleep(0)

    assert len(seen_request) == 1
    assert seen_request[0].metadata["channelBindingToken"] == lease.binding_token
    assert (
        seen_request[0].metadata["inboundMetadata"]["client_message_id"]
        == envelope.message_id
    )
    assert len(delivered) == 1
    outbound, binding = delivered[0]
    assert outbound.body == "echo:hello"
    assert outbound.snapshot_id == lease.snapshot_id
    assert outbound.generation_id == lease.generation_id
    assert outbound.binding_token == lease.binding_token
    assert binding is lease
    assert envelope.state is InboundState.TERMINAL
    assert lease.closed == 1

    worker.stop()
    bus.stop()
    await worker_task
    await dispatch_task
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_recovery_redelivers_existing_turn_without_duplicate(
    tmp_path: Path,
) -> None:
    manager1 = SessionManager(tmp_path)
    session_key = "akashic:chat-1"
    client_message_id = "client-recovered-1"
    manager1.save(manager1.get_or_create(session_key))
    executed: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        executed.append(request)
        return ControlExecutionResult(response=f"echo:\n{request.input}")

    runtime1 = ConversationRuntime(manager1.control_store, execute)
    original = await runtime1.start_turn(
        TurnRequest(
            session_key,
            "hello",
            {"inboundMetadata": {"client_message_id": client_message_id}},
        )
    )
    original_result = await original.result()

    bus1 = MessageBus()
    bus1.bind_durable_inbound_store(manager1.control_store)
    bus1.bind_mobile_session_admission_owner(manager1)
    envelope1, lease1 = _v3_inbound(
        channel="akashic",
        message_id=client_message_id,
        metadata={
            "session_key_override": session_key,
            "client_message_id": client_message_id,
            "mobile_v3_handoff": True,
            "mobile_handoff_id": "handoff-recovered-1",
        },
    )
    await bus1.publish_channel_inbound(envelope1)
    await bus1.aclose()
    assert lease1.closed == 1
    assert len(manager1.control_store.list_inbound_handoffs()) == 1
    await runtime1.shutdown()
    manager1.close()

    manager2 = SessionManager(tmp_path)
    manager2.clear_stale_admissions()
    runtime2 = ConversationRuntime(manager2.control_store, execute)
    bus2 = MessageBus()
    bus2.bind_durable_inbound_store(manager2.control_store)
    bus2.bind_mobile_session_admission_owner(manager2)
    delivered: list[OutboundEnvelope] = []
    recovered_leases: list[_InboundLease] = []
    recovered_event = asyncio.Event()

    async def dispatch(
        envelope: OutboundEnvelope,
        _binding: object,
    ) -> ChannelDeliveryReceipt:
        delivered.append(envelope)
        return ChannelDeliveryReceipt(
            envelope.delivery_id,
            DeliveryStatus.DELIVERED,
        )

    async def recover(raw: RawInbound) -> bool:
        lease = _InboundLease(channel="akashic")
        recovered_leases.append(lease)
        envelope = InboundEnvelope(
            message_id=raw.message_id,
            snapshot_id=lease.snapshot_id,
            generation_id=lease.generation_id,
            binding_token=lease.binding_token,
            message=raw.message,
            lease=lease,
        )
        await bus2.publish_channel_inbound(envelope)
        recovered_event.set()
        return True

    bus2.bind_mobile_channel_inbound_recoverer(recover)
    bus2.bind_channel_outbound_dispatcher(dispatch)
    from bootstrap.passive_worker import PassiveMessageWorker

    worker = PassiveMessageWorker(
        bus2,
        runtime2,
        cast(Any, SimpleNamespace(session_manager=manager2)),
    )
    worker_task = asyncio.create_task(worker.run())
    dispatch_task = asyncio.create_task(bus2.dispatch_outbound())
    await asyncio.wait_for(recovered_event.wait(), timeout=2)
    await asyncio.wait_for(recovered_leases[0].closed_event.wait(), timeout=2)

    turns = manager2.control_store.list_turns(session_key, limit=10)
    assert len(executed) == 1
    assert len(turns) == 1
    assert turns[0].id == original_result.id
    assert len(delivered) == 1
    assert delivered[0].control_turn_id == original_result.id
    assert delivered[0].body == "echo:\nhello"
    assert manager2.control_store.list_inbound_handoffs() == []
    for _ in range(100):
        admission = manager2.control_store._conn.execute(
            "SELECT 1 FROM session_admissions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        if admission is None:
            break
        await asyncio.sleep(0)
    assert admission is None

    worker.stop()
    bus2.stop()
    await worker_task
    await dispatch_task
    await runtime2.shutdown()
    manager2.close()


@pytest.mark.asyncio
async def test_v3_channel_worker_holds_session_admission_until_terminal(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    manager.save(manager.get_or_create("feishu:chat-1"))
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        started.set()
        await release.wait()
        return ControlExecutionResult(response="done")

    runtime = ConversationRuntime(manager.control_store, execute)
    bus = MessageBus()

    async def dispatch(
        envelope: OutboundEnvelope,
        _binding: object,
    ) -> ChannelDeliveryReceipt:
        return ChannelDeliveryReceipt(
            envelope.delivery_id,
            DeliveryStatus.DELIVERED,
        )

    bus.bind_channel_outbound_dispatcher(dispatch)
    from bootstrap.passive_worker import PassiveMessageWorker

    worker = PassiveMessageWorker(
        bus,
        runtime,
        cast(Any, SimpleNamespace(session_manager=manager)),
    )
    worker_task = asyncio.create_task(worker.run())
    dispatch_task = asyncio.create_task(bus.dispatch_outbound())
    envelope, lease = _v3_inbound()

    await bus.publish_channel_inbound(envelope)
    await asyncio.wait_for(started.wait(), timeout=2)
    with pytest.raises(SessionAdmissionConflictError, match="正在处理消息"):
        manager.delete_session("feishu:chat-1")

    release.set()
    await asyncio.wait_for(lease.closed_event.wait(), timeout=2)
    deleted = False
    for _ in range(100):
        try:
            deleted = manager.delete_session("feishu:chat-1")
        except SessionAdmissionConflictError:
            await asyncio.sleep(0)
            continue
        break
    assert deleted is True

    worker.stop()
    bus.stop()
    await worker_task
    await dispatch_task
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_channel_worker_rejects_image_budget_before_acquiring_leases(
    tmp_path: Path,
) -> None:
    from PIL import Image

    from bootstrap.passive_worker import PassiveMessageWorker
    from infra.channels.artifacts import ChannelAttachmentArtifactStore

    session_store = SessionStore(tmp_path / "artifact-sessions.db")
    artifact_store = ChannelAttachmentArtifactStore(
        workspace=tmp_path,
        session_store=session_store,
    )
    image_path = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image_path)
    extensionless_refs = tuple(
        [
            await artifact_store.import_bytes(
                image_path.read_bytes(),
                kind=AttachmentKind.FILE,
                filename=f"extensionless-{index}",
                media_type=None,
            )
            for index in range(5)
        ]
    )
    session_store.close()
    assert all(ref.kind is AttachmentKind.IMAGE for ref in extensionless_refs)

    class Store:
        def __init__(self) -> None:
            self.acquire_calls = 0

        async def acquire(self, _ref: AttachmentRef) -> object:
            self.acquire_calls += 1
            raise AssertionError("budget validation must run before acquire")

    store = Store()
    worker = object.__new__(PassiveMessageWorker)
    worker._attachment_store = cast(Any, store)

    with pytest.raises(ValueError, match="最多可以添加 4 张图片"):
        await worker._acquire_attachment_refs(extensionless_refs)

    too_many = tuple(
        AttachmentRef(
            artifact_id=f"image-{index}",
            kind=AttachmentKind.IMAGE,
            filename=f"{index}.png",
            media_type="image/png",
            size_bytes=1,
            sha256=f"{index:064x}",
        )
        for index in range(5)
    )
    with pytest.raises(ValueError, match="最多可以添加 4 张图片"):
        await worker._acquire_attachment_refs(too_many)

    oversized = (
        AttachmentRef(
            artifact_id="oversized-image",
            kind=AttachmentKind.IMAGE,
            filename="oversized.png",
            media_type="image/png",
            size_bytes=21 * 1024 * 1024,
            sha256=f"{100:064x}",
        ),
    )
    with pytest.raises(ValueError, match="单张图片不能超过 20MB"):
        await worker._acquire_attachment_refs(oversized)

    too_large = tuple(
        AttachmentRef(
            artifact_id=f"large-{index}",
            kind=AttachmentKind.IMAGE,
            filename=f"large-{index}.png",
            media_type="image/png",
            size_bytes=15 * 1024 * 1024,
            sha256=f"{index + 10:064x}",
        )
        for index in range(3)
    )
    with pytest.raises(ValueError, match="图片合计不能超过 40MB"):
        await worker._acquire_attachment_refs(too_large)
    assert store.acquire_calls == 0


@pytest.mark.asyncio
async def test_v3_channel_worker_projects_and_closes_attachment_lease(
    tmp_path: Path,
) -> None:
    from bootstrap.passive_worker import PassiveMessageWorker
    from infra.channels.artifacts import ChannelAttachmentArtifactStore

    manager = SessionManager(tmp_path)
    manager.save(manager.get_or_create("feishu:chat-1"))
    store = manager.control_store
    attachment_store = ChannelAttachmentArtifactStore(
        workspace=tmp_path,
        session_store=store,
    )
    ref = await attachment_store.import_bytes(
        b"attachment-body",
        kind=AttachmentKind.FILE,
        filename="note.txt",
        media_type="text/plain",
    )
    seen_paths: list[str] = []

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        media = cast(list[str], request.metadata["media"])
        assert len(media) == 1
        assert Path(media[0]).read_bytes() == b"attachment-body"
        seen_paths.extend(media)
        return ControlExecutionResult(
            "ok",
            assistant_data={"attachmentIds": [ref.artifact_id]},
        )

    runtime = ConversationRuntime(store, execute)
    bus = MessageBus()

    async def dispatch(
        envelope: OutboundEnvelope,
        _binding: object,
    ) -> ChannelDeliveryReceipt:
        assert envelope.attachments == (ref,)
        return ChannelDeliveryReceipt(
            envelope.delivery_id,
            DeliveryStatus.DELIVERED,
        )

    bus.bind_channel_outbound_dispatcher(dispatch)
    worker = PassiveMessageWorker(
        bus,
        runtime,
        cast(Any, SimpleNamespace(session_manager=manager)),
        attachment_store=attachment_store,
    )
    worker_task = asyncio.create_task(worker.run())
    dispatch_task = asyncio.create_task(bus.dispatch_outbound())
    envelope, lease = _v3_inbound(
        attachments=(ref,),
        metadata={"client_message_id": "client-attachment-1"},
    )

    await bus.publish_channel_inbound(envelope)
    await asyncio.wait_for(lease.closed_event.wait(), timeout=2)

    assert seen_paths and not Path(seen_paths[0]).exists()

    worker.stop()
    bus.stop()
    await worker_task
    await dispatch_task
    await runtime.shutdown()
    manager.close()


@pytest.mark.asyncio
async def test_v3_channel_worker_cancel_closes_running_and_lane_queued_leases(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    manager.save(manager.get_or_create("feishu:chat-1"))
    store = manager.control_store
    started = asyncio.Event()
    never = asyncio.Event()

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        started.set()
        await never.wait()
        return ControlExecutionResult(response="unreachable")

    runtime = ConversationRuntime(store, execute)
    bus = MessageBus()

    async def dispatch(
        envelope: OutboundEnvelope,
        _binding: object,
    ) -> ChannelDeliveryReceipt:
        return ChannelDeliveryReceipt(
            envelope.delivery_id,
            DeliveryStatus.DELIVERED,
        )

    bus.bind_channel_outbound_dispatcher(dispatch)
    from bootstrap.passive_worker import PassiveMessageWorker

    worker = PassiveMessageWorker(
        bus,
        runtime,
        cast(Any, SimpleNamespace(session_manager=manager)),
    )
    worker_task = asyncio.create_task(worker.run())
    dispatch_task = asyncio.create_task(bus.dispatch_outbound())
    first, first_lease = _v3_inbound(message_id="message-1")
    second, second_lease = _v3_inbound(message_id="message-2")
    await bus.publish_channel_inbound(first)
    await bus.publish_channel_inbound(second)
    await asyncio.wait_for(started.wait(), timeout=1)

    worker_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(worker_task, timeout=2)

    assert first.state is InboundState.TERMINAL
    assert second.state is InboundState.TERMINAL
    assert first_lease.closed == second_lease.closed == 1
    assert worker._lane_tasks == {}
    assert worker._lane_queues == {}
    assert worker._channel_result_tasks == {}

    bus.stop()
    await dispatch_task
    await runtime.shutdown()
    manager.close()

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
import infra.mobile_realtime.channel as channel_module
import infra.mobile_realtime.gateway as gateway_module

from agent.config_models import MobileRealtimeConfig
from agent.control.models import TurnRecord, TurnStatus
from agent.plugin_composition.channels import (
    AttachmentKind as V3AttachmentKind,
    AttachmentRef,
    ChannelCommitRole,
    ChannelFactoryContext,
    ChannelRuntimePorts,
    DeliveryStatus as V3DeliveryStatus,
    ProviderDeliveryRequest,
    RawInbound,
)
from infra.mobile_realtime.runtime_inspection import RuntimeInspectionService
from bus.events import (
    AttachmentKind,
    ChannelAttachment,
    ChannelMessage,
    DeliveryStatus,
    OutboundMessage,
    TurnTerminalStatus,
    channel_message_from_outbound,
)
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnOutputCompleted,
    TurnStarted,
    TurnCommitted,
)
from bus.queue import MessageBus
from infra.channels.base import AttachmentStore
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.mobile_realtime.channel import MobileRealtimeChannel
from infra.mobile_realtime.gateway import MobileGatewayRuntime
from infra.mobile_realtime.protocol import (
    GenericCommand,
    MAX_JSON_FRAME_BYTES,
    MessageSendCommand,
    parse_frame,
)
from infra.mobile_realtime.remote_media import RemoteMediaError, RemoteMediaSnapshot
from infra.mobile_realtime.storage import (
    AttachmentRecord,
    DeviceRecord,
    MobileStorageError,
    MobileRealtimeStorage,
)
from session.manager import SessionManager


class _Runtime:
    def __init__(self, storage: MobileRealtimeStorage) -> None:
        self.storage = storage
        self.config = MobileRealtimeConfig(max_attachment_mb=50)
        self.events: list[dict[str, object]] = []

    def _recipient_count(self) -> int:
        list_active_devices = getattr(self.storage, "list_active_devices", None)
        if not callable(list_active_devices):
            return 1
        active_devices = list_active_devices()
        if not isinstance(active_devices, tuple):
            raise TypeError("list_active_devices 必须返回 tuple")
        return len(active_devices)

    async def publish_event(self, **event: object) -> int:
        self.events.append(dict(event))
        return self._recipient_count()

    async def publish_connection_control(self, **control: object) -> None:
        self.events.append(dict(control))

    async def publish_event_with_outbound_attachments(
        self,
        *,
        candidates: tuple[AttachmentRecord, ...],
        payload_builder: Any,
        session_id: str,
    ) -> tuple[AttachmentRecord, ...]:
        resolved, _recipient_count = (
            await self.publish_event_with_outbound_attachments_result(
                candidates=candidates,
                payload_builder=payload_builder,
                session_id=session_id,
            )
        )
        return resolved

    async def publish_event_with_outbound_attachments_result(
        self,
        *,
        candidates: tuple[AttachmentRecord, ...],
        payload_builder: Any,
        session_id: str,
    ) -> tuple[tuple[AttachmentRecord, ...], int]:
        event_id = gateway_module._new_ulid()
        resolved, events = self.storage.commit_outbound_event(
            candidates,
            device_ids=tuple(
                device.device_id for device in self.storage.list_active_devices()
            ),
            event_id=event_id,
            envelope_builder=lambda records: gateway_module._encode_stored_event(
                event_id=event_id,
                event_type="message.proactive",
                payload=payload_builder(records),
                session_id=session_id,
                turn_id=None,
            ),
            created_at=datetime.now(timezone.utc),
        )
        self.events.append({"durable": events})
        return resolved, len(events)

    async def refresh_device_capabilities(
        self,
        *,
        device_id: str,
        capabilities: tuple[str, ...],
    ) -> None:
        self.storage.update_device_capabilities(device_id, capabilities)


class _GatedPublishRuntime(_Runtime):
    """publish_event 在写入事件后挂起，直到测试放行，用于观测发布边界。"""

    def __init__(self, storage: MobileRealtimeStorage) -> None:
        super().__init__(storage)
        self.delta_publish_started = asyncio.Event()
        self.delta_publish_release = asyncio.Event()

    async def publish_event(self, **event: object) -> int:
        self.events.append(dict(event))
        if event.get("event_type") == "react.thinking.delta":
            self.delta_publish_started.set()
            await self.delta_publish_release.wait()
        return self._recipient_count()


class _FinalGatedRuntime(_Runtime):
    """publish_event 在 message.final 写入事件后挂起，用于卡住终态 barrier 临界区。"""

    def __init__(self, storage: MobileRealtimeStorage) -> None:
        super().__init__(storage)
        self.final_started = asyncio.Event()
        self.final_release = asyncio.Event()

    async def publish_event(self, **event: object) -> int:
        self.events.append(dict(event))
        if event.get("event_type") == "message.final":
            self.final_started.set()
            await self.final_release.wait()
        return self._recipient_count()


class _FailOnceTerminalRuntime(_Runtime):
    """指定终态事件第一次调用在持久化前抛 OSError，之后各次成功。"""

    def __init__(
        self,
        storage: MobileRealtimeStorage,
        *,
        fail_type: str = "message.final",
    ) -> None:
        super().__init__(storage)
        self.fail_type = fail_type
        self.terminal_attempts = 0

    async def publish_event(self, **event: object) -> int:
        if event.get("event_type") == self.fail_type:
            self.terminal_attempts += 1
            if self.terminal_attempts == 1:
                raise OSError("inbox write failed")
        self.events.append(dict(event))
        return self._recipient_count()


class _GatedFailOnceTerminalRuntime(_FailOnceTerminalRuntime):
    """终态第一次发布在持久化前挂起后抛 OSError，固定失败间隙的锁编排。"""

    def __init__(
        self,
        storage: MobileRealtimeStorage,
        *,
        fail_type: str = "message.final",
    ) -> None:
        super().__init__(storage, fail_type=fail_type)
        self.terminal_started = asyncio.Event()
        self.terminal_release = asyncio.Event()

    async def publish_event(self, **event: object) -> int:
        if event.get("event_type") == self.fail_type:
            self.terminal_attempts += 1
            if self.terminal_attempts == 1:
                self.terminal_started.set()
                await self.terminal_release.wait()
                raise OSError("inbox write failed")
        self.events.append(dict(event))
        return self._recipient_count()


class _FailDeltaRuntime(_Runtime):
    """第 N 次 delta 发布在持久化前抛 OSError（不入 wire），其余成功；统计尝试数。"""

    def __init__(self, storage: MobileRealtimeStorage, *, fail_at: int = 1) -> None:
        super().__init__(storage)
        self.fail_at = fail_at
        self.delta_attempts = 0

    async def publish_event(self, **event: object) -> int:
        if event.get("event_type") in {"answer.delta", "react.thinking.delta"}:
            self.delta_attempts += 1
            if self.delta_attempts == self.fail_at:
                raise OSError("delta publish failed")
        self.events.append(dict(event))
        return self._recipient_count()


class _Bus:
    def __init__(self) -> None:
        self.inbound: list[object] = []
        self.pending_handoff = False
        self.order: list[str] = []

    async def publish_inbound(self, message: object) -> None:
        self.inbound.append(message)

    def bind_mobile_channel_inbound_recoverer(self, recoverer: object) -> None:
        channel = getattr(recoverer, "__self__", None)
        assert isinstance(channel, MobileRealtimeChannel)
        channel._attach_v3_inbound(
            ChannelRuntimePorts(
                snapshot_id="test-snapshot",
                generation_id="test-generation",
                binding_token="test-binding",
                ingress=self,
                identity=None,
                attachment_import=None,
                recovery_ingress=self,
            )
        )
        channel._open_v3_inbound()

    async def admit(self, raw: RawInbound) -> bool:
        self.order.append("admit")
        self.inbound.append(raw)
        return True

    async def recover(self, raw: RawInbound) -> bool:
        self.order.append("recover")
        self.inbound.append(raw)
        return True

    async def reserve_mobile_channel_handoff(self, raw: RawInbound) -> bool:
        assert raw.message.metadata.get("mobile_v3_handoff") is True
        self.order.append("reserve")
        return True

    async def defer_mobile_channel_handoff(self, handoff_id: str) -> None:
        assert handoff_id
        self.pending_handoff = True

    def has_pending_mobile_handoff(
        self,
        *,
        session_key: str,
        client_message_id: str,
    ) -> bool:
        return self.pending_handoff


class _FailingBus(_Bus):
    async def publish_inbound(self, message: object) -> None:
        raise RuntimeError("bus unavailable")

    async def admit(self, raw: RawInbound) -> bool:
        raise RuntimeError("bus unavailable")


class _GatedReserveBus(_Bus):
    def __init__(self) -> None:
        super().__init__()
        self.reserve_started = asyncio.Event()
        self.reserve_release = asyncio.Event()

    async def reserve_mobile_channel_handoff(self, raw: RawInbound) -> bool:
        self.order.append("reserve")
        self.reserve_started.set()
        await self.reserve_release.wait()
        return True


class _RejectingIngressBus(_Bus):
    async def admit(self, raw: RawInbound) -> bool:
        self.order.append("admit-rejected")
        return False


class _EventBus:
    def __init__(self) -> None:
        self.handlers: dict[type[object], object] = {}

    def on(self, event_type: type[object], handler: object) -> None:
        self.handlers[event_type] = handler


class _PushTool:
    pass


class _ProviderFactory:
    async def create(self, _credentials: object) -> object:
        raise AssertionError("Mobile native adapter 不应创建 provider client")

    async def aclose(self) -> None:
        return None


class _ReadLease:
    def __init__(self, ref: AttachmentRef, data: bytes) -> None:
        self.ref = ref
        self._data = data
        self.closed = False

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        if len(self._data) > max_bytes:
            raise ValueError("read limit")
        return self._data

    async def aclose(self) -> None:
        self.closed = True


class _ReadPort:
    def __init__(self, data: dict[str, bytes]) -> None:
        self._data = data
        self.leases: list[_ReadLease] = []

    async def acquire(self, ref: AttachmentRef) -> _ReadLease:
        lease = _ReadLease(ref, self._data[ref.artifact_id])
        self.leases.append(lease)
        return lease


async def _started_native_mobile_channel(
    tmp_path: Path,
    *,
    active_device: bool = True,
    runtime: Any | None = None,
) -> tuple[MobileRealtimeChannel, MobileRealtimeStorage, SessionManager]:
    if runtime is None:
        storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    else:
        storage = cast(MobileRealtimeStorage, runtime.storage)
    if active_device:
        _register_device(storage, "device-1")
    manager = SessionManager(tmp_path / "workspace")
    channel = MobileRealtimeChannel(
        cast(MobileGatewayRuntime, runtime or _Runtime(storage))
    )
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    return channel, storage, manager


def _native_context(
    read_port: _ReadPort | None = None,
) -> ChannelFactoryContext:
    return ChannelFactoryContext(
        snapshot_id="snapshot-1",
        generation_id="generation-1",
        binding_token="binding-1",
        config={},
        credentials={},
        provider_client_factory=cast(Any, _ProviderFactory()),
        ingress=None,
        identity=None,
        attachment_read=read_port,
    )


def _native_ref(data: bytes, *, artifact_id: str = "core-artifact-1") -> AttachmentRef:
    return AttachmentRef(
        artifact_id=artifact_id,
        kind=V3AttachmentKind.FILE,
        filename="report.txt",
        media_type="text/plain",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _provider_delivery(channel: MobileRealtimeChannel):
    """把测试 outbound 映射为正式 Channel provider callback。"""

    async def deliver(message: OutboundMessage | ChannelMessage):
        if isinstance(message, ChannelMessage):
            return await channel._deliver_message(message)
        channel_message = channel_message_from_outbound(message)
        metadata = dict(channel_message.metadata)
        metadata["_channel_commit_role"] = "passive"
        return await channel._deliver_message(
            replace(channel_message, metadata=metadata)
        )

    return deliver


@pytest.mark.asyncio
async def test_mobile_message_send_uses_exact_v3_ingress_without_legacy_bus(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    manager = SessionManager(tmp_path / "workspace")
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    reply = await channel.handle_command(
        device_id=device_id,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            session_id=session_id,
        ),
    )

    assert reply.type == "message.send.ok"
    assert len(bus.inbound) == 1
    raw = cast(RawInbound, bus.inbound[0])
    assert raw.message.channel == "akashic"
    assert raw.message.metadata["session_key_override"] == session_id
    assert raw.message.metadata["mobile_v3_handoff"] is True
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_mobile_captured_callback_blocks_drain_through_preprocessing(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))
    bus = _GatedReserveBus()
    manager = SessionManager(tmp_path / "workspace")
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    sending = asyncio.create_task(
        channel.handle_command(
            device_id=device_id,
            frame=_message_frame(
                frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAA",
                session_id=session_id,
            ),
        )
    )
    await bus.reserve_started.wait()
    channel._close_v3_inbound()
    draining = asyncio.create_task(channel._drain_v3_inbound())
    await asyncio.sleep(0)
    assert not draining.done()

    with pytest.raises(RuntimeError, match="admission 已关闭"):
        await channel.handle_command(
            device_id=device_id,
            frame=_message_frame(
                frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAB",
                session_id=f"akashic:{uuid4()}",
            ),
        )
    bus.reserve_release.set()
    reply = await sending
    await draining

    assert reply.type == "message.send.ok"
    assert bus.order == ["reserve", "admit"]
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_mobile_exact_ingress_false_never_commits_success_receipt(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))
    bus = _RejectingIngressBus()
    manager = SessionManager(tmp_path / "workspace")
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    frame = _message_frame(
        frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAC",
        session_id=f"akashic:{uuid4()}",
    )

    with pytest.raises(RuntimeError, match="captured binding fence"):
        await channel.handle_command(device_id=device_id, frame=frame)

    receipt = storage._db.execute(
        "SELECT status, reply_type FROM mobile_command_receipts "
        "WHERE device_id = ? AND command_id = ?",
        (device_id, frame.id),
    ).fetchone()
    assert receipt is not None
    assert tuple(receipt) == ("processing", None)
    assert bus.order == ["reserve", "admit-rejected"]
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_adapter_rejects_without_active_device(
    tmp_path: Path,
) -> None:
    channel, storage, manager = await _started_native_mobile_channel(
        tmp_path,
        active_device=False,
    )
    adapter = channel.build_v3_adapter(_native_context())
    ready = await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token=ready.binding_token,
            delivery_id="delivery-no-device",
            recipient=f"akashic:{uuid4()}",
            body="hello",
        )
    )
    assert receipt.status is V3DeliveryStatus.REJECTED
    assert receipt.error is not None
    assert await adapter.stop() == await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_adapter_reads_opaque_ref_and_commits_one_event(
    tmp_path: Path,
) -> None:
    channel, storage, manager = await _started_native_mobile_channel(tmp_path)
    data = b"native-mobile-attachment"
    ref = AttachmentRef(
        artifact_id="core-artifact-1",
        kind=V3AttachmentKind.FILE,
        filename="report.txt",
        media_type="text/plain",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    read_port = _ReadPort({ref.artifact_id: data})
    adapter = channel.build_v3_adapter(_native_context(read_port))
    await adapter.start()
    session_id = f"akashic:{uuid4()}"
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-with-attachment",
            recipient=session_id,
            body="attached",
            attachments=(ref,),
            metadata={
                "delivery_id": "delivery-with-attachment",
                "nested": {"ordinal": 1},
            },
        )
    )
    assert receipt.status is V3DeliveryStatus.DELIVERED
    assert read_port.leases and all(lease.closed for lease in read_port.leases)
    durable = storage.read_durable_events(
        "device-1",
        after_event_seq=0,
        limit=10,
    )
    assert len(durable) == 1
    envelope = json.loads(durable[0].envelope_json)
    descriptor = envelope["payload"]["attachments"][0]
    assert descriptor["filename"] == "report.txt"
    assert descriptor["sha256"] == ref.sha256
    assert envelope["payload"]["metadata"]["nested"] == {"ordinal": 1}
    assert ref.artifact_id not in durable[0].envelope_json
    record = storage.read_attachment(descriptor["attachment_id"])
    assert record is not None
    assert Path(record.local_path).read_bytes() == data
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_adapter_reports_unknown_if_durable_call_raises(
    tmp_path: Path,
) -> None:
    class _FailingRuntime(_Runtime):
        async def publish_event(self, **_event: object) -> int:
            raise OSError("durable write outcome unknown")

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    _register_device(storage, "device-1")
    manager = SessionManager(tmp_path / "workspace")
    channel = MobileRealtimeChannel(
        cast(MobileGatewayRuntime, _FailingRuntime(storage))
    )
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    adapter = channel.build_v3_adapter(_native_context())
    await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-unknown",
            recipient=f"akashic:{uuid4()}",
            body="unknown",
        )
    )
    assert receipt.status is V3DeliveryStatus.UNKNOWN
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_adapter_rejects_mismatched_lease_ref(
    tmp_path: Path,
) -> None:
    channel, storage, manager = await _started_native_mobile_channel(tmp_path)
    data = b"lease-identity"
    requested = _native_ref(data)
    mismatched = _native_ref(data, artifact_id="different-artifact")

    class _MismatchedReadPort(_ReadPort):
        async def acquire(self, _ref: AttachmentRef) -> _ReadLease:
            lease = _ReadLease(mismatched, data)
            self.leases.append(lease)
            return lease

    read_port = _MismatchedReadPort({requested.artifact_id: data})
    adapter = channel.build_v3_adapter(_native_context(read_port))
    await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-mismatched-lease",
            recipient=f"akashic:{uuid4()}",
            body="should reject",
            attachments=(requested,),
        )
    )
    assert receipt.status is V3DeliveryStatus.REJECTED
    assert receipt.error is not None and "lease.ref" in receipt.error
    assert read_port.leases and all(lease.closed for lease in read_port.leases)
    assert storage.count_durable_events("device-1") == 0
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_adapter_rejects_provider_read_failure(
    tmp_path: Path,
) -> None:
    channel, storage, manager = await _started_native_mobile_channel(tmp_path)
    requested = _native_ref(b"read-failure")

    class _FailingReadPort:
        async def acquire(self, _ref: AttachmentRef) -> _ReadLease:
            raise RuntimeError("artifact provider unavailable")

    adapter = channel.build_v3_adapter(_native_context(cast(Any, _FailingReadPort())))
    await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-read-failure",
            recipient=f"akashic:{uuid4()}",
            body="should reject",
            attachments=(requested,),
        )
    )
    assert receipt.status is V3DeliveryStatus.REJECTED
    assert receipt.error is not None and "provider unavailable" in receipt.error
    assert storage.count_durable_events("device-1") == 0
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_adapter_propagates_provider_read_cancel(
    tmp_path: Path,
) -> None:
    channel, storage, manager = await _started_native_mobile_channel(tmp_path)
    data = b"read-cancel"
    requested = _native_ref(data)

    class _CancelledReadLease(_ReadLease):
        async def read_bytes(self, *, max_bytes: int) -> bytes:
            raise asyncio.CancelledError

    class _CancelledReadPort:
        def __init__(self) -> None:
            self.lease: _CancelledReadLease | None = None

        async def acquire(self, ref: AttachmentRef) -> _CancelledReadLease:
            self.lease = _CancelledReadLease(ref, data)
            return self.lease

    read_port = _CancelledReadPort()
    adapter = channel.build_v3_adapter(_native_context(cast(Any, read_port)))
    await adapter.start()
    with pytest.raises(asyncio.CancelledError):
        await adapter.deliver(
            ProviderDeliveryRequest(
                binding_token="binding-1",
                delivery_id="delivery-read-cancel",
                recipient=f"akashic:{uuid4()}",
                body="cancel",
                attachments=(requested,),
            )
        )
    assert read_port.lease is not None and read_port.lease.closed
    assert storage.count_durable_events("device-1") == 0
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_adapter_rejects_when_device_revoked_before_commit(
    tmp_path: Path,
) -> None:
    channel, storage, manager = await _started_native_mobile_channel(tmp_path)
    original_list_active_devices = storage.list_active_devices
    calls = 0

    def list_active_devices_with_race() -> tuple[DeviceRecord, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_list_active_devices()
        return ()

    storage.list_active_devices = (  # type: ignore[method-assign]
        list_active_devices_with_race
    )
    adapter = channel.build_v3_adapter(_native_context())
    await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-device-race",
            recipient=f"akashic:{uuid4()}",
            body="no zero-recipient success",
        )
    )
    assert receipt.status is V3DeliveryStatus.REJECTED
    assert calls == 2
    assert storage.count_durable_events("device-1") == 0
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_passive_rejects_terminal_after_device_race(
    tmp_path: Path,
) -> None:
    channel, storage, manager = await _started_native_mobile_channel(tmp_path)
    original_list_active_devices = storage.list_active_devices
    calls = 0

    def list_active_devices_with_race() -> tuple[DeviceRecord, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_list_active_devices()
        return ()

    storage.list_active_devices = (  # type: ignore[method-assign]
        list_active_devices_with_race
    )
    adapter = channel.build_v3_adapter(_native_context())
    await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-passive-device-race",
            recipient=f"akashic:{uuid4()}",
            body="passive no recipient",
            commit_role=ChannelCommitRole.PASSIVE,
            control_turn_id="turn:passive-race",
        )
    )
    assert receipt.status is V3DeliveryStatus.REJECTED
    assert calls == 2
    assert storage.count_durable_events("device-1") == 0
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_passive_preserves_pending_delta_for_retry(
    tmp_path: Path,
) -> None:
    class _NoZeroEventRuntime(_Runtime):
        async def publish_event(self, **event: object) -> int:
            recipient_count = self._recipient_count()
            if recipient_count > 0:
                self.events.append(dict(event))
            return recipient_count

    runtime = _NoZeroEventRuntime(MobileRealtimeStorage(tmp_path / "mobile.db"))
    channel, storage, manager = await _started_native_mobile_channel(
        tmp_path,
        runtime=runtime,
    )
    session_id = f"akashic:{uuid4()}"
    turn_id = "turn:passive-delta-race"
    key = (session_id, turn_id)
    channel._process_turns[key] = channel_module._ProcessTurnState(
        next_ordinal=0,
        thinking_block=None,
        tool_blocks={},
        answer_segments=[],
        control_turn_id=turn_id,
    )
    async with channel._delta_locked(session_id, turn_id):
        assert not channel._accept_segment_locked(
            session_id=session_id,
            turn_id=turn_id,
            event_type="answer.delta",
            delta="pending delta",
            block_id=None,
            ordinal=None,
        )
    batch = channel._delta_batches[key]
    batch.timer.cancel()
    _ = await asyncio.gather(batch.timer, return_exceptions=True)

    original_list_active_devices = storage.list_active_devices
    calls = 0

    def list_active_devices_with_race() -> tuple[DeviceRecord, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_list_active_devices()
        return ()

    storage.list_active_devices = (  # type: ignore[method-assign]
        list_active_devices_with_race
    )
    adapter = channel.build_v3_adapter(_native_context())
    await adapter.start()
    request = ProviderDeliveryRequest(
        binding_token="binding-1",
        delivery_id="delivery-passive-delta-race",
        recipient=session_id,
        body="done",
        commit_role=ChannelCommitRole.PASSIVE,
        control_turn_id=turn_id,
    )
    rejected = await adapter.deliver(request)
    assert rejected.status is V3DeliveryStatus.REJECTED
    assert calls >= 2
    assert key in channel._delta_batches
    assert channel._delta_batches[key].segments == [
        ("answer.delta", "pending delta", None, None)
    ]
    assert runtime.events == []

    storage.list_active_devices = (  # type: ignore[method-assign]
        original_list_active_devices
    )
    delivered = await adapter.deliver(request)
    assert delivered.status is V3DeliveryStatus.DELIVERED
    assert channel._delta_batches == {}
    assert [event["event_type"] for event in runtime.events] == [
        "answer.delta",
        "message.final",
    ]
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_adapter_cleans_candidates_on_cancelled_precommit(
    tmp_path: Path,
) -> None:
    data = b"cancelled-before-commit"
    requested = _native_ref(data)

    class _PrecommitFailRuntime(_Runtime):
        def __init__(self, storage: MobileRealtimeStorage) -> None:
            super().__init__(storage)
            self.publish_started = asyncio.Event()
            self.publish_release = asyncio.Event()

        async def publish_event_with_outbound_attachments_result(
            self,
            **_kwargs: object,
        ) -> tuple[tuple[AttachmentRecord, ...], int]:
            self.publish_started.set()
            await self.publish_release.wait()
            raise MobileStorageError("durable event 没有可提交的目标设备")

    runtime = _PrecommitFailRuntime(
        MobileRealtimeStorage(tmp_path / "mobile.db")
    )
    channel, storage, manager = await _started_native_mobile_channel(
        tmp_path,
        runtime=runtime,
    )
    read_port = _ReadPort({requested.artifact_id: data})
    adapter = channel.build_v3_adapter(_native_context(read_port))
    await adapter.start()
    delivery = asyncio.create_task(
        adapter.deliver(
            ProviderDeliveryRequest(
                binding_token="binding-1",
                delivery_id="delivery-cancelled-precommit",
                recipient=f"akashic:{uuid4()}",
                body="cancel",
                attachments=(requested,),
            )
        )
    )
    await runtime.publish_started.wait()
    delivery.cancel()
    runtime.publish_release.set()
    with pytest.raises(asyncio.CancelledError):
        await delivery
    assert not [path for path in (tmp_path / "uploads").rglob("*") if path.is_file()]
    assert storage.count_durable_events("device-1") == 0
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_adapter_retains_candidates_after_unknown_effect(
    tmp_path: Path,
) -> None:
    data = b"unknown-after-commit"
    requested = _native_ref(data)

    class _CommitThenRaiseRuntime(_Runtime):
        async def publish_event_with_outbound_attachments_result(
            self,
            **kwargs: object,
        ) -> tuple[tuple[AttachmentRecord, ...], int]:
            _resolved, _count = (
                await super().publish_event_with_outbound_attachments_result(
                    candidates=cast(
                        tuple[AttachmentRecord, ...],
                        kwargs["candidates"],
                    ),
                    payload_builder=cast(Any, kwargs["payload_builder"]),
                    session_id=cast(str, kwargs["session_id"]),
                )
            )
            raise RuntimeError("commit acknowledgement lost")

    runtime = _CommitThenRaiseRuntime(
        MobileRealtimeStorage(tmp_path / "mobile.db")
    )
    channel, storage, manager = await _started_native_mobile_channel(
        tmp_path,
        runtime=runtime,
    )
    read_port = _ReadPort({requested.artifact_id: data})
    adapter = channel.build_v3_adapter(_native_context(read_port))
    await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-unknown-after-commit",
            recipient=f"akashic:{uuid4()}",
            body="unknown",
            attachments=(requested,),
        )
    )
    assert receipt.status is V3DeliveryStatus.UNKNOWN
    assert storage.count_durable_events("device-1") == 1
    assert [path for path in (tmp_path / "uploads").rglob("*") if path.is_file()]
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_native_v3_mobile_passive_keeps_file_after_post_commit_db_error(
    tmp_path: Path,
) -> None:
    data = b"passive-post-commit-error"
    requested = _native_ref(data)
    channel, storage, manager = await _started_native_mobile_channel(tmp_path)
    original_create = storage.create_or_read_outbound_attachments

    def commit_then_raise(
        records: tuple[AttachmentRecord, ...],
        *,
        message_id: str | None = None,
    ) -> tuple[AttachmentRecord, ...]:
        _ = original_create(records, message_id=message_id)
        raise sqlite3.OperationalError("post-commit acknowledgement lost")

    storage.create_or_read_outbound_attachments = (  # type: ignore[method-assign]
        commit_then_raise
    )
    read_port = _ReadPort({requested.artifact_id: data})
    adapter = channel.build_v3_adapter(_native_context(read_port))
    await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-passive-post-commit-error",
            recipient=f"akashic:{uuid4()}",
            body="unknown registration result",
            attachments=(requested,),
            session_message_id="assistant-message-1",
            commit_role=ChannelCommitRole.PASSIVE,
        )
    )
    assert receipt.status is V3DeliveryStatus.UNKNOWN
    row = storage._db.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT local_path FROM mobile_attachments WHERE direction = 'outbound'"
    ).fetchone()
    assert row is not None
    assert Path(row[0]).is_file()
    await adapter.stop()
    await channel.stop()
    manager.close()
    storage.close()


class _RuntimeInspection:
    def list_documents(self) -> dict[str, object]:
        return {"items": [{"id": "memory"}]}

    def get_document(self, document_id: str) -> dict[str, object]:
        return {"id": document_id, "markdown": "# Memory"}


class _ModelRegistry:
    async def refresh(self) -> SimpleNamespace:
        return SimpleNamespace(
            generation_id=3,
            role_runtime_ids={"default": "model-a"},
        )

    def list_runtimes(self) -> list[dict[str, object]]:
        return [
            {
                "id": "model-a",
                "provider": "openai",
                "catalogProvider": "openai",
                "model": "gpt-test",
                "reasoningEffort": "medium",
                "supportedReasoningEfforts": ["low", "medium", "high"],
                "sourceId": "source-a",
                "sourceName": "OpenAI",
                "contextWindow": 128_000,
                "maxOutputTokens": 8_192,
                "inputModalities": ["text", "image"],
                "capabilitySource": "test",
                "capabilitySources": {},
                "roles": ["default", "agent"],
            }
        ]


def _register_device(storage: MobileRealtimeStorage, device_id: str) -> None:
    storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=f"test-public-key:{device_id}",
            display_name=device_id,
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )


@pytest.mark.asyncio
async def test_runtime_document_commands_use_bound_read_service(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))
    channel.bind_runtime_inspection(
        cast(RuntimeInspectionService, _RuntimeInspection())
    )

    listed = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            command_type="runtime.document.list",
        ),
    )
    detail = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
            command_type="runtime.document.get",
            payload={"document_id": "memory"},
        ),
    )

    assert listed.type == "runtime.document.list.ok"
    assert listed.payload["items"] == [{"id": "memory"}]
    assert detail.type == "runtime.document.get.ok"
    assert detail.payload["markdown"] == "# Memory"
    storage.close()


@pytest.mark.asyncio
async def test_model_catalog_returns_bound_registry_and_session_selection(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    session = manager.get_or_create(session_id)
    session.metadata["model_selection"] = {
        "schema_version": 1,
        "model_ref": "model-a",
        "reasoning_effort": "high",
    }
    manager.save(session)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))
    channel.bind_model_registry(cast(Any, _ModelRegistry()))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )

    reply = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            command_type="model.catalog.get",
            session_id=session_id,
        ),
    )

    assert reply.type == "model.catalog.get.ok"
    assert reply.session_id == session_id
    assert reply.payload["generation_id"] == 3
    assert reply.payload["default_runtime"] == "model-a"
    assert reply.payload["selected_runtime_id"] == "model-a"
    assert reply.payload["selected_reasoning_effort"] == "high"
    assert reply.payload["runtimes"] == [
        {
            "id": "model-a",
            "provider": "openai",
            "model": "gpt-test",
            "sourceId": "source-a",
            "sourceName": "OpenAI",
            "reasoningEffort": "medium",
            "supportedReasoningEfforts": ["low", "medium", "high"],
            "roles": ["default", "agent"],
            "contextWindow": 128_000,
            "inputModalities": ["text", "image"],
        }
    ]
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_device_update_refreshes_capabilities(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))

    reply = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            command_type="device.update",
            payload={"capabilities": ["chat", "turn-output-completed-v1"]},
        ),
    )

    assert reply.type == "device.update.ok"
    assert storage.read_device(device_id).capabilities == (
        "chat",
        "turn-output-completed-v1",
    )
    storage.close()


@pytest.mark.asyncio
async def test_device_update_rejects_oversized_capability_set(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))

    too_many = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            command_type="device.update",
            payload={"capabilities": [f"cap-{i}" for i in range(129)]},
        ),
    )
    assert too_many.type == "device.update.error"
    assert too_many.payload["code"] == "invalid_payload"

    too_long = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
            command_type="device.update",
            payload={"capabilities": ["a" * 513]},
        ),
    )
    assert too_long.type == "device.update.error"
    assert too_long.payload["code"] == "invalid_payload"
    storage.close()


def test_mobile_tool_arguments_are_bounded_for_phone_storage() -> None:
    nested: dict[str, object] = {"value": "visible"}
    for _ in range(7):
        nested = {"child": nested}

    projected = channel_module._mobile_tool_arguments(
        {
            "long_text": "x" * 2_001,
            "items": list(range(65)),
            "nested": nested,
        }
    )

    assert projected["long_text"] == "x" * 2_000 + "…"
    assert cast(list[object], projected["items"])[-1] == "[已截断]"
    level = cast(dict[str, object], projected["nested"])
    for _ in range(4):
        level = cast(dict[str, object], level["child"])
    assert level["child"] == "[已截断]"

    sensitive = channel_module._mobile_tool_arguments(
        {
            "openai_api_key": "LEAK",
            "x-api-key": "LEAK",
            "client_credentials_file": "/tmp/credentials.json",
            "command": 'curl -H "Authorization: Bearer sk-private-value"',
            "env_command": "GH_TOKEN=ghp_private AWS_SECRET_ACCESS_KEY=private tool",
            "argv_header": ["curl", "-H", "Authorization: Bearer sk-private-value"],
            "argv_flag": ["curl", "--api-key", "sk-private-value"],
            "argv_assignment": ["tool", "--token=ghp_private"],
        }
    )
    assert sensitive == {
        "openai_api_key": "[已隐藏]",
        "x-api-key": "[已隐藏]",
        "client_credentials_file": "[已隐藏]",
        "command": "[已隐藏]",
        "env_command": "[已隐藏]",
        "argv_header": ["curl", "-H", "[已隐藏]"],
        "argv_flag": ["curl", "--api-key", "[已隐藏]"],
        "argv_assignment": ["tool", "[已隐藏]"],
    }
    assert channel_module._mobile_tool_arguments(
        {"note": "token budget is 1000; keep this text"}
    ) == {"note": "token budget is 1000; keep this text"}

    emoji_arguments = channel_module._mobile_tool_arguments(
        {f"field_{index}": "😀" * 2_000 for index in range(64)}
    )
    assert (
        channel_module._mobile_tool_argument_encoded_size(emoji_arguments) <= 8 * 1024
    )
    encoded = gateway_module._encode_stored_event(
        event_id="01J00000000000000000000000",
        event_type="react.tool.started",
        session_id="akashic:test",
        turn_id="turn-1",
        payload={
            "call_id": "call-1",
            "block_id": "tool:call-1",
            "ordinal": 0,
            "tool_name": "shell",
            "arguments": emoji_arguments,
        },
    )
    assert len(encoded.encode("utf-8")) < 256 * 1024


def test_mobile_history_projects_proactive_delivery_identity() -> None:
    projected = channel_module._mobile_history_item(
        {
            "id": "akashic:test:1",
            "session_key": "akashic:test",
            "seq": 1,
            "role": "assistant",
            "content": "主动提醒",
            "timestamp": "2026-07-19T00:00:00+00:00",
            "proactive": True,
            "delivery_id": "delivery-1",
        }
    )

    assert projected["extra"] == {
        "proactive": True,
        "delivery_id": "delivery-1",
    }


def test_mobile_history_tool_arguments_fit_real_event_frame() -> None:
    calls = [
        {
            "call_id": f"call-{index}",
            "name": "shell",
            "status": "success",
            "arguments": channel_module._mobile_tool_arguments(
                {"command": "😀" * 2_000},
                max_bytes=8 * 1024,
            ),
            "result_preview": "ok",
        }
        for index in range(40)
    ]
    payload: dict[str, object] = {
        "items": [
            {
                "id": "message-1",
                "session_key": "akashic:test",
                "seq": 1,
                "role": "assistant",
                "content": "done",
                "tool_chain": [{"calls": calls}],
                "extra": {},
                "ts": "2026-07-16T00:00:00Z",
            }
        ],
        "total": 1,
        "page": 1,
        "page_size": 50,
    }

    channel_module._fit_mobile_history_payload(payload)
    encoded = gateway_module._encode_stored_event(
        event_id="01J00000000000000000000000",
        event_type="history.page",
        session_id="akashic:test",
        payload=payload,
    )

    assert len(encoded.encode("utf-8")) < 256 * 1024
    assert any("arguments" not in call for call in calls)


def test_mobile_history_tool_descriptions_fit_real_event_frame() -> None:
    calls = [
        {
            "call_id": f"call-{index}",
            "name": "shell",
            "status": "success",
            "description": "😀" * 2_000,
            "arguments": {"description": "😀" * 2_000},
            "result_preview": "ok",
        }
        for index in range(40)
    ]
    payload: dict[str, object] = {
        "items": [
            {
                "id": "message-1",
                "session_key": "akashic:test",
                "seq": 1,
                "role": "assistant",
                "content": "done",
                "tool_chain": [{"calls": calls}],
                "extra": {},
                "ts": "2026-07-16T00:00:00Z",
            }
        ],
        "total": 1,
        "page": 1,
        "page_size": 50,
    }

    channel_module._fit_mobile_history_payload(payload)
    encoded = gateway_module._encode_stored_event(
        event_id="01J00000000000000000000000",
        event_type="history.page",
        session_id="akashic:test",
        payload=payload,
    )

    assert len(encoded.encode("utf-8")) < 256 * 1024
    assert any("description" not in call for call in calls)


def test_mobile_history_cursor_shrinks_page_before_tool_details() -> None:
    items = [
        {
            "id": f"message-{seq}",
            "session_key": "akashic:test",
            "seq": seq,
            "role": "assistant",
            "content": "done",
            "tool_chain": [
                {
                    "calls": [
                        {
                            "call_id": f"call-{seq}",
                            "name": "shell",
                            "status": "success",
                            "result_preview": "结" * 80_000,
                        }
                    ]
                }
            ],
            "extra": {},
            "ts": "2026-08-05T00:00:00Z",
        }
        for seq in (10, 11)
    ]
    payload: dict[str, object] = {
        "items": items,
        "total": 2,
        "page_size": 50,
        "content_ref_version": 1,
        "after_seq": 9,
        "next_after_seq": 11,
        "snapshot_max_seq": 11,
        "has_more": False,
    }

    channel_module._fit_mobile_history_payload(payload, allow_content_refs=True)

    assert payload["items"] == [items[0]]
    assert payload["next_after_seq"] == 10
    assert payload["has_more"] is True
    assert items[0]["tool_chain"][0]["calls"][0]["result_preview"].startswith("结")


def test_mobile_history_marks_oversized_result_previews() -> None:
    calls = [
        {
            "call_id": f"call-{index}",
            "name": "shell",
            "status": "success",
            "result_preview": "结果" * 1_000,
        }
        for index in range(80)
    ]
    payload: dict[str, object] = {
        "items": [
            {
                "id": "message-1",
                "session_key": "akashic:test",
                "seq": 1,
                "role": "assistant",
                "content": "done",
                "tool_chain": [
                    {
                        "reasoning_content": "思考" * 30_000,
                        "calls": calls,
                    }
                ],
                "extra": {},
                "ts": "2026-08-05T00:00:00Z",
            }
        ],
        "total": 1,
        "page_size": 50,
        "content_ref_version": 1,
        "after_seq": 0,
        "next_after_seq": 1,
        "snapshot_max_seq": 1,
        "has_more": False,
    }

    channel_module._fit_mobile_history_payload(payload, allow_content_refs=True)

    assert any(call["result_preview"] == "[历史同步时已省略过长详情]" for call in calls)
    assert channel_module._mobile_tool_argument_encoded_size(payload) <= 240 * 1024


def _message_frame(
    *,
    frame_id: str,
    session_id: str,
    epoch: int = 1,
    reply_to: dict[str, object] | None = None,
    text: str = "你好",
    model_runtime_id: str | None = None,
    model_reasoning_effort: str | None = None,
    media_refs: list[str] | None = None,
) -> MessageSendCommand:
    frame = parse_frame(
        json.dumps(
            {
                "v": 1,
                "kind": "command",
                "type": "message.send",
                "id": frame_id,
                "connection_epoch": epoch,
                "session_id": session_id,
                "payload": {
                    "client_message_id": frame_id,
                    "session_id": session_id,
                    "text": text,
                    "media_refs": media_refs or [],
                    "client_created_at": datetime.now(timezone.utc).isoformat(),
                    **({"reply_to": reply_to} if reply_to is not None else {}),
                    **(
                        {"model_runtime_id": model_runtime_id}
                        if model_runtime_id is not None
                        else {}
                    ),
                    **(
                        {"model_reasoning_effort": model_reasoning_effort}
                        if model_reasoning_effort is not None
                        else {}
                    ),
                },
            }
        )
    )
    assert isinstance(frame, MessageSendCommand)
    return frame


def _generic_frame(
    *,
    frame_id: str,
    command_type: str,
    session_id: str | None = None,
    turn_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> GenericCommand:
    raw: dict[str, object] = {
        "v": 1,
        "kind": "command",
        "type": command_type,
        "id": frame_id,
        "connection_epoch": 1,
        "payload": payload or {},
    }
    if session_id is not None:
        raw["session_id"] = session_id
    if turn_id is not None:
        raw["turn_id"] = turn_id
    frame = parse_frame(json.dumps(raw))
    assert isinstance(frame, GenericCommand)
    return frame


@pytest.mark.asyncio
async def test_message_send_is_idempotent_and_session_is_shared_between_devices(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    first_device = uuid4().hex
    second_device = uuid4().hex
    _register_device(storage, first_device)
    _register_device(storage, second_device)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    manager = SessionManager(tmp_path / "workspace")
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    original = _message_frame(
        frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        session_id=session_id,
    )

    first = await channel.handle_command(device_id=first_device, frame=original)
    duplicate = await channel.handle_command(
        device_id=first_device,
        frame=original.model_copy(update={"connection_epoch": 2}),
    )
    shared = await channel.handle_command(
        device_id=second_device,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
            session_id=session_id,
        ),
    )
    mismatched = await channel.handle_command(
        device_id=first_device,
        frame=original.model_copy(update={"id": "01ARZ3NDEKTSV4RRFFQ69G5FAZ"}),
    )

    assert first.type == duplicate.type
    assert first.payload == duplicate.payload
    assert first.session_id == duplicate.session_id
    assert first.turn_id == duplicate.turn_id
    assert first.replayed is False
    assert duplicate.replayed is True
    assert first.type == "message.send.ok"
    assert len(bus.inbound) == 2
    assert shared.type == "message.send.ok"
    assert mismatched.type == "message.send.error"
    assert mismatched.payload["code"] == "client_message_id_mismatch"
    assert len(bus.inbound) == 2
    assert all(
        cast(RawInbound, item).message.metadata["require_existing_session"] is True
        for item in bus.inbound
    )
    assert manager.delete_session(session_id)
    with pytest.raises(KeyError, match="session 不存在"):
        manager.get_existing(session_id)
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_mobile_message_adopts_finalized_upload_before_bus_admission(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    workspace = tmp_path / "workspace"
    manager = SessionManager(workspace)
    legacy_store = AttachmentStore(tmp_path / "uploads")
    legacy_store.root.mkdir(parents=True)
    content = b"mobile-c23"
    source = legacy_store.root / "upload.png"
    source.write_bytes(content)
    session_id = f"akashic:{uuid4()}"
    storage.create_attachment(
        AttachmentRecord(
            attachment_id="upload-c23",
            device_id=device_id,
            session_id=session_id,
            direction="upload",
            filename="upload.png",
            content_type="image/png",
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            local_path=str(source),
            transferred_bytes=len(content),
            state="ready",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )
    artifact_store = ChannelAttachmentArtifactStore(
        workspace=workspace,
        session_store=manager.control_store,
    )
    original_adopt = artifact_store.adopt_file_with_artifact_id

    async def recorded_adopt(*args: object, **kwargs: object) -> AttachmentRef:
        bus.order.append("artifact")
        return await original_adopt(*args, **kwargs)  # type: ignore[arg-type]

    artifact_store.adopt_file_with_artifact_id = recorded_adopt  # type: ignore[method-assign]
    channel.bind_channel_attachment_store(artifact_store)
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=legacy_store,
            ),
        )
    )

    reply = await channel.handle_command(
        device_id=device_id,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAX",
            session_id=session_id,
            media_refs=["upload-c23"],
        ),
    )

    assert reply.type == "message.send.ok"
    assert len(bus.inbound) == 1
    inbound = cast(RawInbound, bus.inbound[0])
    artifact_ids = cast(list[str], inbound.message.metadata["attachment_ids"])
    assert tuple(ref.artifact_id for ref in inbound.message.attachments) == artifact_ids
    assert len(artifact_ids) == 1
    imported = storage.list_attachment_imports(
        session_id=session_id,
        client_message_id="01ARZ3NDEKTSV4RRFFQ69G5FAX",
    )
    assert imported[0].phase == "artifact_committed"
    assert imported[0].artifact_id == artifact_ids[0]
    assert bus.order == ["reserve", "artifact", "admit"]

    # 1. artifact read lease 可读，且只暴露进程内 fd 路径。
    ref = artifact_store.resolve_refs(tuple(artifact_ids))[0]
    lease = await artifact_store.acquire(ref)
    try:
        assert await lease.read_bytes(max_bytes=1024) == content
    finally:
        await lease.aclose()

    # 2. Session commit 与 Mobile import phase 最终收束到同一 artifact ID。
    session = manager.get_existing(session_id)
    pending: dict[str, object] = {
        "role": "user",
        "content": "with attachment",
        "client_message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAX",
        "attachment_ids": list(artifact_ids),
    }
    await manager.append_messages(session, [pending])
    await channel.stop()
    restarted = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    restarted.bind_channel_attachment_store(artifact_store)
    await restarted.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=legacy_store,
            ),
        )
    )
    assert storage.list_attachment_imports(
        session_id=session_id,
        client_message_id="01ARZ3NDEKTSV4RRFFQ69G5FAX",
    )[0].phase == "message_bound"

    await restarted.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_message_send_does_not_recreate_a_deleted_claimed_session(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    manager = SessionManager(tmp_path / "workspace")
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    manager.save(manager.get_or_create(session_id))
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    assert manager.delete_session(session_id)

    reply = await channel.handle_command(
        device_id=device_id,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FBA",
            session_id=session_id,
        ),
    )

    assert reply.type == "message.send.error"
    assert reply.payload["code"] == "session_not_found"
    assert not manager.session_exists(session_id)
    assert bus.inbound == []
    storage.close()


@pytest.mark.asyncio
async def test_message_send_releases_admission_when_bus_publish_fails(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    manager = SessionManager(tmp_path / "workspace")
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_FailingBus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"

    with pytest.raises(RuntimeError, match="bus unavailable"):
        await channel.handle_command(
            device_id=device_id,
            frame=_message_frame(
                frame_id="01ARZ3NDEKTSV4RRFFQ69G5FBA",
                session_id=session_id,
            ),
        )

    assert manager.delete_session(session_id)
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_claimed_message_admission_does_not_recreate_after_exists_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    manager = SessionManager(tmp_path / "workspace")
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    manager.save(manager.get_or_create(session_id))
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    original_exists = manager.session_exists

    def delete_after_exists_check(key: str) -> bool:
        exists = original_exists(key)
        assert manager.delete_session(key)
        return exists

    monkeypatch.setattr(manager, "session_exists", delete_after_exists_check)

    reply = await channel.handle_command(
        device_id=device_id,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FBA",
            session_id=session_id,
        ),
    )

    assert reply.type == "message.send.ok"
    assert not original_exists(session_id)
    assert len(bus.inbound) == 1
    with pytest.raises(KeyError, match="session 不存在"):
        manager.get_existing(session_id)
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_message_send_recovers_processing_receipt_from_persisted_user(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    manager = SessionManager(tmp_path / "workspace")
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    frame = _message_frame(
        frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        session_id=session_id,
    )
    _, created = storage.reserve_command(
        device_id=device_id,
        command_id=frame.id,
        command_type=frame.type,
        request_hash=channel_module._command_hash(frame),
        created_at=datetime.now(timezone.utc),
    )
    assert created
    session = manager.get_or_create(session_id)
    session.add_message(
        "user",
        "你好",
        client_message_id=frame.payload.client_message_id,
    )
    manager.save(session)

    recovered = await channel.handle_command(device_id=device_id, frame=frame)
    replayed = await channel.handle_command(
        device_id=device_id,
        frame=frame.model_copy(update={"connection_epoch": 2}),
    )

    assert recovered == replayed
    assert recovered.type == "message.send.ok"
    assert recovered.payload == {
        "accepted": True,
        "client_message_id": frame.payload.client_message_id,
    }
    assert bus.inbound == []
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_message_send_keeps_unknown_outcome_without_persisted_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    frame = _message_frame(
        frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        session_id=session_id,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute_slowly(
        *, device_id: str, frame: object
    ) -> channel_module.CommandReply:
        started.set()
        await release.wait()
        return channel_module.CommandReply(
            type="message.send.ok",
            payload={"accepted": True},
            session_id=session_id,
        )

    monkeypatch.setattr(channel, "_execute_command", execute_slowly)
    original = asyncio.create_task(
        channel.handle_command(device_id=device_id, frame=frame)
    )
    await started.wait()

    result = await channel.handle_command(device_id=device_id, frame=frame)

    assert result.type == "message.send.error"
    assert result.payload["code"] == "command_in_progress"
    assert bus.inbound == []
    release.set()
    completed = await original
    assert completed.type == "message.send.ok"
    storage.close()


@pytest.mark.asyncio
async def test_duplicate_processing_non_message_keeps_active_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))
    frame = _generic_frame(
        frame_id="01ARZ3NDEKTSV4RRFFQ69G5FB3",
        command_type="ping",
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute_slowly(
        *, device_id: str, frame: object
    ) -> channel_module.CommandReply:
        started.set()
        await release.wait()
        return channel_module.CommandReply(type="ping.ok", payload={})

    monkeypatch.setattr(channel, "_execute_command", execute_slowly)
    original = asyncio.create_task(
        channel.handle_command(device_id=device_id, frame=frame)
    )
    await started.wait()
    duplicate = await channel.handle_command(device_id=device_id, frame=frame)
    assert duplicate.payload["code"] == "command_in_progress"
    release.set()
    assert (await original).type == "ping.ok"
    replay = await channel.handle_command(device_id=device_id, frame=frame)
    assert replay.type == "ping.ok"
    storage.close()


@pytest.mark.asyncio
async def test_message_send_keeps_current_owner_when_receipt_completion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    frame = _message_frame(
        frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        session_id=f"akashic:{uuid4()}",
    )

    def fail_completion(**_: object) -> object:
        raise OSError("receipt write failed")

    monkeypatch.setattr(storage, "complete_command", fail_completion)
    with pytest.raises(OSError, match="receipt write failed"):
        await channel.handle_command(device_id=device_id, frame=frame)

    assert len(bus.inbound) == 1
    replay = await channel.handle_command(device_id=device_id, frame=frame)
    assert replay.type == "message.send.error"
    assert replay.payload["code"] == "command_outcome_unknown"
    assert len(bus.inbound) == 1
    storage.close()


@pytest.mark.asyncio
async def test_message_send_marks_prestart_processing_receipt_retryable(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    session_id = f"akashic:{uuid4()}"
    frame = _message_frame(
        frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        session_id=session_id,
    )
    _, created = storage.reserve_command(
        device_id=device_id,
        command_id=frame.id,
        command_type=frame.type,
        request_hash=channel_module._command_hash(frame),
        created_at=datetime.now(timezone.utc),
    )
    assert created
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )

    result = await channel.handle_command(device_id=device_id, frame=frame)
    replayed = await channel.handle_command(device_id=device_id, frame=frame)

    assert result == replayed
    assert result.type == "message.send.error"
    assert result.payload["code"] == "command_interrupted"
    storage.close()


@pytest.mark.asyncio
async def test_message_send_pending_handoff_stays_in_progress_until_user_reconciles(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    frame = _message_frame(
        frame_id="01ARZ3NDEKSTV4RRFFQ69G5FAY",
        session_id=session_id,
    )
    _, created = storage.reserve_command(
        device_id=device_id,
        command_id=frame.id,
        command_type=frame.type,
        request_hash=channel_module._command_hash(frame),
        created_at=datetime.now(timezone.utc),
    )
    assert created
    bus = _Bus()
    bus.pending_handoff = True
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )

    pending = await channel.handle_command(device_id=device_id, frame=frame)
    assert pending.type == "message.send.error"
    assert pending.payload["code"] == "command_in_progress"
    receipt = storage._db.execute(
        "SELECT status FROM mobile_command_receipts WHERE device_id = ? AND command_id = ?",
        (device_id, frame.id),
    ).fetchone()
    assert receipt is not None and receipt[0] == "processing"

    session = manager.get_or_create(session_id)
    session.add_message("user", frame.payload.text, client_message_id=frame.id)
    manager.save(session)
    bus.pending_handoff = False
    completed = await channel.handle_command(device_id=device_id, frame=frame)
    assert completed.type == "message.send.ok"
    receipt = storage._db.execute(
        "SELECT status FROM mobile_command_receipts WHERE device_id = ? AND command_id = ?",
        (device_id, frame.id),
    ).fetchone()
    assert receipt is not None and receipt[0] == "completed"
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_message_send_resolves_reply_into_agent_context_and_metadata(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    session = manager.get_or_create(session_id)
    target_content = "第一行\n第二行\n" + "长" * 600
    target = session.add_message("assistant", target_content)
    manager.save(session)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )

    reply = await channel.handle_command(
        device_id=device_id,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            session_id=session_id,
            model_runtime_id="model-a",
            model_reasoning_effort="high",
            reply_to={
                "message_id": target["id"],
            },
        ),
    )

    assert reply.type == "message.send.ok"
    inbound = cast(RawInbound, bus.inbound[0]).message
    assert channel_module._normalize_v3_content(
        f"被回复消息（来自 Akashic）：\n{target_content}"
    ) in inbound.content
    assert inbound.content.endswith("【你当前新消息】\u2028你好")
    assert inbound.metadata["display_content"] == "你好"
    assert inbound.metadata["reply_to_message_id"] == target["id"]
    assert inbound.metadata["reply_role"] == "assistant"
    assert inbound.metadata["reply_preview"] == " ".join(target_content.split())[:512]
    assert inbound.metadata["require_existing_session"] is True
    assert inbound.metadata["model_runtime_id"] == "model-a"
    assert inbound.metadata["model_reasoning_effort"] == "high"

    user_target = session.add_message(
        "user",
        "之前的问题",
        client_message_id="01ARZ3NDEKTSV4RRFFQ69G5FAW",
    )
    manager.save(session)
    second = await channel.handle_command(
        device_id=device_id,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAX",
            session_id=session_id,
            reply_to={
                "client_message_id": user_target["client_message_id"],
            },
        ),
    )
    assert second.type == "message.send.ok"
    second_inbound = cast(RawInbound, bus.inbound[1]).message
    assert second_inbound.metadata["require_existing_session"] is True
    assert second_inbound.metadata["reply_to_message_id"] == user_target["id"]
    assert "被回复消息（来自 你）：\u2028之前的问题" in second_inbound.content

    media_target = session.add_message(
        "user",
        "",
        media=["/tmp/photo.png"],
        client_message_id="01ARZ3NDEKTSV4RRFFQ69G5FAY",
    )
    manager.save(session)
    third = await channel.handle_command(
        device_id=device_id,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAZ",
            session_id=session_id,
            reply_to={
                "client_message_id": media_target["client_message_id"],
            },
        ),
    )
    assert third.type == "message.send.ok"
    third_inbound = cast(RawInbound, bus.inbound[2]).message
    assert third_inbound.metadata["reply_preview"] == "[附件]"
    assert "被回复消息（来自 你）：\u2028[附件]" in third_inbound.content

    proactive_target = session.add_message(
        "assistant",
        "尚未同步历史的主动消息",
        proactive=True,
        delivery_id="delivery-1",
    )
    manager.save(session)
    fourth = await channel.handle_command(
        device_id=device_id,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FB0",
            session_id=session_id,
            reply_to={"delivery_id": "delivery-1"},
        ),
    )
    assert fourth.type == "message.send.ok"
    fourth_inbound = cast(RawInbound, bus.inbound[3]).message
    assert fourth_inbound.metadata["reply_to_message_id"] == proactive_target["id"]
    assert (
        "被回复消息（来自 Akashic）：\u2028尚未同步历史的主动消息"
        in fourth_inbound.content
    )
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_message_send_rejects_reply_from_another_session(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    other = manager.get_or_create(f"akashic:{uuid4()}")
    target = other.add_message("assistant", "其他会话")
    manager.save(other)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )

    result = await channel.handle_command(
        device_id=device_id,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            session_id=session_id,
            reply_to={
                "message_id": target["id"],
            },
        ),
    )

    assert result.type == "message.send.error"
    assert result.payload["code"] == "reply_target_session_mismatch"
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_session_list_and_history_sync_publish_all_mobile_sessions(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    other_device_id = uuid4().hex
    _register_device(storage, device_id)
    _register_device(storage, other_device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    push = _PushTool()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=push,
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    storage.claim_session(
        device_id=other_device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    session = manager.get_or_create(session_id)
    media_path = tmp_path / "answer.png"
    media_path.write_bytes(b"not-a-real-png-but-stable")
    session.add_message(
        "user",
        "恢复这段对话",
        control_turn_id="turn-history",
        llm_context_frame="private context",
        client_message_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        reply_to_message_id=f"{session_id}:0",
        reply_role="assistant",
        reply_preview="更早的回答",
    )
    session.add_message(
        "assistant",
        "历史回答",
        control_turn_id="turn-history",
        media=[str(media_path)],
        reasoning_content="历史思考",
        tool_chain=[
            {
                "text": "先检查",
                "calls": [
                    {
                        "call_id": "call-1",
                        "name": "shell",
                        "status": "success",
                        "arguments": {
                            "description": "读取状态",
                            "path": "/sandbox/workspace/status.json",
                            "secret": "hidden",
                            "nested": {"access_token": "hidden", "page": 2},
                        },
                        "result": "完成",
                    }
                ],
            }
        ],
    )
    manager.save(session)
    web_session = manager.get_or_create(f"web:{uuid4()}")
    web_session.add_message("user", "不要同步 Web 会话")
    manager.save(web_session)
    empty_session_id = f"akashic:{uuid4()}"
    manager.get_or_create(empty_session_id)

    listed = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAX",
            command_type="session.list",
        ),
    )
    history = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAY",
            command_type="history.get",
            session_id=session_id,
            payload={"page": 1, "page_size": 10},
        ),
    )

    assert listed.type == "session.list.ok"
    assert listed.payload["total"] == 2
    session_event = runtime.events[-2]
    assert session_event["event_type"] == "session.list"
    session_payload = cast(dict[str, object], session_event["payload"])
    session_items = cast(list[dict[str, object]], session_payload["items"])
    assert len(session_items) == 2
    session_items_by_id = {str(item["session_id"]): item for item in session_items}
    assert session_items_by_id[session_id]["title"] == "恢复这段对话"
    assert session_items_by_id[empty_session_id]["title"] == "新对话"
    assert str(session_items_by_id[session_id]["updated_at"]).endswith("Z")
    assert "snapshot_max_seq" not in session_items_by_id[session_id]
    assert history.type == "history.get.ok"
    history_event = runtime.events[-1]
    history_payload = cast(dict[str, object], history_event["payload"])
    assert history_payload["title"] == "恢复这段对话"
    history_items = cast(list[dict[str, object]], history_payload["items"])
    assert history_items[0]["extra"] == {}
    assert history_items[0]["client_message_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert history_items[0]["reply_to_message_id"] == f"{session_id}:0"
    assert history_items[0]["reply_role"] == "assistant"
    assert history_items[0]["reply_preview"] == "更早的回答"
    assert "llm_context_frame" not in history_items[0]
    assert history_items[1]["extra"] == {
        "reasoning_content": "历史思考",
        "control_turn_id": "turn-history",
    }

    versioned = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAZ",
            command_type="session.list",
            payload={"history_snapshot_version": 1},
        ),
    )
    assert versioned.type == "session.list.ok"
    versioned_items = cast(
        list[dict[str, object]],
        cast(dict[str, object], runtime.events[-1]["payload"])["items"],
    )
    versioned_by_id = {str(item["session_id"]): item for item in versioned_items}
    assert versioned_by_id[session_id]["snapshot_max_seq"] == 1
    assert versioned_by_id[empty_session_id]["snapshot_max_seq"] == -1
    invalid_version = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FB0",
            command_type="session.list",
            payload={"history_snapshot_version": 2},
        ),
    )
    assert invalid_version.type == "session.list.error"
    assert invalid_version.payload["code"] == "invalid_payload"
    tool_chain = cast(list[dict[str, object]], history_items[1]["tool_chain"])
    calls = cast(list[dict[str, object]], tool_chain[0]["calls"])
    assert calls[0]["description"] == "读取状态"
    assert "secret" not in calls[0]
    projected_arguments = cast(dict[str, object], calls[0]["arguments"])
    assert projected_arguments["path"] == "/sandbox/workspace/status.json"
    assert projected_arguments["secret"] == "[已隐藏]"
    assert projected_arguments["nested"] == {"access_token": "[已隐藏]", "page": 2}
    attachments = cast(list[dict[str, object]], history_items[1]["attachments"])
    assert len(attachments) == 1
    assert attachments[0]["filename"] == "answer.png"
    assert "local_path" not in attachments[0]

    turn_id = uuid4().hex
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="再次生成",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
    )
    session.add_message("assistant", "实时回答", media=[str(media_path)])
    manager.save(session)
    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="实时回答",
            media=[str(media_path)],
            control_turn_id=turn_id,
            session_message_id=str(session.messages[-1]["id"]),
        )
    )
    live_payload = cast(dict[str, object], runtime.events[-1]["payload"])
    assert live_payload["message_id"] == session.messages[-1]["id"]
    live = cast(list[dict[str, object]], live_payload["attachments"])
    assert live[0]["attachment_id"] == attachments[0]["attachment_id"]

    media_path.unlink()
    _ = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FA0",
            command_type="history.get",
            session_id=session_id,
            payload={"page": 1, "page_size": 10},
        ),
    )
    restored_payload = cast(dict[str, object], runtime.events[-1]["payload"])
    restored_items = cast(list[dict[str, object]], restored_payload["items"])
    restored = cast(list[dict[str, object]], restored_items[-1]["attachments"])
    assert restored[0]["attachment_id"] == live[0]["attachment_id"]

    session.add_message("assistant", "旧媒体已失效", media=[str(media_path)])
    manager.save(session)
    _ = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FA1",
            command_type="history.get",
            session_id=session_id,
            payload={"page": 1, "page_size": 10},
        ),
    )
    degraded_payload = cast(dict[str, object], runtime.events[-1]["payload"])
    degraded_items = cast(list[dict[str, object]], degraded_payload["items"])
    assert degraded_items[-1]["content"] == "旧媒体已失效"
    assert degraded_items[-1]["attachments"] == []
    attachment_error = cast(dict[str, object], degraded_items[-1]["attachment_error"])
    assert attachment_error["code"] == "media_unavailable"
    assert str(media_path) not in json.dumps(attachment_error, ensure_ascii=False)
    manager.close()
    storage.close()


def test_session_list_timestamp_boundary_requires_timezone() -> None:
    assert (
        channel_module._format_server_timestamp(
            "2026-07-14T01:37:51.488915+08:00",
            field="sessions.updated_at:test",
        )
        == "2026-07-13T17:37:51.488915Z"
    )
    with pytest.raises(RuntimeError, match="缺少时区"):
        channel_module._format_server_timestamp(
            "2026-07-14T01:37:51.488915",
            field="sessions.updated_at:test",
        )


@pytest.mark.asyncio
async def test_remote_outbound_media_keeps_response_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    store = AttachmentStore(tmp_path / "uploads")
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=store,
            ),
        )
    )

    async def snapshot(*args: object, **kwargs: object) -> RemoteMediaSnapshot:
        path = store.create_persistent_path("remote_", ".gif")
        content = b"GIF89a"
        path.write_bytes(content)
        return RemoteMediaSnapshot(
            path=path,
            filename="reaction.gif",
            content_type="image/gif",
            size_bytes=len(content),
            sha256="f" * 64,
        )

    monkeypatch.setattr("infra.mobile_realtime.channel.snapshot_remote_media", snapshot)
    descriptors = await channel._outbound_descriptors(
        f"akashic:{uuid4()}",
        ["https://media.example/reaction"],
    )

    assert descriptors[0]["filename"] == "reaction.gif"
    assert descriptors[0]["content_type"] == "image/gif"
    assert not list(store.root.glob("remote_*.gif"))
    await channel.stop()
    storage.close()


@pytest.mark.asyncio
async def test_remote_media_failure_keeps_final_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    manager = SessionManager(tmp_path / "workspace")
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    media = ["https://expired.example/reaction.gif"]
    session = manager.get_or_create(session_id)
    session.add_message("assistant", "文字仍应送达", media=media)
    manager.save(session)

    async def fail(*args: object, **kwargs: object) -> RemoteMediaSnapshot:
        raise RemoteMediaError("签名链接已失效")

    monkeypatch.setattr("infra.mobile_realtime.channel.snapshot_remote_media", fail)
    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="文字仍应送达",
            media=media,
            control_turn_id=uuid4().hex,
            session_message_id=str(session.messages[-1]["id"]),
        )
    )

    assert runtime.events[-1]["event_type"] == "message.final"
    payload = cast(dict[str, object], runtime.events[-1]["payload"])
    assert payload["content"] == "文字仍应送达"
    assert payload["attachments"] == []
    metadata = cast(dict[str, object], payload["metadata"])
    assert cast(dict[str, object], metadata["media_delivery"])["status"] == "failed"
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_final_event_maps_optimistic_user_to_persisted_identity(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    session_id = f"akashic:{uuid4()}"

    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="完成",
            metadata={
                "client_message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "persisted_user_message_id": f"{session_id}:0",
            },
            control_turn_id=uuid4().hex,
            session_message_id=f"{session_id}:1",
        )
    )

    payload = cast(dict[str, object], runtime.events[-1]["payload"])
    assert payload["client_message_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert payload["user_message_id"] == f"{session_id}:0"
    storage.close()


@pytest.mark.asyncio
async def test_final_payload_accepts_client_message_id_without_user_message_id(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """client_message_id 可单独存在（failed 终态）；state 缺失时 tl:final.published
    用已验证 outbound client_message_id 贯通，不用 current turn 猜。"""

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    with caplog.at_level(logging.INFO, logger="infra.mobile_realtime.channel"):
        await _provider_delivery(channel)(
            OutboundMessage(
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                content="处理消息时出错，请稍后再试。",
                metadata={"client_message_id": "cmid-fail"},
                control_turn_id=turn_id,
            )
        )

    final = runtime.events[-1]
    assert final["event_type"] == "message.final"
    assert final["turn_id"] == turn_id
    payload = cast(dict[str, object], final["payload"])
    assert payload["client_message_id"] == "cmid-fail"
    assert "user_message_id" not in payload
    final_records = [
        record
        for record in caplog.records
        if record.akashic_fields.get("event") == "tl:final.published"
    ]
    assert len(final_records) == 1
    assert final_records[0].akashic_fields["turn_id"] == turn_id
    assert final_records[0].akashic_fields["client_message_id"] == "cmid-fail"
    storage.close()


@pytest.mark.asyncio
async def test_typed_interrupted_outbound_publishes_one_durable_terminal(
    tmp_path: Path,
) -> None:
    """Worker interrupted projection 与既有终态墓碑幂等，共用权威 identity。"""

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    outbound = OutboundMessage(
        channel="akashic",
        chat_id=session_id.removeprefix("akashic:"),
        content="本轮已中断。",
        metadata={"client_message_id": "cmid-interrupted"},
        control_turn_id=turn_id,
        terminal_status=TurnTerminalStatus.INTERRUPTED,
    )

    await _provider_delivery(channel)(outbound)
    await _provider_delivery(channel)(outbound)

    terminals = [
        event
        for event in runtime.events
        if event.get("event_type") == "turn.interrupted"
    ]
    assert len(terminals) == 1
    assert terminals[0]["session_id"] == session_id
    assert terminals[0]["turn_id"] == turn_id
    assert terminals[0]["payload"] == {
        "status": "interrupted",
        "message": "本轮已中断。",
        "control_turn_id": turn_id,
        "client_message_id": "cmid-interrupted",
    }
    storage.close()


@pytest.mark.asyncio
async def test_control_reply_never_reuses_previous_message_id(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    session = manager.get_or_create(session_id)
    session.add_message("assistant", "相同的固定回复")
    manager.save(session)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )

    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="相同的固定回复",
        )
    )

    payload = cast(dict[str, object], runtime.events[-1]["payload"])
    assert "message_id" not in payload
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_turn_stop_accepts_a_shared_mobile_session(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    owner_device = uuid4().hex
    current_device = uuid4().hex
    _register_device(storage, owner_device)
    _register_device(storage, current_device)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    storage.claim_session(
        device_id=owner_device,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    manager.save(manager.get_or_create(session_id))
    interrupt = SimpleNamespace(
        request_interrupt=lambda **_: SimpleNamespace(
            status="interrupted", message="已停止"
        ),
    )
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=interrupt,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    turn_id = "01ARZ3NDEKTSV4RRFFQ69G5FAY"
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="正在生成",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
    )

    reply = await channel.handle_command(
        device_id=current_device,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAZ",
            command_type="turn.stop",
            session_id=session_id,
            turn_id=turn_id,
        ),
    )

    assert reply.type == "turn.stop.ok"
    assert runtime.events[-1]["event_type"] == "turn.interrupted"
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_turn_stop_idle_result_still_closes_stale_mobile_turn(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    manager.save(manager.get_or_create(session_id))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=SimpleNamespace(
                    request_interrupt=lambda **_: SimpleNamespace(
                        status="idle",
                        message="当前没有正在执行的任务。",
                    ),
                ),
                attachment_store=AttachmentStore(tmp_path / "uploads"),
                command_catalog_provider=None,
            ),
        )
    )
    turn_id = "01ARZ3NDEKTSV4RRFFQ69G5FAY"
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="正在生成",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
    )

    reply = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAZ",
            command_type="turn.stop",
            session_id=session_id,
            turn_id=turn_id,
        ),
    )

    assert reply.type == "turn.stop.ok"
    assert reply.payload["status"] == "idle"
    assert runtime.events[-1]["event_type"] == "turn.interrupted"
    assert session_id not in channel._active_turn_ids
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_resume_reconciles_recovered_terminal_turn_for_mobile_device(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = "01ARZ3NDEKTSV4RRFFQ69G5FAY"
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    manager.save(manager.get_or_create(session_id))
    manager.control_store.create_turn(
        TurnRecord(
            id=turn_id,
            thread_id=session_id,
            status=TurnStatus.QUEUED,
            input="维护前的提问",
            created_at=datetime.now(timezone.utc),
        )
    )
    manager.control_store.transition_turn(
        turn_id,
        expected_status=TurnStatus.QUEUED,
        status=TurnStatus.CANCELLED,
    )
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )

    await channel.reconcile_active_turns(
        device_id=device_id,
        active_turns=(turn_id,),
    )

    assert runtime.events == [
        {
            "event_type": "turn.interrupted",
            "session_id": session_id,
            "turn_id": turn_id,
            "payload": {
                "status": "cancelled",
                "message": "服务端已确认本轮生成结束",
                "control_turn_id": turn_id,
                "reason": "resume_reconciliation",
            },
            "device_id": device_id,
        }
    ]
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_stop_is_idempotent_after_authoritative_terminal_turn(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = "01ARZ3NDEKTSV4RRFFQ69G5FAY"
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    manager.save(manager.get_or_create(session_id))
    manager.control_store.create_turn(
        TurnRecord(
            id=turn_id,
            thread_id=session_id,
            status=TurnStatus.QUEUED,
            input="维护前的提问",
            created_at=datetime.now(timezone.utc),
        )
    )
    manager.control_store.transition_turn(
        turn_id,
        expected_status=TurnStatus.QUEUED,
        status=TurnStatus.IN_PROGRESS,
    )
    manager.control_store.transition_turn(
        turn_id,
        expected_status=TurnStatus.IN_PROGRESS,
        status=TurnStatus.INTERRUPTED,
    )
    channel._active_turn_ids[session_id] = turn_id
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )

    reply = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAZ",
            command_type="turn.stop",
            session_id=session_id,
            turn_id=turn_id,
        ),
    )

    assert reply.type == "turn.stop.ok"
    assert reply.payload == {
        "status": "already_terminal",
        "terminal_status": "interrupted",
        "message": "目标 turn 已经结束",
    }
    assert session_id not in channel._active_turn_ids
    assert runtime.events[-1]["event_type"] == "turn.interrupted"

    next_turn_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    channel._active_turn_ids[session_id] = next_turn_id
    replay = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FB0",
            command_type="turn.stop",
            session_id=session_id,
            turn_id=turn_id,
        ),
    )
    assert replay.type == "turn.stop.ok"
    assert channel._active_turn_ids[session_id] == next_turn_id

    channel._active_turn_ids.clear()
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_channel_stop_publishes_terminal_before_clearing_active_turn(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = "01ARZ3NDEKTSV4RRFFQ69G5FAY"
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    channel._active_turn_ids[session_id] = turn_id

    await channel.stop()

    assert runtime.events[-1] == {
        "event_type": "turn.interrupted",
        "session_id": session_id,
        "turn_id": turn_id,
        "payload": {
            "status": "interrupted",
            "message": "服务端正在维护，本轮生成已中断",
            "control_turn_id": turn_id,
            "reason": "runtime_shutdown",
        },
    }
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_command_list_uses_active_channel_catalog_without_stop(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    active_catalog = [
        ("undo", "撤销上一轮对话"),
        ("/memorystatus", "查看记忆整理状态"),
        ("emoji", "😀" * 129),
        ("stop", "中断当前回复"),
    ]
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
                command_catalog_provider=lambda: tuple(active_catalog),
            ),
        )
    )

    reply = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAZ",
            command_type="command.list",
        ),
    )

    assert reply.type == "command.list.ok"
    assert reply.payload == {
        "items": [
            {"command": "undo", "description": "撤销上一轮对话"},
            {"command": "memorystatus", "description": "查看记忆整理状态"},
            {"command": "emoji", "description": "😀" * 129},
        ]
    }
    active_catalog[:] = [("status", "查看状态")]
    refreshed = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FB0",
            command_type="command.list",
        ),
    )
    assert refreshed.payload == {
        "items": [{"command": "status", "description": "查看状态"}]
    }
    await channel.stop()
    storage.close()
    storage.close()


@pytest.mark.asyncio
async def test_message_send_preserves_mobile_slash_command_for_bus(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    bus = _Bus()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
                command_catalog_provider=lambda: (("undo", "撤销上一轮对话"),),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"

    reply = await channel.handle_command(
        device_id=device_id,
        frame=_message_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAY",
            session_id=session_id,
            text="/undo",
        ),
    )

    assert reply.type == "message.send.ok"
    assert len(bus.inbound) == 1
    assert cast(RawInbound, bus.inbound[0]).message.content == "/undo"
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_plugin_ui_catalog_is_empty_without_plugin_manager(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))

    reply = await channel.handle_plugin_ui_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FB0",
            command_type="plugin.ui.catalog",
        ),
    )

    assert reply.type == "plugin.ui.catalog.ok"
    assert reply.payload == {
        "catalog_revision": channel_module.hashlib.sha256(b"[]").hexdigest(),
        "items": [],
    }
    storage.close()


@pytest.mark.asyncio
async def test_plugin_ui_catalog_returns_not_modified_for_matching_revision(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))
    revision = channel_module.hashlib.sha256(b"[]").hexdigest()

    reply = await channel.handle_plugin_ui_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FB9",
            command_type="plugin.ui.catalog",
            payload={"subscribe": True, "if_revision": revision},
        ),
    )

    assert reply.type == "plugin.ui.catalog.not_modified"
    assert reply.payload == {"catalog_revision": revision}
    storage.close()


@pytest.mark.asyncio
async def test_plugin_ui_hot_update_only_targets_subscribed_connection(
    tmp_path: Path,
) -> None:
    class _MutableProvider:
        def __init__(self) -> None:
            self.revision = "a" * 64

        def catalog(self) -> dict[str, object]:
            return {"catalog_revision": self.revision, "items": []}

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    provider = _MutableProvider()
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    channel.bind_mobile_ui_provider(cast(Any, provider))

    await channel.handle_plugin_ui_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FA0",
            command_type="plugin.ui.catalog",
            payload={"subscribe": True},
        ),
    )
    provider.revision = "b" * 64
    await channel.refresh_mobile_ui_catalog()

    assert runtime.events == [
        {
            "control_type": "plugin.ui.changed",
            "payload": {"catalog_revision": "b" * 64},
            "device_id": device_id,
            "connection_epoch": 1,
        }
    ]

    await channel.handle_plugin_ui_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FA1",
            command_type="plugin.ui.catalog",
            payload={"subscribe": False},
        ),
    )
    provider.revision = "c" * 64
    await channel.refresh_mobile_ui_catalog()

    assert len(runtime.events) == 1
    storage.close()


@pytest.mark.asyncio
async def test_plugin_ui_catalog_does_not_create_durable_receipt(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))
    frame = _generic_frame(
        frame_id="01ARZ3NDEKTSV4RRFFQ69G5FB1",
        command_type="plugin.ui.catalog",
    )

    reply = await channel.handle_plugin_ui_command(device_id=device_id, frame=frame)

    assert reply.type == "plugin.ui.catalog.ok"
    count = storage._db.execute(
        "SELECT COUNT(*) FROM mobile_command_receipts WHERE command_id = ?",
        (frame.id,),
    ).fetchone()[0]
    assert count == 0
    storage.close()


@pytest.mark.asyncio
async def test_bootstrap_snapshot_queries_do_not_create_durable_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))

    async def execute_snapshot(**kwargs: object) -> channel_module.CommandReply:
        frame = cast(GenericCommand, kwargs["frame"])
        return channel_module.CommandReply(type=f"{frame.type}.ok", payload={})

    monkeypatch.setattr(channel, "_execute_command", execute_snapshot)
    command_types = (
        "command.list",
        "runtime.document.list",
        "runtime.capability.list",
        "scheduler.job.list",
        "model.catalog.get",
    )
    for index, command_type in enumerate(command_types):
        reply = await channel.handle_command(
            device_id=device_id,
            frame=_generic_frame(
                frame_id=f"01ARZ3NDEKTSV4RRFFQ69G5FA{index}",
                command_type=command_type,
            ),
        )
        assert reply.type == f"{command_type}.ok"

    durable = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FA5",
            command_type="runtime.document.get",
        ),
    )
    assert durable.type == "runtime.document.get.ok"
    receipts = storage._db.execute(
        "SELECT command_type FROM mobile_command_receipts ORDER BY command_type"
    ).fetchall()
    assert [row[0] for row in receipts] == ["runtime.document.get"]
    storage.close()


def _plugin_ui_query_frame(frame_id: str) -> GenericCommand:
    return _generic_frame(
        frame_id=frame_id,
        command_type="plugin.ui.query",
        payload={
            "owner_id": "dashboard:sample",
            "plugin_id": "sample@github",
            "plugin_revision": "revision-1",
            "method": "recall.current",
            "payload": {},
            "slot": "dashboard.main",
        },
    )


@pytest.mark.asyncio
async def test_plugin_ui_timeout_becomes_transient_command_error(
    tmp_path: Path,
) -> None:
    class _TimeoutProvider:
        def catalog(self) -> dict[str, object]:
            return {"catalog_revision": "a" * 64, "items": []}

        async def query(self, *args: object, **kwargs: object) -> dict[str, object]:
            raise channel_module.MobileUiQueryTimeout("插件 mobile UI query 超时")

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))
    channel.bind_mobile_ui_provider(cast(Any, _TimeoutProvider()))

    with pytest.raises(channel_module.MobileCommandError) as caught:
        await channel.handle_plugin_ui_command(
            device_id=device_id,
            frame=_plugin_ui_query_frame("01ARZ3NDEKTSV4RRFFQ69G5FB2"),
        )
    assert caught.value.code == "plugin_timeout"
    storage.close()


@pytest.mark.asyncio
async def test_plugin_ui_invalid_request_becomes_transient_command_error(
    tmp_path: Path,
) -> None:
    class _InvalidRequestProvider:
        def catalog(self) -> dict[str, object]:
            return {"catalog_revision": "a" * 64, "items": []}

        async def query(self, *args: object, **kwargs: object) -> dict[str, object]:
            raise channel_module.MobileUiRpcInvalidRequest("消息不属于请求会话")

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))
    channel.bind_mobile_ui_provider(cast(Any, _InvalidRequestProvider()))

    with pytest.raises(channel_module.MobileCommandError) as caught:
        await channel.handle_plugin_ui_command(
            device_id=device_id,
            frame=_plugin_ui_query_frame("01ARZ3NDEKTSV4RRFFQ69G5FB3"),
        )
    assert caught.value.code == "plugin_invalid_request"
    assert str(caught.value) == "消息不属于请求会话"
    storage.close()


@pytest.mark.asyncio
async def test_plugin_ui_execution_failure_becomes_transient_command_error(
    tmp_path: Path,
) -> None:
    class _FailedProvider:
        def __init__(self) -> None:
            self.calls = 0

        def catalog(self) -> dict[str, object]:
            return {"catalog_revision": "a" * 64, "items": []}

        async def query(self, *args: object, **kwargs: object) -> dict[str, object]:
            self.calls += 1
            raise channel_module.MobileUiRpcExecutionError(
                "插件 mobile UI RPC 执行失败: sample@github.recall.current"
            )

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, _Runtime(storage)))
    provider = _FailedProvider()
    channel.bind_mobile_ui_provider(cast(Any, provider))
    with pytest.raises(channel_module.MobileCommandError) as caught:
        await channel.handle_plugin_ui_command(
            device_id=device_id,
            frame=_plugin_ui_query_frame("01ARZ3NDEKTSV4RRFFQ69G5FB4"),
        )

    assert caught.value.code == "plugin_failed"
    assert (
        str(caught.value) == "插件 mobile UI RPC 执行失败: sample@github.recall.current"
    )
    assert provider.calls == 1
    storage.close()


@pytest.mark.asyncio
async def test_turn_stop_rejects_missing_or_stale_turn_identity(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    manager.save(manager.get_or_create(session_id))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=SimpleNamespace(
                    request_interrupt=lambda **_: SimpleNamespace(
                        status="interrupted",
                        message="已停止",
                    ),
                ),
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    active_turn = "01ARZ3NDEKTSV4RRFFQ69G5FAT"
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="正在生成",
            timestamp=datetime.now(timezone.utc),
            turn_id=active_turn,
        )
    )

    missing = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAS",
            command_type="turn.stop",
            session_id=session_id,
        ),
    )
    assert missing.type == "turn.stop.error"
    assert missing.payload["code"] == "turn_id_required"
    stale = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAR",
            command_type="turn.stop",
            session_id=session_id,
            turn_id="01ARZ3NDEKTSV4RRFFQ69G5FAQ",
        ),
    )
    assert stale.type == "turn.stop.error"
    assert stale.payload["code"] == "stale_turn"

    channel._active_turn_ids.clear()
    unknown = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAP",
            command_type="turn.stop",
            session_id=session_id,
            turn_id="01ARZ3NDEKTSV4RRFFQ69G5FAN",
        ),
    )
    assert unknown.type == "turn.stop.error"
    assert unknown.payload["code"] == "turn_not_active"

    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_delta_paths_reuse_existing_lock_without_allocating_lock() -> None:
    runtime = _Runtime(cast(MobileRealtimeStorage, object()))
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    key = ("akashic:test", "turn-1")
    existing_lock = asyncio.Lock()
    channel._delta_locks[key] = existing_lock
    channel._process_turns[key] = channel_module._ProcessTurnState(
        next_ordinal=0,
        thinking_block=None,
        tool_blocks={},
        answer_segments=[],
        control_turn_id="turn:logical-1",
    )

    real_lock = channel_module.asyncio.Lock
    allocations = 0

    def counting_lock() -> asyncio.Lock:
        nonlocal allocations
        allocations += 1
        return real_lock()

    channel._delta_locks.default_factory = counting_lock
    await channel._buffer_delta(
        session_id=key[0],
        turn_id=key[1],
        event_type="answer.delta",
        delta="x" * 4096,
        block_id=None,
        ordinal=None,
    )

    assert allocations == 0
    assert channel._delta_locks == {key: existing_lock}
    assert runtime.events == [
        {
            "event_type": "answer.delta",
            "session_id": key[0],
            "turn_id": key[1],
            "payload": {
                "delta": "x" * 4096,
                "control_turn_id": "turn:logical-1",
            },
        }
    ]


def test_stream_delta_transport_window_is_independent_from_display_refresh() -> None:
    assert channel_module._DELTA_TRANSPORT_COALESCE_SECONDS == pytest.approx(0.008)


@pytest.mark.asyncio
async def test_stream_deltas_batch_within_transport_window_and_flush_before_tool_and_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. 前五个 tick 供 turn 时间链里程碑消费（turn.started/首 thinking received
    #    + published/首 answer received + published），tool start/complete 必须保持
    #    100.0/105.125（tool 时长断言 5_125ms），final 里程碑消费最后一个 tick。
    ticks = iter((99.0, 99.1, 99.2, 99.3, 99.4, 100.0, 105.125, 106.0))
    monkeypatch.setattr(channel_module, "monotonic", lambda: next(ticks))
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    turn_id = uuid4().hex
    logical_turn_id = f"turn:{uuid4().hex}"
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="帮我检查",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            control_turn_id=logical_turn_id,
        )
    )

    for delta in ("思", "考", "中"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                thinking_delta=delta,
            )
        )
        await asyncio.sleep(0)
    # 首个 thinking delta 立即 flush；后续按短传输窗口合批，不绑定显示刷新率。
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "react.thinking.delta",
    ]
    await asyncio.sleep(channel_module._DELTA_TRANSPORT_COALESCE_SECONDS + 0.01)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "react.thinking.delta",
        "react.thinking.delta",
    ]
    first_thinking = cast(dict[str, object], runtime.events[1]["payload"])
    assert first_thinking["delta"] == "思"
    assert first_thinking["ordinal"] == 0
    assert cast(dict[str, object], runtime.events[2]["payload"])["delta"] == "考中"

    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            turn_id=turn_id,
            content_delta="A" * 4096,
        )
    )
    await channel._on_tool_call_started(
        ToolCallStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            iteration=1,
            call_id="call-1",
            tool_name="shell",
            arguments={
                "command": "pwd",
                "authorization": "Bearer private",
                "nested": {"client_secret": "private", "page": 1},
            },
            turn_id=turn_id,
        )
    )
    await channel._on_tool_call_completed(
        ToolCallCompleted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            iteration=1,
            call_id="call-1",
            tool_name="shell",
            arguments={"command": "pwd"},
            final_arguments={"command": "pwd -P"},
            status="success",
            result_preview="ok",
            turn_id=turn_id,
        )
    )
    started_payload = cast(dict[str, object], runtime.events[-2]["payload"])
    assert started_payload["arguments"] == {
        "command": "pwd",
        "authorization": "[已隐藏]",
        "nested": {"client_secret": "[已隐藏]", "page": 1},
    }
    completed_payload = cast(dict[str, object], runtime.events[-1]["payload"])
    assert completed_payload["arguments"] == {"command": "pwd -P"}
    assert completed_payload["duration_ms"] == 5_125
    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            turn_id=turn_id,
            thinking_delta="继续思考",
        )
    )
    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="完成",
            thinking="思考中",
            metadata={"mobile_attention": "confirmation"},
            control_turn_id=logical_turn_id,
            execution_attempt_id=turn_id,
        )
    )

    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "react.thinking.delta",
        "react.thinking.delta",
        "answer.delta",
        "react.tool.started",
        "react.tool.completed",
        "react.thinking.delta",
        "message.final",
    ]
    tool_started = cast(dict[str, object], runtime.events[4]["payload"])
    tool_completed = cast(dict[str, object], runtime.events[5]["payload"])
    second_thinking = cast(dict[str, object], runtime.events[6]["payload"])
    final_metadata = cast(dict[str, object], runtime.events[7]["payload"])["metadata"]
    assert all(
        cast(dict[str, object], event["payload"])["control_turn_id"]
        == logical_turn_id
        for event in runtime.events
    )
    assert tool_started["block_id"] == tool_completed["block_id"]
    assert tool_started["ordinal"] == tool_completed["ordinal"] == 1
    assert second_thinking["ordinal"] == 2
    assert second_thinking["block_id"] != first_thinking["block_id"]
    assert cast(dict[str, object], final_metadata)["mobile_attention"] == "confirmation"
    await channel.stop()
    storage.close()


@pytest.mark.asyncio
async def test_first_delta_orders_received_then_publish_then_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """首个 thinking delta 顺序：received 打点 → 真实 publish_event → published 打点。

    published 必须等到 runtime.publish_event 真实返回之后才记录（异常不伪装）。
    """
    # 1. turn.started 起点 10.0；received 11.0；publish 完成后 published 12.0。
    ticks = iter((10.0, 11.0, 12.0))
    monkeypatch.setattr(channel_module, "monotonic", lambda: next(ticks))
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _GatedPublishRuntime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    turn_id = "turn-1"
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="查一下",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-first",
        )
    )

    # 2. 发布被闸门挂起时：delta 已写入 runtime，received 已打点，published 未打。
    with caplog.at_level(logging.INFO, logger="infra.mobile_realtime.channel"):
        delta_task = asyncio.create_task(
            channel._on_stream_delta(
                StreamDeltaReady(
                    session_key=session_id,
                    channel="akashic",
                    chat_id=session_id.removeprefix("akashic:"),
                    turn_id=turn_id,
                    thinking_delta="思",
                )
            )
        )
        await asyncio.wait_for(runtime.delta_publish_started.wait(), timeout=5)
        assert [event["event_type"] for event in runtime.events] == [
            "turn.started",
            "react.thinking.delta",
        ]
        milestones = [
            record.akashic_fields
            for record in caplog.records
            if record.akashic_fields.get("event", "").startswith(
                "tl:delta.first_thinking"
            )
        ]
        assert [item["event"] for item in milestones] == [
            "tl:delta.first_thinking_received",
        ]
        assert milestones[0]["duration_ms"] == pytest.approx(1_000.0)
        assert milestones[0]["session_id"] == session_id
        assert milestones[0]["turn_id"] == turn_id
        assert milestones[0]["client_message_id"] == "cmid-first"

        # 3. 放行后 published 才打点，同一三元 identity、duration 从 turn.started。
        runtime.delta_publish_release.set()
        await asyncio.wait_for(delta_task, timeout=5)
        milestones = [
            record.akashic_fields
            for record in caplog.records
            if record.akashic_fields.get("event", "").startswith(
                "tl:delta.first_thinking"
            )
        ]
        assert [item["event"] for item in milestones] == [
            "tl:delta.first_thinking_received",
            "tl:delta.first_thinking_published",
        ]
        published = milestones[1]
        assert published["duration_ms"] == pytest.approx(2_000.0)
        assert published["session_id"] == session_id
        assert published["turn_id"] == turn_id
        assert published["client_message_id"] == "cmid-first"
    await channel.stop()
    storage.close()


@pytest.mark.asyncio
async def test_dual_field_delta_accepts_thinking_and_answer_without_short_circuit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """同一 StreamDeltaReady 同时携带 thinking_delta 与 content_delta 时，两个
    locked helper 必须各自无条件执行一次：thinking→answer 顺序不变，state 与
    wire 均不丢 answer，first published 里程碑各恰一次，终态无正文丢失。

    旧实现 or 短路：thinking 首段几乎必令 flush_now=True，answer helper 完全
    不执行，同一 chunk 的 content_delta 从 state、batch、wire、观测全部静默
    丢失。
    """
    # turn.started 起点 10.0；thinking received 11.0；answer received 11.5；
    # 发布后 thinking published 12.0、answer published 13.0；final 14.0。
    ticks = iter((10.0, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0, 16.0))
    monkeypatch.setattr(channel_module, "monotonic", lambda: next(ticks))
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = "turn-dual"
    manager.save(manager.get_or_create(session_id))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="双字段",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-dual",
        )
    )

    def _first_milestones() -> list[dict[str, object]]:
        return [
            record.akashic_fields
            for record in caplog.records
            if getattr(record, "akashic_fields", {})
            .get("event", "")
            .startswith("tl:delta.first_")
        ]

    # 1. 首个双字段事件：两个 helper 各执行一次，state 全量建立，首段即时
    #    flush，wire 严格 react.thinking.delta → answer.delta。
    with caplog.at_level(logging.INFO, logger="infra.mobile_realtime.channel"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                thinking_delta="思",
                content_delta="答",
            )
        )
        key = (session_id, turn_id)
        state = channel._process_turns[key]
        assert state.thinking_block is not None
        assert state.thinking_block[1] == 0
        assert state.next_ordinal == 1
        assert state.first_thinking_received
        assert state.answer_segments == ["答"]
        assert state.first_answer_received
        assert state.first_thinking_published
        assert state.first_answer_published
        assert [event["event_type"] for event in runtime.events] == [
            "turn.started",
            "react.thinking.delta",
            "answer.delta",
        ]
        thinking_payload = cast(dict[str, object], runtime.events[1]["payload"])
        answer_payload = cast(dict[str, object], runtime.events[2]["payload"])
        assert thinking_payload["delta"] == "思"
        assert thinking_payload["ordinal"] == 0
        assert answer_payload["delta"] == "答"
        # received 与 published 四个里程碑各恰一次、顺序 thinking → answer。
        assert [item["event"] for item in _first_milestones()] == [
            "tl:delta.first_thinking_received",
            "tl:delta.first_answer_received",
            "tl:delta.first_thinking_published",
            "tl:delta.first_answer_published",
        ]

        # 2. 第二个双字段事件：两个 helper 再次各执行一次；thinking 复用既有块
        #    不重复分配 ordinal，answer 追加不进批丢失。
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                thinking_delta="继",
                content_delta="答2",
            )
        )
        assert channel._process_turns[key].thinking_block == state.thinking_block
        assert state.next_ordinal == 1
        assert state.answer_segments == ["答", "答2"]
        assert [item["event"] for item in _first_milestones()] == [
            "tl:delta.first_thinking_received",
            "tl:delta.first_answer_received",
            "tl:delta.first_thinking_published",
            "tl:delta.first_answer_published",
        ]

        # 3. 终态：残留批 flush → terminal，正文严格 已接受 delta → message.final。
        await _provider_delivery(channel)(
            OutboundMessage(
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                content="答答2",
                thinking="思继",
                control_turn_id=turn_id,
            )
        )
        assert [event["event_type"] for event in runtime.events] == [
            "turn.started",
            "react.thinking.delta",
            "answer.delta",
            "react.thinking.delta",
            "answer.delta",
            "message.final",
        ]
        assert cast(dict[str, object], runtime.events[3]["payload"])["delta"] == "继"
        assert cast(dict[str, object], runtime.events[4]["payload"])["delta"] == "答2"
        assert channel._process_turns == {}
        assert channel._delta_batches == {}
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_terminal_and_reconcile_flush_pending_delta_before_terminal_event(
    tmp_path: Path,
) -> None:
    """终态前必须先 flush 已缓冲 delta；终态后批与定时器消失，无迟到发布。"""

    # 1. 起一轮 turn，首段立即发布，第二段留在批里等待定时器。
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = "01ARZ3NDEKTSV4RRFFQ69G5FAY"
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    manager.save(manager.get_or_create(session_id))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="A",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-A",
        )
    )
    for delta in ("一", "二"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                thinking_delta=delta,
            )
        )
        await asyncio.sleep(0)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "react.thinking.delta",
    ]
    assert (session_id, turn_id) in channel._delta_batches

    # 2. terminal：残留批必须先于 message.final 发布，随后批与定时器都被清理。
    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="终稿",
            control_turn_id=turn_id,
        )
    )
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "react.thinking.delta",
        "react.thinking.delta",
        "message.final",
    ]
    assert cast(dict[str, object], runtime.events[2]["payload"])["delta"] == "二"
    assert channel._delta_batches == {}
    assert channel._process_turns == {}
    await asyncio.sleep(channel_module._DELTA_TRANSPORT_COALESCE_SECONDS + 0.01)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "react.thinking.delta",
        "react.thinking.delta",
        "message.final",
    ]

    # 3. reconcile 终态同样先 flush 残留批再发布 turn.interrupted。
    second_turn = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    manager.control_store.create_turn(
        TurnRecord(
            id=second_turn,
            thread_id=session_id,
            status=TurnStatus.QUEUED,
            input="第二问",
            created_at=datetime.now(timezone.utc),
        )
    )
    manager.control_store.transition_turn(
        second_turn,
        expected_status=TurnStatus.QUEUED,
        status=TurnStatus.CANCELLED,
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="B",
            timestamp=datetime.now(timezone.utc),
            turn_id=second_turn,
            client_message_id="cmid-B",
        )
    )
    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            turn_id=second_turn,
            content_delta="残",
        )
    )
    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            turn_id=second_turn,
            content_delta="余",
        )
    )
    events_before = [event["event_type"] for event in runtime.events]
    await channel.reconcile_active_turns(
        device_id=device_id,
        active_turns=(second_turn,),
    )
    assert [event["event_type"] for event in runtime.events] == [
        *events_before,
        "answer.delta",
        "turn.interrupted",
    ]
    assert channel._delta_batches == {}
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_terminal_barrier_flushes_accepted_deltas_then_terminal_and_drops_late(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """终态 barrier 在锁内收口：已接受 delta（含 final suffix）先于 terminal，
    临界区内与收口后到达的迟到 delta 都被丢弃，不重建 batch/timer、无 failure。"""

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _FinalGatedRuntime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-A",
        )
    )
    # 首个 content delta 立即 flush，第二段留在批里等待定时器。
    for delta in ("你", "好"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta=delta,
            )
        )
        await asyncio.sleep(0)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
    ]

    # 1. final 路径：top flush 发布已缓冲 delta → suffix delta → barrier 内发布
    #    message.final；闸门卡住 terminal 发布，让 barrier 临界区真实持锁。
    with caplog.at_level(logging.INFO, logger="infra.mobile_realtime.channel"):
        final_task = asyncio.create_task(
            _provider_delivery(channel)(
                OutboundMessage(
                    channel="akashic",
                    chat_id=session_id.removeprefix("akashic:"),
                    content="你好世界🙂",
                    control_turn_id=turn_id,
                )
            )
        )
        await asyncio.wait_for(runtime.final_started.wait(), timeout=5)
        assert [event["event_type"] for event in runtime.events] == [
            "turn.started",
            "answer.delta",
            "answer.delta",
            "answer.delta",
            "message.final",
        ]
        # 2. publish 尚未成功返回：in-flight 终态不提交 closed 墓碑（失败可重试），
        #    barrier 仍由 per-turn 锁持有，process state 保留。
        assert (session_id, turn_id) not in channel._turn_terminals
        assert channel._delta_locks.get((session_id, turn_id)) is not None
        assert (session_id, turn_id) in channel._process_turns

        # 3. 临界区内 late delta 以任务排队等同一把锁，绝不新建 batch/timer。
        late_task = asyncio.create_task(
            channel._on_stream_delta(
                StreamDeltaReady(
                    session_key=session_id,
                    channel="akashic",
                    chat_id=session_id.removeprefix("akashic:"),
                    turn_id=turn_id,
                    content_delta="晚",
                )
            )
        )
        await asyncio.sleep(0.01)
        assert not late_task.done()
        assert [event["event_type"] for event in runtime.events] == [
            "turn.started",
            "answer.delta",
            "answer.delta",
            "answer.delta",
            "message.final",
        ]
        assert channel._delta_batches == {}
        assert channel._delta_failure is None

        # 4. 放行：final 成功关闭后，排队的 late delta 拿到锁见墓碑被丢弃。
        runtime.final_release.set()
        await asyncio.wait_for(final_task, timeout=5)
        await asyncio.wait_for(late_task, timeout=5)
        events = [event["event_type"] for event in runtime.events]
        assert events == [
            "turn.started",
            "answer.delta",
            "answer.delta",
            "answer.delta",
            "message.final",
        ]
        final_payload = cast(dict[str, object], runtime.events[-1]["payload"])
        assert final_payload["content"] == ""
        deltas = [
            cast(str, cast(dict[str, object], event["payload"])["delta"])
            for event in runtime.events
            if event["event_type"] == "answer.delta"
        ]
        assert "".join(deltas) == "你好世界🙂"
        assert channel._delta_batches == {}
        assert channel._process_turns == {}
        assert (session_id, turn_id) in channel._turn_terminals
        assert channel._delta_locks.get((session_id, turn_id)) is None
        assert channel._delta_failure is None

    # 5. 定时器窗口过后仍无迟到发布
    await asyncio.sleep(channel_module._DELTA_TRANSPORT_COALESCE_SECONDS + 0.01)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
        "answer.delta",
        "answer.delta",
        "message.final",
    ]

    # 6. 收口完成后（maps 已清理）的迟到 delta 仍被拒绝，且不能重建 lock/batch
    with caplog.at_level(logging.INFO, logger="infra.mobile_realtime.channel"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta="更晚",
            )
        )
    assert channel._delta_locks.get((session_id, turn_id)) is None
    assert channel._delta_batches == {}
    assert channel._delta_failure is None
    dropped = [
        record
        for record in caplog.records
        if record.akashic_fields.get("event") == "tl:turn.late.drop"
    ]
    assert len(dropped) == 2
    assert {item.akashic_fields["counts"] for item in dropped} == {
        "event_type=answer.delta",
    }
    assert all(
        item.akashic_fields["session_id"] == session_id
        and item.akashic_fields["turn_id"] == turn_id
        for item in dropped
    )
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_terminal_and_late_delta_queued_on_same_lock_release_terminal_then_drop(
    tmp_path: Path,
) -> None:
    """terminal 与 late delta 同时排队等待同一把 per-turn 锁：FIFO 释放后先
    terminal（锁内 flush 已接受 delta 再发布），后到的 delta 被拒绝且无重建。"""

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-A",
        )
    )
    for delta in ("一", "二"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta=delta,
            )
        )
        await asyncio.sleep(0)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
    ]

    # 1. 测试先占住 per-turn 锁；terminal 与 late delta 依次排队等待同一把锁。
    key = (session_id, turn_id)
    lock = channel._delta_locks[key]
    await lock.acquire()
    terminal_task = asyncio.create_task(
        channel._publish_terminal(
            session_id=session_id,
            turn_id=turn_id,
            event_type="message.final",
            payload={"content": "终"},
        )
    )
    await asyncio.sleep(0.01)
    late_task = asyncio.create_task(
        channel._buffer_bounded_delta(
            session_id=session_id,
            turn_id=turn_id,
            event_type="answer.delta",
            delta="晚",
            block_id=None,
            ordinal=None,
        )
    )
    await asyncio.sleep(0.01)
    lock.release()

    # 2. FIFO：terminal 先拿锁（flush 已接受 delta 再发布 terminal），
    #    late delta 拿锁后看到 closed 被拒绝，绝不在终态后发布。
    await asyncio.wait_for(terminal_task, timeout=5)
    assert await asyncio.wait_for(late_task, timeout=5) is False
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
        "answer.delta",
        "message.final",
    ]
    assert cast(dict[str, object], runtime.events[2]["payload"])["delta"] == "二"
    assert channel._delta_batches == {}
    assert channel._process_turns == {}
    assert key in channel._turn_terminals
    assert channel._delta_locks.get(key) is None
    assert channel._delta_failure is None

    # 3. 定时器窗口过后仍无迟到发布，也没有 timer 崩溃。
    await asyncio.sleep(channel_module._DELTA_TRANSPORT_COALESCE_SECONDS + 0.01)
    assert len(runtime.events) == 4
    await channel.stop()
    manager.close()
    storage.close()


async def _race_channel(
    tmp_path: Path,
) -> tuple[
    _Runtime, MobileRealtimeChannel, SessionManager, MobileRealtimeStorage, str, str
]:
    """起一轮带 client_message_id 的 turn，返回 (runtime, channel, manager, storage, session_id, turn_id)。"""

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-race",
        )
    )
    return runtime, channel, manager, storage, session_id, turn_id


@pytest.mark.asyncio
async def test_stream_delta_racing_terminal_commits_no_state_or_wire(
    tmp_path: Path,
) -> None:
    """delta 与 terminal 竞争（terminal 先收口）：检查+mutation 都在锁内，delta
    拿到锁后见 closed 整事件丢弃——state 的 thinking 块/正文/first 标志零变化，
    wire 严格 已接受 delta → terminal，terminal 后无 delta。"""

    runtime, channel, manager, storage, session_id, turn_id = await _race_channel(
        tmp_path
    )
    # 1. 首个 answer delta 立即发布；测试占住 per-turn 锁，让 terminal 与携带
    #    thinking+content 的 delta 依次排队，FIFO 保证 terminal 先收口。
    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            turn_id=turn_id,
            content_delta="一",
        )
    )
    await asyncio.sleep(0)
    key = (session_id, turn_id)
    state_ref = channel._process_turns[key]
    lock = channel._delta_locks[key]
    await lock.acquire()
    terminal_task = asyncio.create_task(
        channel._publish_terminal(
            session_id=session_id,
            turn_id=turn_id,
            event_type="message.final",
            payload={"content": "终"},
        )
    )
    await asyncio.sleep(0.01)
    delta_task = asyncio.create_task(
        channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                thinking_delta="思",
                content_delta="二",
            )
        )
    )
    await asyncio.sleep(0.01)
    lock.release()

    # 2. wire：已接受 delta → terminal，terminal 后零 delta。
    await asyncio.wait_for(terminal_task, timeout=5)
    await asyncio.wait_for(delta_task, timeout=5)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
        "message.final",
    ]

    # 3. state：被丢弃的事件未提交任何 mutation——无 thinking 块、无正文追加、
    #    无 thinking first 标志、无 ordinal 消耗；race 前的 "一" 状态原样保留。
    assert state_ref.answer_segments == ["一"]
    assert state_ref.thinking_block is None
    assert state_ref.next_ordinal == 0
    assert not state_ref.first_thinking_received
    assert not state_ref.first_thinking_published
    assert state_ref.first_answer_received
    assert state_ref.first_answer_published
    assert channel._process_turns == {}
    assert channel._delta_batches == {}
    assert channel._delta_locks.get(key) is None
    assert key in channel._turn_terminals
    assert channel._delta_failure is None
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_output_completed_never_follows_terminal(tmp_path: Path) -> None:
    """output.completed 与 /stop terminal 竞争：completion 的 durable publish
    与 terminal 在同一 per-turn 锁临界区，因此要么先于 terminal，要么在
    terminal 收口后被丢弃，绝不排在 terminal 之后。"""

    runtime, channel, manager, storage, session_id, turn_id = await _race_channel(
        tmp_path
    )
    key = (session_id, turn_id)
    lock = channel._delta_locks[key]
    await lock.acquire()

    output_task = asyncio.create_task(
        channel._on_output_completed(
            TurnOutputCompleted(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                client_message_id="cmid-race",
            )
        )
    )
    await asyncio.sleep(0.01)
    terminal_task = asyncio.create_task(
        channel._publish_terminal(
            session_id=session_id,
            turn_id=turn_id,
            event_type="turn.interrupted",
            payload={"status": "interrupted", "message": "已中断"},
        )
    )
    await asyncio.sleep(0.01)
    lock.release()

    await asyncio.wait_for(output_task, timeout=5)
    await asyncio.wait_for(terminal_task, timeout=5)

    event_types = [event["event_type"] for event in runtime.events]
    assert event_types[-1] == "turn.interrupted"
    if "turn.output.completed" in event_types:
        assert event_types.index("turn.output.completed") < event_types.index(
            "turn.interrupted"
        )
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_tool_events_racing_terminal_dropped_without_state_touch(
    tmp_path: Path,
) -> None:
    """tool.started/completed 与 terminal 竞争（terminal 先收口）：已接受 delta
    严格先于 terminal，terminal 后的 tool 事件被丢弃，process state 零触碰。"""

    runtime, channel, manager, storage, session_id, turn_id = await _race_channel(
        tmp_path
    )
    # 1. 首段立即发布，第二段留在批里等 terminal 代为 flush。
    for delta in ("一", "二"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta=delta,
            )
        )
        await asyncio.sleep(0)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
    ]
    key = (session_id, turn_id)
    state_ref = channel._process_turns[key]
    lock = channel._delta_locks[key]
    await lock.acquire()
    terminal_task = asyncio.create_task(
        channel._publish_terminal(
            session_id=session_id,
            turn_id=turn_id,
            event_type="message.final",
            payload={"content": "终"},
        )
    )
    await asyncio.sleep(0.01)
    tool_started_task = asyncio.create_task(
        channel._on_tool_call_started(
            ToolCallStarted(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                iteration=1,
                call_id="call-1",
                tool_name="shell",
                arguments={"command": "pwd"},
                turn_id=turn_id,
            )
        )
    )
    await asyncio.sleep(0.01)
    tool_completed_task = asyncio.create_task(
        channel._on_tool_call_completed(
            ToolCallCompleted(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                iteration=1,
                call_id="call-1",
                tool_name="shell",
                arguments={"command": "pwd"},
                final_arguments={"command": "pwd"},
                status="success",
                result_preview="ok",
                turn_id=turn_id,
            )
        )
    )
    await asyncio.sleep(0.01)
    lock.release()

    # 2. terminal 先收口并代为 flush 残留批；tool 事件全部丢弃，wire 零 react.*。
    await asyncio.wait_for(terminal_task, timeout=5)
    await asyncio.wait_for(tool_started_task, timeout=5)
    await asyncio.wait_for(tool_completed_task, timeout=5)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
        "answer.delta",
        "message.final",
    ]
    assert cast(dict[str, object], runtime.events[2]["payload"])["delta"] == "二"

    # 3. state：tool 未触碰 thinking 块/tool_blocks/ordinal，无残留、无异常。
    assert state_ref.thinking_block is None
    assert state_ref.tool_blocks == {}
    assert state_ref.next_ordinal == 0
    assert state_ref.answer_segments == ["一", "二"]
    assert channel._process_turns == {}
    assert channel._delta_batches == {}
    assert channel._delta_locks.get(key) is None
    assert channel._delta_failure is None
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_post_terminal_flush_duplicate_and_late_events_never_rebuild(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """terminal 完成后的 flush/重复 terminal/迟到 delta 与 tool 一律拒绝，且
    _delta_locks/_delta_batches 不被 defaultdict 重建。"""

    runtime, channel, manager, storage, session_id, turn_id = await _race_channel(
        tmp_path
    )
    key = (session_id, turn_id)
    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="完成",
            control_turn_id=turn_id,
        )
    )
    assert key in channel._turn_terminals
    assert channel._delta_locks.get(key) is None
    assert channel._delta_batches == {}
    assert channel._process_turns == {}

    # 1. cleanup 后的 flush 与重复 terminal：返回 False，不重建锁。
    assert await channel._flush_deltas(session_id, turn_id) is False
    assert (
        await channel._publish_terminal(
            session_id=session_id,
            turn_id=turn_id,
            event_type="message.final",
            payload={"content": "重复"},
        )
        is False
    )
    # 2. 迟到 delta / tool.started / tool.completed：全部丢弃，无 state 异常。
    with caplog.at_level(logging.INFO, logger="infra.mobile_realtime.channel"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta="晚",
            )
        )
        await channel._on_tool_call_started(
            ToolCallStarted(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                iteration=1,
                call_id="call-1",
                tool_name="shell",
                arguments={"command": "pwd"},
                turn_id=turn_id,
            )
        )
        await channel._on_tool_call_completed(
            ToolCallCompleted(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                iteration=1,
                call_id="call-1",
                tool_name="shell",
                arguments={"command": "pwd"},
                final_arguments={"command": "pwd"},
                status="success",
                result_preview="ok",
                turn_id=turn_id,
            )
        )

    # 3. 墓碑仍在、锁/批/state 零重建，wire 只有 turn.started + message.final。
    assert dict(channel._delta_locks) == {}
    assert channel._delta_batches == {}
    assert channel._process_turns == {}
    assert key in channel._turn_terminals
    assert channel._delta_failure is None
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "message.final",
    ]
    dropped = [
        record
        for record in caplog.records
        if record.akashic_fields.get("event") == "tl:turn.late.drop"
    ]
    assert {item.akashic_fields["counts"] for item in dropped} == {
        "event_type=answer.delta",
        "event_type=react.tool.started",
        "event_type=react.tool.completed",
    }
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_racing_delta_dropped_so_final_suffix_covers_full_body(
    tmp_path: Path,
) -> None:
    """竞争失败的 delta 不得污染 answer_segments：正文完整由 terminal 的
    suffix 路径补齐，wire 上 delta 拼接 == durable final 正文。"""

    runtime, channel, manager, storage, session_id, turn_id = await _race_channel(
        tmp_path
    )
    # 1. 首段立即发布，第二段留在批里；final owner 与第三个 delta 竞争。
    for delta in ("你", "好"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta=delta,
            )
        )
        await asyncio.sleep(0)
    key = (session_id, turn_id)
    state_ref = channel._process_turns[key]
    lock = channel._delta_locks[key]
    await lock.acquire()
    final_task = asyncio.create_task(
        _provider_delivery(channel)(
            OutboundMessage(
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                content="你好中",
                control_turn_id=turn_id,
            )
        )
    )
    await asyncio.sleep(0.01)
    delta_task = asyncio.create_task(
        channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta="中",
            )
        )
    )
    await asyncio.sleep(0.01)
    lock.release()

    # 2. final 先收口：flush 残留批 → suffix 补齐正文 → terminal；竞争的 delta
    #    被原子丢弃，answer_segments 不被污染。
    await asyncio.wait_for(final_task, timeout=5)
    await asyncio.wait_for(delta_task, timeout=5)
    assert state_ref.answer_segments == ["你", "好"]
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
        "answer.delta",
        "answer.delta",
        "message.final",
    ]
    deltas = "".join(
        cast(str, cast(dict[str, object], event["payload"])["delta"])
        for event in runtime.events
        if event["event_type"] == "answer.delta"
    )
    assert deltas == "你好中"
    final_payload = cast(dict[str, object], runtime.events[-1]["payload"])
    assert final_payload["content"] == ""
    assert channel._delta_batches == {}
    assert channel._process_turns == {}
    assert channel._delta_locks.get(key) is None
    assert channel._delta_failure is None
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_late_a_final_keeps_b_active_and_identity(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A/B overlap：迟到的 A final 归属 A（不 fallback 归 B）、不清 B 的 active/
    process/send maps；终态 identity 贯通 A 的 client_message_id。"""

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_a = "turn-A"
    turn_b = "turn-B"
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="A",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_a,
            client_message_id="cmid-A",
        )
    )
    for delta in ("A1", "A2"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_a,
                content_delta=delta,
            )
        )
        await asyncio.sleep(0)
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="B",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_b,
            client_message_id="cmid-B",
        )
    )
    channel._send_received_at[(session_id, "cmid-A")] = 10.0
    channel._send_received_at[(session_id, "cmid-B")] = 20.0

    # 1. 迟到的 A final 通过 execution attempt 归属 A；逻辑 Turn 独立投影。
    logical_turn_a = "turn:logical-A"
    with caplog.at_level(logging.INFO, logger="infra.mobile_realtime.channel"):
        await _provider_delivery(channel)(
            OutboundMessage(
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                content="A1A2终",
                control_turn_id=logical_turn_a,
                execution_attempt_id=turn_a,
                metadata={
                    "client_message_id": "cmid-A",
                    "persisted_user_message_id": "uid-A",
                },
            )
        )
    final = runtime.events[-1]
    assert final["event_type"] == "message.final"
    assert final["turn_id"] == turn_a
    final_payload = cast(dict[str, object], final["payload"])
    assert final_payload["control_turn_id"] == logical_turn_a
    assert final_payload["client_message_id"] == "cmid-A"
    assert final_payload["user_message_id"] == "uid-A"
    final_records = [
        record
        for record in caplog.records
        if record.akashic_fields.get("event") == "tl:final.published"
    ]
    assert len(final_records) == 1
    assert final_records[0].akashic_fields["turn_id"] == turn_a
    assert final_records[0].akashic_fields["client_message_id"] == "cmid-A"

    # 2. A cleanup 只清 A：B 的 active/process/turn/send maps 全部保留。
    assert channel._active_turn_ids == {session_id: turn_b}
    assert set(channel._process_turns) == {(session_id, turn_b)}
    assert channel._process_turns[(session_id, turn_b)].client_message_id == "cmid-B"
    assert channel._turn_started_at == {
        (session_id, turn_b): channel._turn_started_at[(session_id, turn_b)]
    }
    assert channel._send_received_at == {(session_id, "cmid-B"): 20.0}
    assert channel._delta_batches == {}
    assert (session_id, turn_a) in channel._turn_terminals
    assert (session_id, turn_b) not in channel._turn_terminals
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_interrupt_publish_paths_carry_known_client_message_id(
    tmp_path: Path,
) -> None:
    """active stop 与 shutdown 的 turn.interrupted 发布都贯通进程内已知
    client_message_id；恢复态未知时允许缺失（由既有精确 payload 测试覆盖）。"""

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    manager.save(manager.get_or_create(session_id))
    interrupt = SimpleNamespace(
        request_interrupt=lambda **_: SimpleNamespace(
            status="interrupted", message="已停止"
        ),
    )
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=interrupt,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    active_turn = "turn-stop-active"
    manager.control_store.create_turn(
        TurnRecord(
            id=active_turn,
            thread_id=session_id,
            status=TurnStatus.QUEUED,
            input="A",
            created_at=datetime.now(timezone.utc),
            metadata={"interactionId": "turn-logical-stop"},
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="A",
            timestamp=datetime.now(timezone.utc),
            turn_id=active_turn,
            client_message_id="cmid-stop",
        )
    )

    # 1. active stop：payload 保持 status/message，并贯通已知 client_message_id
    reply = await channel.handle_command(
        device_id=device_id,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAZ",
            command_type="turn.stop",
            session_id=session_id,
            turn_id=active_turn,
        ),
    )
    assert reply.type == "turn.stop.ok"
    interrupted = [
        event for event in runtime.events if event["event_type"] == "turn.interrupted"
    ]
    assert interrupted[-1]["payload"] == {
        "status": "interrupted",
        "message": "已停止",
        "control_turn_id": "turn-logical-stop",
        "client_message_id": "cmid-stop",
    }
    assert session_id not in channel._active_turn_ids

    # 2. shutdown：另一活动 turn 的 interrupted 同样贯通已知 client_message_id
    shutdown_turn = "turn-shutdown"
    manager.control_store.create_turn(
        TurnRecord(
            id=shutdown_turn,
            thread_id=session_id,
            status=TurnStatus.QUEUED,
            input="B",
            created_at=datetime.now(timezone.utc),
            metadata={"interactionId": "turn-logical-shutdown"},
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="B",
            timestamp=datetime.now(timezone.utc),
            turn_id=shutdown_turn,
            client_message_id="cmid-shutdown",
        )
    )
    await channel.stop()
    interrupted = [
        event for event in runtime.events if event["event_type"] == "turn.interrupted"
    ]
    assert interrupted[-1]["payload"] == {
        "status": "interrupted",
        "message": "服务端正在维护，本轮生成已中断",
        "reason": "runtime_shutdown",
        "control_turn_id": "turn-logical-shutdown",
        "client_message_id": "cmid-shutdown",
    }
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_final_projects_only_explicit_mobile_metadata(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    session_id = f"akashic:{uuid4()}"

    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="完成",
            metadata={
                "mobile_attention": "confirmation",
                "tool_chain": [{"result": "x" * 350_000}],
                "tools_used": ["shell"],
            },
            control_turn_id=uuid4().hex,
        )
    )

    final = runtime.events[-1]
    payload = cast(dict[str, object], final["payload"])
    assert payload["content"] == "完成"
    assert payload["metadata"] == {"mobile_attention": "confirmation"}
    assert (
        len(json.dumps(final, ensure_ascii=False).encode("utf-8"))
        < MAX_JSON_FRAME_BYTES
    )
    storage.close()


@pytest.mark.asyncio
async def test_nonstreamed_large_unicode_answer_uses_bounded_deltas(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    session_id = f"akashic:{uuid4()}"
    content = ("长回复🙂\n" * 150_000)[:1_000_000]

    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content=content,
            control_turn_id=uuid4().hex,
        )
    )

    deltas = [
        cast(dict[str, object], event["payload"])["delta"]
        for event in runtime.events
        if event["event_type"] == "answer.delta"
    ]
    final = runtime.events[-1]
    final_payload = cast(dict[str, object], final["payload"])
    assert "".join(cast(list[str], deltas)) == content
    assert final_payload["content"] == ""
    assert all(
        len(json.dumps(event, ensure_ascii=False).encode("utf-8"))
        < MAX_JSON_FRAME_BYTES
        for event in runtime.events
    )
    storage.close()


@pytest.mark.asyncio
async def test_final_emits_only_missing_streamed_answer_suffix(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
    )
    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            turn_id=turn_id,
            content_delta="你好",
        )
    )

    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="你好世界🙂",
            control_turn_id=turn_id,
        )
    )

    deltas = [
        cast(str, cast(dict[str, object], event["payload"])["delta"])
        for event in runtime.events
        if event["event_type"] == "answer.delta"
    ]
    final_payload = cast(dict[str, object], runtime.events[-1]["payload"])
    assert "".join(deltas) == "你好世界🙂"
    assert final_payload["content"] == ""
    storage.close()


@pytest.mark.asyncio
async def test_divergent_stream_keeps_final_correction_inline(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
    )
    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            turn_id=turn_id,
            content_delta="草稿",
        )
    )

    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="定稿",
            control_turn_id=turn_id,
        )
    )

    final_payload = cast(dict[str, object], runtime.events[-1]["payload"])
    assert final_payload["content"] == "定稿"
    storage.close()


@pytest.mark.asyncio
async def test_proactive_sender_uses_mobile_event_path(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    push = _PushTool()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=push,
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    chat_id = str(uuid4())
    storage.claim_session(
        device_id=device_id,
        session_id=f"akashic:{chat_id}",
        created_at=datetime.now(timezone.utc),
    )
    deliver = cast(Any, _provider_delivery(channel))

    receipt = await deliver(
        ChannelMessage(
            channel="akashic",
            chat_id=chat_id,
            content="该休息一下了",
        )
    )

    assert receipt.status is DeliveryStatus.SUCCESS
    assert len(runtime.events) == 1
    proactive = cast(dict[str, Any], runtime.events[0])
    assert proactive["event_type"] == "message.proactive"
    assert proactive["session_id"] == f"akashic:{chat_id}"
    payload = cast(dict[str, Any], proactive["payload"])
    assert payload["content"] == "该休息一下了"
    assert payload["attachments"] == []
    assert payload["metadata"] == {"source": "message_push"}
    assert payload["control_turn_id"].startswith("turn:")
    await channel.stop()
    storage.close()


@pytest.mark.asyncio
async def test_proactive_metadata_sender_forwards_delivery_id(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    _register_device(storage, uuid4().hex)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    push = _PushTool()
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=push,
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    chat_id = str(uuid4())
    deliver = cast(Any, _provider_delivery(channel))

    receipt = await deliver(
        ChannelMessage(
            channel="akashic",
            chat_id=chat_id,
            content="该休息一下了",
            metadata={"delivery_id": "delivery-1"},
        )
    )

    assert receipt.status is DeliveryStatus.SUCCESS
    payload = cast(dict[str, Any], runtime.events[-1]["payload"])
    assert payload["control_turn_id"].startswith("turn:")
    assert {key: value for key, value in payload.items() if key != "control_turn_id"} == {
        "content": "该休息一下了",
        "attachments": [],
        "metadata": {"source": "message_push"},
        "delivery_id": "delivery-1",
    }
    await channel.stop()
    storage.close()


@pytest.mark.asyncio
async def test_proactive_attachment_commits_one_replayable_logical_message(
    tmp_path: Path,
) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_ids = (uuid4().hex, uuid4().hex)
    for device_id in device_ids:
        _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    push = _PushTool()
    upload_store = AttachmentStore(tmp_path / "uploads")
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=SessionManager(tmp_path / "workspace"),
                event_bus=_EventBus(),
                push_tool=push,
                interrupt_controller=None,
                attachment_store=upload_store,
            ),
        )
    )
    source = tmp_path / "photo.png"
    source.write_bytes(b"png-payload")
    chat_id = str(uuid4())
    deliver = cast(Any, _provider_delivery(channel))

    receipt = await deliver(
        ChannelMessage(
            channel="akashic",
            chat_id=chat_id,
            content="看图",
            attachments=(ChannelAttachment(AttachmentKind.IMAGE, str(source)),),
            metadata={"delivery_id": "delivery-with-image"},
        )
    )

    assert receipt.status is DeliveryStatus.SUCCESS
    assert len(receipt.canonical_media) == 1
    assert receipt.canonical_media[0] != str(source)
    assert Path(receipt.canonical_media[0]).read_bytes() == b"png-payload"
    envelopes = []
    for device_id in device_ids:
        replay = storage.read_durable_events(
            device_id,
            after_event_seq=0,
            limit=10,
        )
        assert len(replay) == 1
        envelopes.append(json.loads(replay[0].envelope_json))
    assert envelopes[0] == envelopes[1]
    assert envelopes[0]["type"] == "message.proactive"
    assert envelopes[0]["payload"]["content"] == "看图"
    assert envelopes[0]["payload"]["delivery_id"] == "delivery-with-image"
    assert len(envelopes[0]["payload"]["attachments"]) == 1

    await channel.stop()
    storage.close()

    reopened = MobileRealtimeStorage(tmp_path / "mobile.db")
    assert (
        len(
            reopened.read_durable_events(
                device_ids[0],
                after_event_seq=0,
                limit=10,
            )
        )
        == 1
    )
    reopened.close()
    storage.close()


@pytest.mark.asyncio
async def test_send_and_turn_started_bind_each_client_message_id_per_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 1. 同 session 两条 send 各自持有 send 时间；turn.started 按同三元组绑定。
    ticks = iter((100.0, 101.0, 110.0, 111.0, 200.0, 203.0, 300.0, 303.0))
    monkeypatch.setattr(channel_module, "monotonic", lambda: next(ticks))
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    bus = _Bus()
    manager = SessionManager(tmp_path / "workspace")
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    session_id = f"akashic:{uuid4()}"
    first_id = "01ARZ3NDEKTSV4RRFFQ69G5FAA"
    second_id = "01ARZ3NDEKTSV4RRFFQ69G5FAB"
    with caplog.at_level(logging.INFO, logger="infra.mobile_realtime.channel"):
        first = await channel.handle_command(
            device_id=device_id,
            frame=_message_frame(frame_id=first_id, session_id=session_id),
        )
        second = await channel.handle_command(
            device_id=device_id,
            frame=_message_frame(frame_id=second_id, session_id=session_id),
        )
        assert first.type == "message.send.ok"
        assert second.type == "message.send.ok"
        assert len(bus.inbound) == 2
        assert channel._send_received_at == {
            (session_id, first_id): 100.0,
            (session_id, second_id): 110.0,
        }
        await channel._on_turn_started(
            TurnStarted(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                content="A",
                timestamp=datetime.now(timezone.utc),
                turn_id="turn-A",
                client_message_id=first_id,
            )
        )
        await channel._on_turn_started(
            TurnStarted(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                content="B",
                timestamp=datetime.now(timezone.utc),
                turn_id="turn-B",
                client_message_id=second_id,
            )
        )

    # 2. 同 session 两个 client_message_id 互不覆盖，duration 各自独立。
    started = [
        record
        for record in caplog.records
        if record.akashic_fields.get("event") == "tl:turn.started"
    ]
    by_cmid = {record.akashic_fields["client_message_id"]: record for record in started}
    assert set(by_cmid) == {first_id, second_id}
    assert by_cmid[first_id].akashic_fields["duration_ms"] == pytest.approx(103_000.0)
    assert by_cmid[second_id].akashic_fields["duration_ms"] == pytest.approx(193_000.0)
    acks = [
        record
        for record in caplog.records
        if record.akashic_fields.get("event") == "tl:send.ack"
    ]
    assert {record.akashic_fields["client_message_id"] for record in acks} == {
        first_id,
        second_id,
    }

    # 3. turn.started 事件载荷携带 client_message_id，Android 可绑定。
    started_payloads = [
        cast(dict[str, object], event["payload"])
        for event in runtime.events
        if event["event_type"] == "turn.started"
    ]
    assert started_payloads == [
        {"content": "A", "client_message_id": first_id, "control_turn_id": "turn-A"},
        {"content": "B", "client_message_id": second_id, "control_turn_id": "turn-B"},
    ]
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_terminal_and_stop_clear_send_and_turn_maps(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    device_id = uuid4().hex
    _register_device(storage, device_id)
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    manager.save(manager.get_or_create(session_id))
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    turn_id = "turn-A"
    channel._active_turn_ids[session_id] = turn_id
    channel._process_turns[(session_id, turn_id)] = channel_module._ProcessTurnState(
        next_ordinal=0,
        thinking_block=None,
        tool_blocks={},
        answer_segments=[],
        client_message_id="cmid-A",
    )
    channel._turn_started_at[(session_id, turn_id)] = 100.0
    channel._send_received_at[(session_id, "cmid-A")] = 50.0
    channel._send_received_at[(session_id, "cmid-B")] = 60.0
    channel._send_received_at[("akashic:other", "cmid-C")] = 70.0

    # 1. terminal（message.final）只清理 A 的 send/turn maps，
    #    同 session 排队中的 cmid-B 与其他会话条目必须保留。
    await _provider_delivery(channel)(
        OutboundMessage(
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="完成",
            control_turn_id=turn_id,
        )
    )
    assert channel._active_turn_ids == {}
    assert channel._process_turns == {}
    assert channel._turn_started_at == {}
    assert channel._send_received_at == {
        (session_id, "cmid-B"): 60.0,
        ("akashic:other", "cmid-C"): 70.0,
    }

    # 2. A/B overlap：B 已接替 active 时，迟到的 A cleanup 只清 A 自己的状态，
    #    绝不 compare-delete 掉 B 的 active，也不动 B 的 process/send 起点。
    overlap_turn = "turn-B"
    channel._active_turn_ids[session_id] = overlap_turn
    channel._process_turns[(session_id, overlap_turn)] = (
        channel_module._ProcessTurnState(
            next_ordinal=0,
            thinking_block=None,
            tool_blocks={},
            answer_segments=[],
            client_message_id="cmid-B",
        )
    )
    channel._turn_started_at[(session_id, overlap_turn)] = 200.0
    pending = channel_module._DeltaBatch(
        segments=[],
        byte_count=0,
        timer=asyncio.create_task(asyncio.sleep(10)),
    )
    channel._delta_batches[(session_id, turn_id)] = pending
    channel._clear_turn_maps(session_id, turn_id)
    assert channel._active_turn_ids == {session_id: overlap_turn}
    assert set(channel._process_turns) == {(session_id, overlap_turn)}
    assert channel._turn_started_at == {(session_id, overlap_turn): 200.0}
    assert channel._send_received_at == {
        (session_id, "cmid-B"): 60.0,
        ("akashic:other", "cmid-C"): 70.0,
    }
    assert channel._delta_batches == {}
    assert pending.timer.cancelling()
    _ = await asyncio.gather(pending.timer, return_exceptions=True)
    assert pending.timer.cancelled()

    # 3. stop() 清空剩余状态。
    await channel.stop()
    assert channel._send_received_at == {}
    assert channel._turn_started_at == {}
    assert channel._process_turns == {}
    assert channel._active_turn_ids == {}
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_terminal_final_publish_fail_once_is_retryable_without_fake_success(
    tmp_path: Path,
) -> None:
    """合同C1：message.final 持久化前失败不得写 closed、不得 cleanup、不得假成功。

    第一次异常原样上抛，key 不在 _turn_terminals，process state/client id 仍在；
    第二次同一 OutboundMessage 重试真实再次 publish_event，publish 调用数 2，
    durable events 严格 delta → message.final 且 final 恰一，cleanup 完成。
    """

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _FailOnceTerminalRuntime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-fail",
        )
    )
    for delta in ("你", "好"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta=delta,
            )
        )
        await asyncio.sleep(0)
    key = (session_id, turn_id)
    outbound = OutboundMessage(
        channel="akashic",
        chat_id=session_id.removeprefix("akashic:"),
        content="你好世界",
        control_turn_id=turn_id,
    )

    # 1. 第一次：publish 在持久化前抛 OSError——原样上抛、无墓碑、无 cleanup。
    with pytest.raises(OSError):
        await _provider_delivery(channel)(outbound)
    assert runtime.terminal_attempts == 1
    assert key not in channel._turn_terminals
    assert channel._process_turns[key].client_message_id == "cmid-fail"
    assert channel._process_turns[key].answer_segments == ["你", "好"]
    assert channel._process_turns[key].final_suffix_emitted == "世界"
    assert channel._active_turn_ids[session_id] == turn_id
    assert channel._delta_locks.get(key) is not None
    assert key in channel._turn_started_at
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
        "answer.delta",
        "answer.delta",
    ]

    # 2. 第二次：同 OutboundMessage 重试只补缺失 suffix（已 flush 则不再发），
    #    publish 调用数 2，wire 严格 已接受 delta → message.final，final 恰一。
    await _provider_delivery(channel)(outbound)
    assert runtime.terminal_attempts == 2
    events = runtime.events
    assert [event["event_type"] for event in events] == [
        "turn.started",
        "answer.delta",
        "answer.delta",
        "answer.delta",
        "message.final",
    ]
    assert sum(event["event_type"] == "message.final" for event in events) == 1
    deltas = "".join(
        cast(str, cast(dict[str, object], event["payload"])["delta"])
        for event in events
        if event["event_type"] == "answer.delta"
    )
    assert deltas == "你好世界"
    final_payload = cast(dict[str, object], events[-1]["payload"])
    assert final_payload["content"] == ""

    # 3. cleanup 完成：state/lock/batch 全清，墓碑提交。
    assert channel._process_turns == {}
    assert channel._active_turn_ids == {}
    assert channel._delta_batches == {}
    assert channel._delta_locks.get(key) is None
    assert key in channel._turn_terminals
    assert channel._delta_failure is None
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_terminal_failure_after_batch_flush_retry_does_not_duplicate_deltas(
    tmp_path: Path,
) -> None:
    """合同C2：终态 publish 失败前已 flush 的 delta 不回卷、不重复。

    残留批在失败前已发布；第二次同一 OutboundMessage 只补缺失 suffix 并发布
    terminal，wire 上每个 delta 恰一次，最终恰好一个 durable final。
    """

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _FailOnceTerminalRuntime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-batch",
        )
    )
    for delta in ("一", "二"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta=delta,
            )
        )
        await asyncio.sleep(0)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
    ]
    assert (session_id, turn_id) in channel._delta_batches
    outbound = OutboundMessage(
        channel="akashic",
        chat_id=session_id.removeprefix("akashic:"),
        content="一二终",
        control_turn_id=turn_id,
    )

    # 1. 第一次：残留批先 flush（"二" 发布），suffix "终" 入批 flush，然后
    #    message.final 持久化前失败。
    with pytest.raises(OSError):
        await _provider_delivery(channel)(outbound)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
        "answer.delta",
        "answer.delta",
    ]
    assert (
        "".join(
            cast(str, cast(dict[str, object], event["payload"])["delta"])
            for event in runtime.events
            if event["event_type"] == "answer.delta"
        )
        == "一二终"
    )

    # 2. 第二次：同一 OutboundMessage 不重复任何已 flush delta，只发布 terminal。
    await _provider_delivery(channel)(outbound)
    assert runtime.terminal_attempts == 2
    events = runtime.events
    assert [event["event_type"] for event in events] == [
        "turn.started",
        "answer.delta",
        "answer.delta",
        "answer.delta",
        "message.final",
    ]
    deltas = [
        cast(str, cast(dict[str, object], event["payload"])["delta"])
        for event in events
        if event["event_type"] == "answer.delta"
    ]
    assert deltas == ["一", "二", "终"]
    assert sum(event["event_type"] == "message.final" for event in events) == 1
    assert cast(dict[str, object], events[-1]["payload"])["content"] == ""
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_late_delta_queued_during_terminal_failure_gap_accepted_then_retry_closes(
    tmp_path: Path,
) -> None:
    """合同C3：失败间隙排队的 late delta 只能等失败释放锁后按 active 语义接受。

    第一次 final 持久化前挂起时无 closed 墓碑，late delta 真实排队在同一把
    per-turn 锁上；失败释放锁后被接受进 state；retry 发布 final 正文完整；
    terminal 之后才到的 delta 被 tombstone 丢弃。
    """

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _GatedFailOnceTerminalRuntime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-gap",
        )
    )
    for delta in ("你", "好"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta=delta,
            )
        )
        await asyncio.sleep(0)
    key = (session_id, turn_id)
    outbound = OutboundMessage(
        channel="akashic",
        chat_id=session_id.removeprefix("akashic:"),
        content="你好晚🙂",
        control_turn_id=turn_id,
    )

    # 1. 第一次 final 持久化前挂起（持锁）；late delta 排队等待同一把锁——
    #    真实 Event/锁编排：不等待锁、不因墓碑直接 drop。
    final_task = asyncio.create_task(_provider_delivery(channel)(outbound))
    await asyncio.wait_for(runtime.terminal_started.wait(), timeout=5)
    assert key not in channel._turn_terminals
    late_task = asyncio.create_task(
        channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta="晚",
            )
        )
    )
    await asyncio.sleep(0.01)
    assert not late_task.done()

    # 2. 放行：attempt 1 抛 OSError 释放锁，late delta 拿锁后按 active 语义接受。
    runtime.terminal_release.set()
    with pytest.raises(OSError):
        await asyncio.wait_for(final_task, timeout=5)
    await asyncio.wait_for(late_task, timeout=5)
    assert runtime.terminal_attempts == 1
    state = channel._process_turns[key]
    assert state.answer_segments == ["你", "好", "晚"]
    assert state.final_suffix_emitted == "晚🙂"

    # 3. retry：同一 OutboundMessage 真实再次 publish，final 正文完整、恰一次。
    await _provider_delivery(channel)(outbound)
    assert runtime.terminal_attempts == 2
    finals = [
        event for event in runtime.events if event["event_type"] == "message.final"
    ]
    assert len(finals) == 1
    assert cast(dict[str, object], finals[0]["payload"])["content"] == "你好晚🙂"
    wire = [event["event_type"] for event in runtime.events]
    assert wire[-1] == "message.final"
    assert all(event_type != "message.final" for event_type in wire[:-1])

    # 4. terminal 之后才到的 delta 被 tombstone 丢弃，锁/批零重建。
    before = len(runtime.events)
    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            turn_id=turn_id,
            content_delta="更晚",
        )
    )
    assert len(runtime.events) == before
    assert key in channel._turn_terminals
    assert channel._delta_locks.get(key) is None
    assert channel._delta_batches == {}
    assert channel._process_turns == {}
    assert channel._delta_failure is None
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_interrupted_terminal_publish_fail_once_is_retryable(
    tmp_path: Path,
) -> None:
    """合同C4：turn.interrupted 终态发布同样 fail-once 可重试。

    第一次异常原样上抛且不写 closed、不 cleanup；第二次重试真实再次
    publish 并收口，interrupted durable event 恰一次且 identity 贯通。
    """

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _FailOnceTerminalRuntime(storage, fail_type="turn.interrupted")
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = "turn-interrupt-retry"
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-stop",
        )
    )
    key = (session_id, turn_id)
    payload = channel._interrupt_payload(
        session_id,
        turn_id,
        status="interrupted",
        message="已停止",
    )

    # 1. 第一次：turn.interrupted 持久化前失败——异常上抛、无墓碑、无 cleanup。
    with pytest.raises(OSError):
        await channel._publish_terminal(
            session_id=session_id,
            turn_id=turn_id,
            event_type="turn.interrupted",
            payload=payload,
        )
    assert runtime.terminal_attempts == 1
    assert key not in channel._turn_terminals
    assert channel._process_turns[key].client_message_id == "cmid-stop"
    assert channel._active_turn_ids[session_id] == turn_id
    assert channel._delta_locks.get(key) is not None

    # 2. 第二次：重试真实再次 publish 并收口，durable interrupted 恰一次。
    assert (
        await channel._publish_terminal(
            session_id=session_id,
            turn_id=turn_id,
            event_type="turn.interrupted",
            payload=payload,
        )
        is True
    )
    assert runtime.terminal_attempts == 2
    interrupted = [
        event for event in runtime.events if event["event_type"] == "turn.interrupted"
    ]
    assert len(interrupted) == 1
    assert interrupted[0]["payload"] == {
        "status": "interrupted",
        "message": "已停止",
        "control_turn_id": turn_id,
        "client_message_id": "cmid-stop",
    }
    assert channel._process_turns == {}
    assert channel._active_turn_ids == {}
    assert channel._delta_locks.get(key) is None
    assert key in channel._turn_terminals
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_dual_terminal_race_only_lock_first_winner_publishes(
    tmp_path: Path,
) -> None:
    """合同C5：并发两个 terminal 竞争同一把 per-turn 锁，只有锁内第一个
    成功者提交墓碑与 durable event；第二个见 closed 不再发布，wire 恰一终态。"""

    runtime, channel, manager, storage, session_id, turn_id = await _race_channel(
        tmp_path
    )
    for delta in ("一", "二"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=session_id.removeprefix("akashic:"),
                turn_id=turn_id,
                content_delta=delta,
            )
        )
        await asyncio.sleep(0)
    key = (session_id, turn_id)
    lock = channel._delta_locks[key]
    await lock.acquire()
    first = asyncio.create_task(
        channel._publish_terminal(
            session_id=session_id,
            turn_id=turn_id,
            event_type="message.final",
            payload={"content": "终1"},
        )
    )
    await asyncio.sleep(0.01)
    second = asyncio.create_task(
        channel._publish_terminal(
            session_id=session_id,
            turn_id=turn_id,
            event_type="message.final",
            payload={"content": "终2"},
        )
    )
    await asyncio.sleep(0.01)
    lock.release()

    # 1. FIFO：第一个成功（flush 残留批 → publish → 墓碑），第二个返回 False。
    assert await asyncio.wait_for(first, timeout=5) is True
    assert await asyncio.wait_for(second, timeout=5) is False
    finals = [
        event for event in runtime.events if event["event_type"] == "message.final"
    ]
    assert len(finals) == 1
    assert finals[0]["payload"] == {"content": "终1"}
    # 输入只有 一、二 两段：wire 恰两条 answer.delta，payload 原序逐段一次。
    deltas = [
        cast(str, cast(dict[str, object], event["payload"])["delta"])
        for event in runtime.events
        if event["event_type"] == "answer.delta"
    ]
    assert deltas == ["一", "二"]
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "answer.delta",
        "answer.delta",
        "message.final",
    ]

    # 2. 唯一胜者提交墓碑并完成 cleanup。
    assert key in channel._turn_terminals
    assert channel._delta_locks.get(key) is None
    assert channel._delta_batches == {}
    assert channel._process_turns == {}
    assert channel._delta_failure is None
    await channel.stop()
    manager.close()
    storage.close()


async def _fail_delta_channel(
    tmp_path: Path,
    *,
    fail_at: int,
) -> tuple[
    _FailDeltaRuntime,
    MobileRealtimeChannel,
    SessionManager,
    MobileRealtimeStorage,
    str,
    str,
]:
    """起一轮 turn，delta 第 fail_at 次持久化前抛 OSError，用于逐段消费验证。"""

    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _FailDeltaRuntime(storage, fail_at=fail_at)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"akashic:{uuid4()}"
    turn_id = uuid4().hex
    await channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=_Bus(),
                session_manager=manager,
                event_bus=_EventBus(),
                push_tool=_PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(tmp_path / "uploads"),
            ),
        )
    )
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="akashic",
            chat_id=session_id.removeprefix("akashic:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
            client_message_id="cmid-fail-delta",
        )
    )
    return runtime, channel, manager, storage, session_id, turn_id


@pytest.mark.asyncio
async def test_flush_batch_first_segment_failure_keeps_full_batch_and_exact_bytes(
    tmp_path: Path,
) -> None:
    """首段持久化前失败：失败段与后续段原样保留、byte_count 精确；重试后
    wire 原序每段一次，成功段不重发。"""

    runtime, channel, manager, storage, session_id, turn_id = await _fail_delta_channel(
        tmp_path, fail_at=1
    )
    key = (session_id, turn_id)
    # 1. 经真实流式锁路径建立 per-turn 锁，再以独立身份段（merge=False）种批，
    #    保证失败段不与其他段合并。
    async with channel._delta_locked(session_id, turn_id):
        pass
    segments = ("第一段", "第二段", "第三段")
    for segment in segments:
        assert not channel._accept_segment_locked(
            session_id=session_id,
            turn_id=turn_id,
            event_type="answer.delta",
            delta=segment,
            block_id=None,
            ordinal=None,
            merge=False,
        )
    batch = channel._delta_batches[key]
    total_bytes = sum(len(segment.encode("utf-8")) for segment in segments)
    assert batch.byte_count == total_bytes
    assert [event["event_type"] for event in runtime.events] == ["turn.started"]

    # 2. 首段 publish 抛 OSError：批原样保留、byte_count 精确，wire 零 delta。
    with pytest.raises(OSError):
        await channel._flush_deltas(session_id, turn_id)
    assert runtime.delta_attempts == 1
    assert batch is channel._delta_batches[key]
    assert [segment[1] for segment in batch.segments] == [
        "第一段",
        "第二段",
        "第三段",
    ]
    assert batch.byte_count == total_bytes
    assert [event["event_type"] for event in runtime.events] == ["turn.started"]

    # 3. 重试全部成功：wire 原序每段一次，批 pop、timer 取消、无 failure。
    assert await channel._flush_deltas(session_id, turn_id) is True
    deltas = [
        cast(str, cast(dict[str, object], event["payload"])["delta"])
        for event in runtime.events
        if event["event_type"] == "answer.delta"
    ]
    assert deltas == ["第一段", "第二段", "第三段"]
    assert runtime.delta_attempts == 4
    assert channel._delta_batches == {}
    assert channel._delta_failure is None
    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_flush_batch_middle_segment_failure_consumes_only_successful_prefix(
    tmp_path: Path,
) -> None:
    """中段（第 2 段）持久化前失败：已发布首段不重发且精确扣减 byte_count，
    失败段与后续段保留；重试后 wire 原序每段一次。"""

    runtime, channel, manager, storage, session_id, turn_id = await _fail_delta_channel(
        tmp_path, fail_at=2
    )
    key = (session_id, turn_id)
    # 1. 经真实流式锁路径建立 per-turn 锁，再以独立身份段（merge=False）种批。
    async with channel._delta_locked(session_id, turn_id):
        pass
    segments = ("第一段", "第二段", "第三段")
    for segment in segments:
        assert not channel._accept_segment_locked(
            session_id=session_id,
            turn_id=turn_id,
            event_type="answer.delta",
            delta=segment,
            block_id=None,
            ordinal=None,
            merge=False,
        )
    batch = channel._delta_batches[key]
    total_bytes = sum(len(segment.encode("utf-8")) for segment in segments)
    assert batch.byte_count == total_bytes

    # 2. 第 2 段 publish 抛 OSError：首段已消费并扣减，失败段与后续段保留。
    with pytest.raises(OSError):
        await channel._flush_deltas(session_id, turn_id)
    assert runtime.delta_attempts == 2
    assert batch is channel._delta_batches[key]
    assert [segment[1] for segment in batch.segments] == ["第二段", "第三段"]
    assert batch.byte_count == total_bytes - len("第一段".encode("utf-8"))
    deltas = [
        cast(str, cast(dict[str, object], event["payload"])["delta"])
        for event in runtime.events
        if event["event_type"] == "answer.delta"
    ]
    assert deltas == ["第一段"]

    # 3. 重试只发布剩余两段：成功段不重发，wire 最终原序每段一次。
    assert await channel._flush_deltas(session_id, turn_id) is True
    deltas = [
        cast(str, cast(dict[str, object], event["payload"])["delta"])
        for event in runtime.events
        if event["event_type"] == "answer.delta"
    ]
    assert deltas == ["第一段", "第二段", "第三段"]
    assert runtime.delta_attempts == 4
    assert channel._delta_batches == {}
    assert channel._delta_failure is None
    await channel.stop()
    manager.close()
    storage.close()

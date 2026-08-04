from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
import infra.mobile_realtime.channel as channel_module
import infra.mobile_realtime.gateway as gateway_module

from agent.config_models import MobileRealtimeConfig
from infra.mobile_realtime.runtime_inspection import RuntimeInspectionService
from bus.events import (
    AttachmentKind,
    ChannelAttachment,
    ChannelMessage,
    DeliveryStatus,
    OutboundMessage,
)
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from infra.channels.base import AttachmentStore
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
    MobileRealtimeStorage,
)
from session.manager import SessionManager


class _Runtime:
    def __init__(self, storage: MobileRealtimeStorage) -> None:
        self.storage = storage
        self.config = MobileRealtimeConfig(max_attachment_mb=50)
        self.events: list[dict[str, object]] = []

    async def publish_event(self, **event: object) -> None:
        self.events.append(dict(event))

    async def publish_connection_control(self, **control: object) -> None:
        self.events.append(dict(control))

    async def publish_event_with_outbound_attachments(
        self,
        *,
        candidates: tuple[AttachmentRecord, ...],
        payload_builder: Any,
        session_id: str,
    ) -> tuple[AttachmentRecord, ...]:
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
        return resolved


class _Bus:
    def __init__(self) -> None:
        self.inbound: list[object] = []
        self.outbound: dict[str, object] = {}
        self.pending_handoff = False

    async def publish_inbound(self, message: object) -> None:
        self.inbound.append(message)

    def subscribe_outbound(self, channel: str, callback: object) -> None:
        self.outbound[channel] = callback

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


class _EventBus:
    def __init__(self) -> None:
        self.handlers: dict[type[object], object] = {}

    def on(self, event_type: type[object], handler: object) -> None:
        self.handlers[event_type] = handler


class _PushTool:
    def __init__(self) -> None:
        self.registered: dict[str, dict[str, object]] = {}

    def register_channel(self, channel: str, **senders: object) -> None:
        self.registered[channel] = senders


class _RuntimeInspection:
    def list_documents(self) -> dict[str, object]:
        return {"items": [{"id": "memory"}]}

    def get_document(self, document_id: str) -> dict[str, object]:
        return {"id": document_id, "markdown": "# Memory"}


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
        session_id="mobile:test",
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
            "id": "mobile:test:1",
            "session_key": "mobile:test",
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
                "session_key": "mobile:test",
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
        session_id="mobile:test",
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
                "session_key": "mobile:test",
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
        session_id="mobile:test",
        payload=payload,
    )

    assert len(encoded.encode("utf-8")) < 256 * 1024
    assert any("description" not in call for call in calls)


def test_mobile_history_cursor_shrinks_page_before_tool_details() -> None:
    items = [
        {
            "id": f"message-{seq}",
            "session_key": "mobile:test",
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
                "session_key": "mobile:test",
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
                    "media_refs": [],
                    "client_created_at": datetime.now(timezone.utc).isoformat(),
                    **({"reply_to": reply_to} if reply_to is not None else {}),
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
    session_id = f"mobile:{uuid4()}"
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

    assert first == duplicate
    assert first.type == "message.send.ok"
    assert len(bus.inbound) == 2
    assert shared.type == "message.send.ok"
    assert mismatched.type == "message.send.error"
    assert mismatched.payload["code"] == "client_message_id_mismatch"
    assert len(bus.inbound) == 2
    assert all(
        item.metadata["require_existing_session"] is True for item in bus.inbound
    )
    with pytest.raises(ValueError, match="正在处理消息"):
        manager.delete_session(session_id)
    for item in bus.inbound:
        assert item.session_admission_id is not None
        manager.release_admission(item.session_admission_id)
    assert manager.delete_session(session_id)
    with pytest.raises(KeyError, match="session 不存在"):
        manager.get_existing(session_id)
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
    session_id = f"mobile:{uuid4()}"
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
    session_id = f"mobile:{uuid4()}"

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
    session_id = f"mobile:{uuid4()}"
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

    assert reply.type == "message.send.error"
    assert reply.payload["code"] == "session_not_found"
    assert not original_exists(session_id)
    assert bus.inbound == []
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
    session_id = f"mobile:{uuid4()}"
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
    session_id = f"mobile:{uuid4()}"
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
        session_id=f"mobile:{uuid4()}",
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
    session_id = f"mobile:{uuid4()}"
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
    session_id = f"mobile:{uuid4()}"
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
    session_id = f"mobile:{uuid4()}"
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
            reply_to={
                "message_id": target["id"],
            },
        ),
    )

    assert reply.type == "message.send.ok"
    inbound = bus.inbound[0]
    assert f"被回复消息（来自 Akashic）：\n{target_content}" in inbound.content
    assert inbound.content.endswith("【你当前新消息】\n你好")
    assert inbound.metadata["display_content"] == "你好"
    assert inbound.metadata["reply_to_message_id"] == target["id"]
    assert inbound.metadata["reply_role"] == "assistant"
    assert inbound.metadata["reply_preview"] == " ".join(target_content.split())[:512]
    assert inbound.metadata["require_existing_session"] is True

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
    assert bus.inbound[1].metadata["require_existing_session"] is True
    assert bus.inbound[1].metadata["reply_to_message_id"] == user_target["id"]
    assert "被回复消息（来自 你）：\n之前的问题" in bus.inbound[1].content

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
    assert bus.inbound[2].metadata["reply_preview"] == "[附件]"
    assert "被回复消息（来自 你）：\n[附件]" in bus.inbound[2].content

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
    assert bus.inbound[3].metadata["reply_to_message_id"] == proactive_target["id"]
    assert (
        "被回复消息（来自 Akashic）：\n尚未同步历史的主动消息" in bus.inbound[3].content
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
    session_id = f"mobile:{uuid4()}"
    other = manager.get_or_create(f"mobile:{uuid4()}")
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
    session_id = f"mobile:{uuid4()}"
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
        llm_context_frame="private context",
        client_message_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        reply_to_message_id=f"{session_id}:0",
        reply_role="assistant",
        reply_preview="更早的回答",
    )
    session.add_message(
        "assistant",
        "历史回答",
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
    assert listed.payload["total"] == 1
    session_event = runtime.events[-2]
    assert session_event["event_type"] == "session.list"
    session_payload = cast(dict[str, object], session_event["payload"])
    session_items = cast(list[dict[str, object]], session_payload["items"])
    assert len(session_items) == 1
    assert session_items[0]["session_id"] == session_id
    assert session_items[0]["title"] == "恢复这段对话"
    assert history.type == "history.get.ok"
    history_event = runtime.events[-1]
    history_payload = cast(dict[str, object], history_event["payload"])
    history_items = cast(list[dict[str, object]], history_payload["items"])
    assert history_items[0]["extra"] == {}
    assert history_items[0]["client_message_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert history_items[0]["reply_to_message_id"] == f"{session_id}:0"
    assert history_items[0]["reply_role"] == "assistant"
    assert history_items[0]["reply_preview"] == "更早的回答"
    assert "llm_context_frame" not in history_items[0]
    assert history_items[1]["extra"] == {"reasoning_content": "历史思考"}
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
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            content="再次生成",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
    )
    session.add_message("assistant", "实时回答", media=[str(media_path)])
    manager.save(session)
    await channel._on_response(
        OutboundMessage(
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
        f"mobile:{uuid4()}",
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
    session_id = f"mobile:{uuid4()}"
    media = ["https://expired.example/reaction.gif"]
    session = manager.get_or_create(session_id)
    session.add_message("assistant", "文字仍应送达", media=media)
    manager.save(session)

    async def fail(*args: object, **kwargs: object) -> RemoteMediaSnapshot:
        raise RemoteMediaError("签名链接已失效")

    monkeypatch.setattr("infra.mobile_realtime.channel.snapshot_remote_media", fail)
    await channel._on_response(
        OutboundMessage(
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
    session_id = f"mobile:{uuid4()}"

    await channel._on_response(
        OutboundMessage(
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
async def test_control_reply_never_reuses_previous_message_id(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    manager = SessionManager(tmp_path / "workspace")
    session_id = f"mobile:{uuid4()}"
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

    await channel._on_response(
        OutboundMessage(
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
    session_id = f"mobile:{uuid4()}"
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
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
    session_id = f"mobile:{uuid4()}"
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
                mobile_bot_commands=[],
            ),
        )
    )
    turn_id = "01ARZ3NDEKTSV4RRFFQ69G5FAY"
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
async def test_command_list_uses_active_channel_catalog_without_stop(
    tmp_path: Path,
) -> None:
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
                mobile_bot_commands=[
                    ("undo", "撤销上一轮对话"),
                    ("/memorystatus", "查看记忆整理状态"),
                    ("emoji", "😀" * 129),
                    ("stop", "中断当前回复"),
                ],
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
                mobile_bot_commands=[("undo", "撤销上一轮对话")],
            ),
        )
    )
    session_id = f"mobile:{uuid4()}"

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
    assert cast(Any, bus.inbound[0]).content == "/undo"
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
    session_id = f"mobile:{uuid4()}"
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
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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

    await channel.stop()
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_delta_paths_reuse_existing_lock_without_allocating_lock() -> None:
    runtime = _Runtime(cast(MobileRealtimeStorage, object()))
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    key = ("mobile:test", "turn-1")
    existing_lock = asyncio.Lock()
    channel._delta_locks[key] = existing_lock

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
            "payload": {"delta": "x" * 4096},
        }
    ]


def test_stream_delta_flush_cadence_targets_60hz() -> None:
    assert channel_module._DELTA_FLUSH_INTERVAL_SECONDS == pytest.approx(1.0 / 60.0)


@pytest.mark.asyncio
async def test_stream_deltas_batch_within_one_frame_window_and_flush_before_tool_and_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((100.0, 105.125))
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
    session_id = f"mobile:{uuid4()}"
    storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    turn_id = uuid4().hex
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            content="帮我检查",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
    )

    for delta in ("思", "考", "中"):
        await channel._on_stream_delta(
            StreamDeltaReady(
                session_key=session_id,
                channel="mobile",
                chat_id=session_id.removeprefix("mobile:"),
                turn_id=turn_id,
                thinking_delta=delta,
            )
        )
        await asyncio.sleep(0.005)
    assert [event["event_type"] for event in runtime.events] == ["turn.started"]
    await asyncio.sleep(channel_module._DELTA_FLUSH_INTERVAL_SECONDS + 0.01)
    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "react.thinking.delta",
    ]
    first_thinking = cast(dict[str, object], runtime.events[1]["payload"])
    assert first_thinking["delta"] == "思考中"
    assert first_thinking["ordinal"] == 0

    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            turn_id=turn_id,
            content_delta="A" * 4096,
        )
    )
    await channel._on_tool_call_started(
        ToolCallStarted(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            turn_id=turn_id,
            thinking_delta="继续思考",
        )
    )
    await channel._on_response(
        OutboundMessage(
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            content="完成",
            thinking="思考中",
            metadata={"mobile_attention": "confirmation"},
            control_turn_id=turn_id,
        )
    )

    assert [event["event_type"] for event in runtime.events] == [
        "turn.started",
        "react.thinking.delta",
        "answer.delta",
        "react.tool.started",
        "react.tool.completed",
        "react.thinking.delta",
        "message.final",
    ]
    tool_started = cast(dict[str, object], runtime.events[3]["payload"])
    tool_completed = cast(dict[str, object], runtime.events[4]["payload"])
    second_thinking = cast(dict[str, object], runtime.events[5]["payload"])
    final_metadata = cast(dict[str, object], runtime.events[6]["payload"])["metadata"]
    assert tool_started["block_id"] == tool_completed["block_id"]
    assert tool_started["ordinal"] == tool_completed["ordinal"] == 1
    assert second_thinking["ordinal"] == 2
    assert second_thinking["block_id"] != first_thinking["block_id"]
    assert cast(dict[str, object], final_metadata)["mobile_attention"] == "confirmation"
    await channel.stop()
    storage.close()


@pytest.mark.asyncio
async def test_final_projects_only_explicit_mobile_metadata(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    runtime = _Runtime(storage)
    channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
    session_id = f"mobile:{uuid4()}"

    await channel._on_response(
        OutboundMessage(
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
    session_id = f"mobile:{uuid4()}"
    content = ("长回复🙂\n" * 150_000)[:1_000_000]

    await channel._on_response(
        OutboundMessage(
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
    session_id = f"mobile:{uuid4()}"
    turn_id = uuid4().hex
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
    )
    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            turn_id=turn_id,
            content_delta="你好",
        )
    )

    await channel._on_response(
        OutboundMessage(
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
    session_id = f"mobile:{uuid4()}"
    turn_id = uuid4().hex
    await channel._on_turn_started(
        TurnStarted(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            content="继续",
            timestamp=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
    )
    await channel._on_stream_delta(
        StreamDeltaReady(
            session_key=session_id,
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
            turn_id=turn_id,
            content_delta="草稿",
        )
    )

    await channel._on_response(
        OutboundMessage(
            channel="mobile",
            chat_id=session_id.removeprefix("mobile:"),
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
        session_id=f"mobile:{chat_id}",
        created_at=datetime.now(timezone.utc),
    )
    deliver = cast(Any, push.registered["mobile"]["deliver"])

    receipt = await deliver(
        ChannelMessage(
            channel="mobile",
            chat_id=chat_id,
            content="该休息一下了",
        )
    )

    assert receipt.status is DeliveryStatus.SUCCESS
    assert runtime.events == [
        {
            "event_type": "message.proactive",
            "session_id": f"mobile:{chat_id}",
            "payload": {
                "content": "该休息一下了",
                "attachments": [],
                "metadata": {"source": "message_push"},
            },
        }
    ]
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
    deliver = cast(Any, push.registered["mobile"]["deliver"])

    receipt = await deliver(
        ChannelMessage(
            channel="mobile",
            chat_id=chat_id,
            content="该休息一下了",
            metadata={"delivery_id": "delivery-1"},
        )
    )

    assert receipt.status is DeliveryStatus.SUCCESS
    assert runtime.events[-1]["payload"] == {
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
    deliver = cast(Any, push.registered["mobile"]["deliver"])

    receipt = await deliver(
        ChannelMessage(
            channel="mobile",
            chat_id=chat_id,
            content="看图",
            attachments=(
                ChannelAttachment(AttachmentKind.IMAGE, str(source)),
            ),
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
    assert len(
        reopened.read_durable_events(
            device_ids[0],
            after_event_seq=0,
            limit=10,
        )
    ) == 1
    reopened.close()
    storage.close()

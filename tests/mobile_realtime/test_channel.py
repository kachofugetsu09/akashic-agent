from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest

from agent.config_models import MobileRealtimeConfig
from bus.events import OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from infra.channels.base import AttachmentStore
from infra.mobile_realtime.channel import MobileRealtimeChannel
from infra.mobile_realtime.gateway import MobileGatewayRuntime
from infra.mobile_realtime.protocol import GenericCommand, MessageSendCommand, parse_frame
from infra.mobile_realtime.remote_media import RemoteMediaError, RemoteMediaSnapshot
from infra.mobile_realtime.storage import DeviceRecord, MobileRealtimeStorage
from session.manager import SessionManager


class _Runtime:
    def __init__(self, storage: MobileRealtimeStorage) -> None:
        self.storage = storage
        self.config = MobileRealtimeConfig(max_attachment_mb=50)
        self.events: list[dict[str, object]] = []

    async def publish_event(self, **event: object) -> None:
        self.events.append(dict(event))


class _Bus:
    def __init__(self) -> None:
        self.inbound: list[object] = []
        self.outbound: dict[str, object] = {}

    async def publish_inbound(self, message: object) -> None:
        self.inbound.append(message)

    def subscribe_outbound(self, channel: str, callback: object) -> None:
        self.outbound[channel] = callback


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


def _message_frame(
    *,
    frame_id: str,
    session_id: str,
    epoch: int = 1,
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
                    "text": "你好",
                    "media_refs": [],
                    "client_created_at": datetime.now(timezone.utc).isoformat(),
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
                        "arguments": {"description": "读取状态", "secret": "hidden"},
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
    assert "llm_context_frame" not in history_items[0]
    assert history_items[1]["extra"] == {"reasoning_content": "历史思考"}
    tool_chain = cast(list[dict[str, object]], history_items[1]["tool_chain"])
    calls = cast(list[dict[str, object]], tool_chain[0]["calls"])
    assert calls[0]["description"] == "读取状态"
    assert "secret" not in calls[0]
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
        request_interrupt=lambda **_: SimpleNamespace(status="accepted", message="已停止"),
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

    reply = await channel.handle_command(
        device_id=current_device,
        frame=_generic_frame(
            frame_id="01ARZ3NDEKTSV4RRFFQ69G5FAZ",
            command_type="turn.stop",
            session_id=session_id,
        ),
    )

    assert reply.type == "turn.stop.ok"
    assert runtime.events[-1]["event_type"] == "turn.interrupted"
    manager.close()
    storage.close()


@pytest.mark.asyncio
async def test_stream_deltas_batch_at_50ms_and_flush_before_tool_and_final(
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
    assert [event["event_type"] for event in runtime.events] == ["turn.started"]
    await asyncio.sleep(0.07)
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
            arguments={"command": "pwd"},
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
            final_arguments={"command": "pwd"},
            status="completed",
            result_preview="ok",
            turn_id=turn_id,
        )
    )
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
    assert tool_started["block_id"] == tool_completed["block_id"]
    assert tool_started["ordinal"] == tool_completed["ordinal"] == 1
    assert second_thinking["ordinal"] == 2
    assert second_thinking["block_id"] != first_thinking["block_id"]
    await channel.stop()
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
    sender = cast(Any, push.registered["mobile"]["text"])

    await sender(chat_id, "该休息一下了")

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

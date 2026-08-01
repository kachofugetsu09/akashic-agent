from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState

from bootstrap.chat_api import create_chat_app
from bus.events import OutboundMessage
from infra.channels.base import AttachmentStore
from infra.channels.web_chat_channel import UploadTooLargeError, WebChatChannel
from session.manager import Session


class _Bus:
    def __init__(self) -> None:
        self.inbound: list[Any] = []
        self.subscribers: dict[str, Any] = {}

    async def publish_inbound(self, msg: Any) -> None:
        self.inbound.append(msg)

    def subscribe_outbound(self, channel: str, callback: Any) -> None:
        self.subscribers[channel] = callback


class _EventBus:
    def __init__(self) -> None:
        self.handlers: dict[type[object], Any] = {}

    def on(self, event_type: type[object], handler: Any) -> None:
        self.handlers[event_type] = handler


class _PushTool:
    def __init__(self) -> None:
        self.registered: dict[str, Any] = {}

    def register_channel(self, channel: str, **senders: Any) -> None:
        self.registered[channel] = senders


class _WebSocket:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.application_state = WebSocketState.CONNECTED

    async def send_json(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)


class _SessionManager:
    def __init__(self) -> None:
        self.saved: list[Any] = []
        self.sessions: dict[str, Session] = {}
        self.appended: list[tuple[Session, list[dict[str, Any]]]] = []
        self._store = _SessionStore()

    def get_or_create(self, key: str) -> Any:
        self.sessions.setdefault(key, Session(key=key))
        return self.sessions[key]

    async def save_async(self, session: Any) -> None:
        self.saved.append(session)

    async def append_messages(self, session: Session, messages: list[dict[str, Any]]) -> None:
        self.appended.append((session, list(messages)))

    @property
    def control_store(self) -> _SessionStore:
        return self._store


class _SessionStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages: dict[str, dict[str, Any]] = {}

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        return self.messages.get(message_id)

    def list_sessions_for_dashboard(self, **_: Any) -> tuple[list[dict[str, Any]], int]:
        return [], 0

    def list_messages_for_dashboard(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        self.calls.append(kwargs)
        return [
            {"id": "m0", "role": "user", "content": "用户问题"},
            {"id": "m1", "role": "assistant", "content": "助手回答"},
        ], 2


@pytest.mark.asyncio
async def test_web_chat_session_and_message_flow(tmp_path: Path) -> None:
    bus = _Bus()
    session_manager = _SessionManager()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=bus,
        session_manager=session_manager,
        event_bus=_EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    app = create_chat_app(workspace=tmp_path, channel=channel)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.create", "request_id": "r1"})
            created = ws.receive_json()
            session_id = created["session_id"]

            ws.send_json({
                "type": "message.send",
                "request_id": "r2",
                "session_id": session_id,
                "text": "你好",
                "media": [],
            })

    assert created["type"] == "session.created"
    assert str(session_id).startswith("web:")
    assert session_manager.saved == []
    assert len(bus.inbound) == 1
    assert bus.inbound[0].content == "你好"
    assert bus.inbound[0].session_key == session_id


@pytest.mark.asyncio
async def test_web_chat_message_send_can_create_session_without_persisting_empty_one(tmp_path: Path) -> None:
    bus = _Bus()
    session_manager = _SessionManager()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=bus,
        session_manager=session_manager,
        event_bus=_EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    app = create_chat_app(workspace=tmp_path, channel=channel)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "message.send",
                "request_id": "r1",
                "text": "你好",
                "media": [],
            })

    assert session_manager.saved == []
    assert len(bus.inbound) == 1
    assert bus.inbound[0].content == "你好"
    assert str(bus.inbound[0].session_key).startswith("web:")


@pytest.mark.asyncio
async def test_web_chat_message_send_resolves_canonical_reply(tmp_path: Path) -> None:
    bus = _Bus()
    session_manager = _SessionManager()
    session_manager._store.messages["m1"] = {
        "id": "m1",
        "session_key": "web:abc",
        "role": "assistant",
        "content": "先前回答",
    }
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=bus,
        session_manager=session_manager,
        event_bus=_EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    app = create_chat_app(workspace=tmp_path, channel=channel)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "message.send",
                "request_id": "reply-1",
                "session_id": "web:abc",
                "text": "继续说明",
                "media": [],
                "reply_to_message_id": "m1",
            })

    assert len(bus.inbound) == 1
    inbound = bus.inbound[0]
    assert "先前回答" in inbound.content
    assert "继续说明" in inbound.content
    assert inbound.metadata == {
        "client_request_id": "reply-1",
        "display_content": "继续说明",
        "reply_to_message_id": "m1",
        "reply_role": "assistant",
        "reply_preview": "先前回答",
    }


@pytest.mark.asyncio
async def test_web_chat_message_send_rejects_invalid_reply_target(tmp_path: Path) -> None:
    bus = _Bus()
    session_manager = _SessionManager()
    session_manager._store.messages["other"] = {
        "id": "other",
        "session_key": "web:other",
        "role": "user",
        "content": "其他会话",
    }
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=bus,
        session_manager=session_manager,
        event_bus=_EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    app = create_chat_app(workspace=tmp_path, channel=channel)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "message.send",
                "request_id": "bad-reply",
                "session_id": "web:abc",
                "text": "继续",
                "media": [],
                "reply_to_message_id": "other",
            })
            assert ws.receive_json() == {
                "type": "error",
                "request_id": "bad-reply",
                "message": "不能引用其他会话的消息",
            }

    assert bus.inbound == []


def test_chat_navigation_uses_explicit_public_dashboard_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_DASHBOARD_PUBLIC_PORT", "19321")
    app = create_chat_app(workspace=tmp_path, channel=WebChatChannel())

    with TestClient(app) as client:
        response = client.get("/api/chat/navigation")

    assert response.json() == {"dashboard_port": 19321}


@pytest.mark.asyncio
async def test_web_chat_rejects_malformed_fields_without_closing_connection(tmp_path: Path) -> None:
    bus = _Bus()
    session_manager = _SessionManager()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=bus,
        session_manager=session_manager,
        event_bus=_EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    app = create_chat_app(workspace=tmp_path, channel=channel)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "message.send",
                "request_id": "bad",
                "text": "你好",
                "media": "not-a-list",
            })
            error = ws.receive_json()
            assert error == {
                "type": "error",
                "request_id": "bad",
                "message": "media 必须是数组",
            }

            ws.send_json({
                "type": "message.send",
                "request_id": "bad-text",
                "text": 123,
                "media": [],
            })
            assert ws.receive_json() == {
                "type": "error",
                "request_id": "bad-text",
                "message": "text 必须是字符串",
            }

            ws.send_json({
                "type": "message.send",
                "request_id": "bad-media-item",
                "text": "你好",
                "media": ["ok.png", 123],
            })
            assert ws.receive_json() == {
                "type": "error",
                "request_id": "bad-media-item",
                "message": "media 必须是字符串数组",
            }

            ws.send_json({"type": "ping", "request_id": "ping-1"})
            assert ws.receive_json() == {"type": "pong", "request_id": "ping-1"}

            ws.send_json({
                "type": "message.send",
                "request_id": "good",
                "text": "继续",
                "media": [],
            })

    assert len(bus.inbound) == 1
    assert bus.inbound[0].content == "继续"


def test_chat_upload_returns_local_path(tmp_path: Path) -> None:
    channel = WebChatChannel()
    app = create_chat_app(workspace=tmp_path, channel=channel)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/uploads",
            params={"filename": "note.txt"},
            content=b"hello",
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["filename"] == "note.txt"
    assert Path(payload["upload_path"]).is_file()
    assert payload["upload_url"].startswith("/api/chat/media?path=")


@pytest.mark.asyncio
async def test_chat_upload_stream_rejects_before_publishing_partial_file(tmp_path: Path) -> None:
    channel = WebChatChannel()
    store = AttachmentStore(tmp_path / "uploads")
    channel._attachments = store

    async def _chunks():
        yield b"ab"
        yield b"cd"

    with pytest.raises(UploadTooLargeError):
        await channel.save_upload_stream(_chunks(), "note.txt", max_bytes=3)

    upload_root = tmp_path / "uploads"
    assert not list(upload_root.glob("*.part"))
    assert not list(upload_root.glob("web_*"))


@pytest.mark.asyncio
async def test_chat_upload_cancel_cleans_staging_without_swallowing_cancel(
    tmp_path: Path,
) -> None:
    channel = WebChatChannel()
    channel._attachments = AttachmentStore(tmp_path / "uploads")

    async def _chunks():
        yield b"ab"
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await channel.save_upload_stream(_chunks(), "note.txt")
    assert not list((tmp_path / "uploads").glob("*.part"))


def test_chat_media_reads_uploaded_file(tmp_path: Path) -> None:
    channel = WebChatChannel()
    channel._attachments = AttachmentStore(tmp_path / "uploads")
    app = create_chat_app(workspace=tmp_path, channel=channel)

    with TestClient(app) as client:
        upload = client.post(
            "/api/chat/uploads",
            params={"filename": "note.txt"},
            content=b"hello",
        ).json()
        response = client.get("/api/chat/media", params={"path": upload["upload_path"]})

    assert response.status_code == 200
    assert response.content == b"hello"


def test_chat_media_rejects_outside_upload_root(tmp_path: Path) -> None:
    channel = WebChatChannel()
    channel._attachments = AttachmentStore(tmp_path / "uploads")
    app = create_chat_app(workspace=tmp_path, channel=channel)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with TestClient(app) as client:
        response = client.get("/api/chat/media", params={"path": str(outside)})

    assert response.status_code == 404


def test_chat_media_reads_registered_outbound_file(tmp_path: Path) -> None:
    channel = WebChatChannel()
    channel._attachments = AttachmentStore(tmp_path / "uploads")
    app = create_chat_app(workspace=tmp_path, channel=channel)
    outside = tmp_path / "outside" / "meme.png"
    outside.parent.mkdir()
    outside.write_bytes(b"image")
    channel.remember_media([str(outside)])

    with TestClient(app) as client:
        response = client.get("/api/chat/media", params={"path": str(outside)})

    assert response.status_code == 200
    assert response.content == b"image"


def test_chat_messages_default_to_turn_order(tmp_path: Path) -> None:
    channel = WebChatChannel()
    session_manager = _SessionManager()
    channel._ctx = cast(Any, SimpleNamespace(session_manager=session_manager))
    app = create_chat_app(workspace=tmp_path, channel=channel)

    with TestClient(app) as client:
        response = client.get("/api/chat/sessions/web:abc/messages")

    payload = response.json()
    assert [item["role"] for item in payload["items"]] == ["user", "assistant"]
    assert session_manager._store.calls[0]["sort_by"] == "seq"
    assert session_manager._store.calls[0]["sort_order"] == "asc"


@pytest.mark.asyncio
async def test_web_message_push_image_only_broadcasts_realtime_frame(tmp_path: Path) -> None:
    session_manager = _SessionManager()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=_Bus(),
        session_manager=session_manager,
        event_bus=_EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    image = tmp_path / "meme.png"
    image.write_bytes(b"image")

    await channel.send_image("abc", str(image))

    assert "web:abc" not in session_manager.sessions
    assert session_manager.appended == []


@pytest.mark.asyncio
async def test_web_final_preserves_full_outbound_projection(tmp_path: Path) -> None:
    channel = WebChatChannel()
    socket = _WebSocket()
    channel._connections["web:abc"] = {cast(Any, socket)}
    channel._active_turn_ids["web:abc"] = "turn-1"
    image = tmp_path / "result.png"
    image.write_bytes(b"image")

    await channel._on_response(
        OutboundMessage(
            channel="web",
            chat_id="abc",
            content="answer",
            thinking="reasoning",
            media=[str(image)],
            metadata={"render": "card", "turn_duration_ms": 17},
        )
    )

    assert socket.frames == [
        {
            "type": "message.final",
            "session_id": "web:abc",
            "turn_id": "turn-1",
            "content": "answer",
            "thinking": "reasoning",
            "media": [str(image)],
            "duration_ms": 17,
            "metadata": {"render": "card", "turn_duration_ms": 17},
        }
    ]
    assert channel.has_media(image)

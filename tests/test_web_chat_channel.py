from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from bootstrap.chat_api import create_chat_app
from infra.channels.base import AttachmentStore
from infra.channels.web_chat_channel import WebChatChannel


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


class _SessionManager:
    def __init__(self) -> None:
        self.saved: list[Any] = []

    def get_or_create(self, key: str) -> Any:
        return SimpleNamespace(key=key, metadata={})

    async def save_async(self, session: Any) -> None:
        self.saved.append(session)


@pytest.mark.asyncio
async def test_web_chat_session_and_message_flow(tmp_path: Path) -> None:
    bus = _Bus()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=bus,
        session_manager=_SessionManager(),
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
    assert len(bus.inbound) == 1
    assert bus.inbound[0].content == "你好"
    assert bus.inbound[0].session_key == session_id


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

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState

from bootstrap.chat_api import create_chat_app
from agent.plugin_composition import (
    CapabilitySources,
    ConnectionDescriptor,
    ModelAvailability,
    ModelCapabilities,
    ModelCatalogSnapshot,
    ModelDescriptor,
    ModelKind,
    ModelRole,
)
from agent.plugin_composition.channels import (
    AttachmentKind as V3AttachmentKind,
    AttachmentRef,
    ChannelCommitRole,
    ChannelRuntimePorts,
    ChannelFactoryContext,
    DeliveryStatus as V3DeliveryStatus,
    InboundEnvelope,
    InboundOwner,
    ProviderDeliveryRequest,
    RawInbound,
)
from agent.plugins.manager import PluginManager
from agent.plugins.model_catalog import ModelCatalogUnavailable
from bus.events import AttachmentKind, ChannelAttachment, ChannelMessage
from bus.event_bus import EventBus
from bus.events_lifecycle import StreamDeltaReady, TurnOutputCompleted, TurnStarted
from bus.queue import MessageBus
from infra.channels.base import AttachmentStore
from infra.channels.web_chat_channel import UploadTooLargeError, WebChatChannel
from bootstrap.core_channel_adapter import build_core_channel_definition
from session.manager import Session, SessionManager


class _Bus:
    def __init__(self) -> None:
        self.inbound: list[Any] = []

    async def publish_inbound(self, msg: Any) -> None:
        self.inbound.append(msg)


class _EventBus:
    def __init__(self) -> None:
        self.handlers: dict[type[object], Any] = {}

    def on(self, event_type: type[object], handler: Any) -> None:
        self.handlers[event_type] = handler


class _PushTool:
    pass


class _WebSocket:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.application_state = WebSocketState.CONNECTED

    async def send_json(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)


class _FailingWebSocket(_WebSocket):
    async def send_json(self, frame: dict[str, Any]) -> None:
        _ = frame
        raise OSError("socket closed")


@pytest.mark.asyncio
async def test_web_socket_projects_only_one_current_session() -> None:
    channel = WebChatChannel()
    switching_socket = _WebSocket()
    other_socket = _WebSocket()

    assert await channel._add_connection(
        "akashic:first", cast(Any, switching_socket)
    ) is True
    assert await channel._add_connection(
        "akashic:first", cast(Any, other_socket)
    ) is True
    assert await channel._add_connection(
        "akashic:second", cast(Any, switching_socket)
    ) is True

    assert channel._connections == {
        "akashic:first": {other_socket},
        "akashic:second": {switching_socket},
    }

    await channel._remove_connection(cast(Any, switching_socket))
    assert channel._connections == {"akashic:first": {other_socket}}


class _ProviderClientFactory:
    async def create(self, credentials: Any) -> Any:
        _ = credentials
        raise AssertionError("Web native adapter 不应创建 provider client")

    async def aclose(self) -> None:
        return None


class _AttachmentLease:
    def __init__(self, ref: AttachmentRef) -> None:
        self.ref = ref
        self.closed = False

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        _ = max_bytes
        return b"artifact"

    async def aclose(self) -> None:
        self.closed = True


class _AttachmentRead:
    def __init__(self) -> None:
        self.leases: list[_AttachmentLease] = []

    async def acquire(self, ref: AttachmentRef) -> _AttachmentLease:
        lease = _AttachmentLease(ref)
        self.leases.append(lease)
        return lease


class _Inbound:
    def __init__(self, *, block: bool = False) -> None:
        self.messages: list[RawInbound] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not block:
            self.release.set()

    async def admit(self, raw: RawInbound) -> bool:
        self.messages.append(raw)
        self.started.set()
        await self.release.wait()
        return True


def _v3_context(
    read: _AttachmentRead | None = None,
    *,
    ingress: _Inbound | None = None,
) -> ChannelFactoryContext:
    return ChannelFactoryContext(
        snapshot_id="snapshot-1",
        generation_id="generation-1",
        binding_token="binding-1",
        config={},
        credentials={},
        provider_client_factory=cast(Any, _ProviderClientFactory()),
        ingress=cast(Any, ingress),
        identity=None,
        attachment_read=cast(Any, read),
    )


async def _open_inbound_adapter(
    channel: WebChatChannel,
    ingress: _Inbound,
    *,
    read: _AttachmentRead | None = None,
    binding_token: str = "binding-1",
) -> Any:
    context = ChannelFactoryContext(
        snapshot_id="snapshot-1",
        generation_id="generation-1",
        binding_token=binding_token,
        config={},
        credentials={},
        provider_client_factory=cast(Any, _ProviderClientFactory()),
        ingress=cast(Any, ingress),
        identity=None,
        attachment_read=cast(Any, read),
    )
    adapter = channel.build_v3_adapter(context)
    await adapter.start()
    adapter.attach_runtime(ChannelRuntimePorts(
        snapshot_id=context.snapshot_id,
        generation_id=context.generation_id,
        binding_token=context.binding_token,
        ingress=cast(Any, ingress),
        identity=None,
        attachment_import=None,
    ))
    adapter.open_admission()
    return adapter


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

    def list_chat_history_page(
        self,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        self.calls.append(kwargs)
        return [
            {"id": "m0", "seq": 8, "role": "user", "content": "用户问题"},
            {"id": "m1", "seq": 9, "role": "assistant", "content": "助手回答"},
        ], 12, True


class _PluginUiProvider:
    def __init__(self) -> None:
        self.queries: list[dict[str, object]] = []

    def catalog(self) -> dict[str, object]:
        return {
            "catalog_revision": "a" * 64,
            "items": [{
                "id": "akasha",
                "revision": "revision-1",
                "module_sha256": "b" * 64,
                "stylesheet_sha256": None,
                "navigation": {"label": "Akasha Inspector", "description": "移动端独立页面"},
                "slots": ["turn.before_reasoning"],
            }],
        }

    def asset(
        self,
        plugin_id: str,
        plugin_revision: str,
        kind: str,
        sha256: str,
    ) -> dict[str, object]:
        return {
            "plugin_id": plugin_id,
            "plugin_revision": plugin_revision,
            "kind": kind,
            "sha256": sha256,
            "content": "export default {};",
        }

    async def query(
        self,
        plugin_id: str,
        plugin_revision: str,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        self.queries.append({
            "plugin_id": plugin_id,
            "plugin_revision": plugin_revision,
            "method": method,
            "payload": payload,
            "session_id": session_id,
            "turn_id": turn_id,
        })
        return {"left": [], "right": []}


@pytest.mark.asyncio
async def test_web_chat_session_and_message_flow(tmp_path: Path) -> None:
    bus = _Bus()
    ingress = _Inbound()
    session_manager = _SessionManager()
    channel = WebChatChannel()
    adapter = await _open_inbound_adapter(channel, ingress)
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
                "model_runtime_id": "runtime-b",
            })

    assert created["type"] == "session.created"
    assert str(session_id).startswith("akashic:")
    assert session_manager.saved == []
    assert bus.inbound == []
    assert len(ingress.messages) == 1
    inbound = ingress.messages[0]
    assert inbound.message.content == "你好"
    assert inbound.message.chat_id == session_id.removeprefix("akashic:")
    assert inbound.message.metadata["model_runtime_id"] == "runtime-b"
    assert inbound.message.attachments == ()
    await adapter.stop()


def test_web_ui_bootstrap_returns_one_complete_uncached_payload(tmp_path: Path) -> None:
    payload = (
        b'{"catalogId":"catalog-a","modules":[],"schemaVersion":1,'
        b'"snapshotId":"snapshot-a"}'
    )

    class Provider:
        unavailable = False

        async def bootstrap(self) -> bytes:
            if self.unavailable:
                raise RuntimeError("snapshot is switching")
            return payload

        async def state(self) -> dict[str, str]:
            if self.unavailable:
                raise RuntimeError("snapshot is switching")
            return {"snapshotId": "snapshot-a", "catalogId": "catalog-a"}

    provider = Provider()
    app = create_chat_app(
        workspace=tmp_path,
        channel=WebChatChannel(),
        web_ui_provider=cast(Any, provider),
    )

    with TestClient(app) as client:
        response = client.get("/api/chat/web-ui/bootstrap")
        state = client.get("/api/chat/web-ui/state")
        provider.unavailable = True
        unavailable = client.get("/api/chat/web-ui/bootstrap")

    assert response.content == payload
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert state.json() == {"snapshotId": "snapshot-a", "catalogId": "catalog-a"}
    assert unavailable.status_code == 503


def test_chat_model_catalog_reports_session_override(tmp_path: Path) -> None:
    channel = WebChatChannel()
    sessions = _SessionManager()
    session = sessions.get_or_create("akashic:abc")
    session.metadata["model_runtime_override"] = "runtime-b"
    channel._ctx = cast(Any, SimpleNamespace(session_manager=sessions))
    catalog = ModelCatalogSnapshot(
        revision=7,
        connections=(
            ConnectionDescriptor(
                connection_id="source-a",
                name="OpenAI",
                driver_id="openai-compatible",
                auth_identity="account-a",
                availability=ModelAvailability.AVAILABLE,
            ),
        ),
        models=(
            ModelDescriptor(
                model_id="runtime-a",
                connection_id="source-a",
                kind=ModelKind.CHAT,
                model="model-a",
                default_reasoning_effort=None,
                capabilities=ModelCapabilities(),
                capability_sources=CapabilitySources(),
                availability=ModelAvailability.AVAILABLE,
            ),
        ),
        role_bindings={ModelRole.DEFAULT: "runtime-a"},
        default_embedding_model_id=None,
    )

    async def read_catalog() -> ModelCatalogSnapshot:
        return catalog

    app = create_chat_app(
        workspace=tmp_path,
        channel=channel,
        model_catalog_reader=read_catalog,
    )

    response = TestClient(app).get(
        "/api/chat/models",
        params={"session_key": "akashic:abc"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "generationId": 7,
            "defaultRuntime": "runtime-a",
            "sessionOverride": "runtime-b",
            "sessionSelection": {
                "modelRef": "runtime-b",
                "reasoningEffort": "",
            },
            "runtimes": [
                {
                    "id": "runtime-a",
                    "provider": "openai-compatible",
                    "catalogProvider": "openai-compatible",
                    "model": "model-a",
                    "reasoningEffort": "",
                    "supportedReasoningEfforts": [],
                    "sourceId": "source-a",
                    "sourceName": "OpenAI",
                    "contextWindow": 0,
                    "maxOutputTokens": 0,
                    "inputModalities": ["text"],
                    "capabilitySource": "unknown",
                    "capabilitySources": {
                        "contextWindow": "unknown",
                        "maxOutputTokens": "unknown",
                        "inputModalities": "unknown",
                    },
                    "roles": ["default"],
                }
            ],
        }


def test_chat_model_catalog_maps_missing_models_plugin_to_503(
    tmp_path: Path,
) -> None:
    async def unavailable() -> ModelCatalogSnapshot:
        raise ModelCatalogUnavailable("models plugin missing")

    app = create_chat_app(
        workspace=tmp_path,
        channel=WebChatChannel(),
        model_catalog_reader=unavailable,
    )
    response = TestClient(app).get("/api/chat/models")
    assert response.status_code == 503
    assert response.json() == {"detail": "模型注册表不可用"}


def test_web_plugin_ui_exposes_shared_slots_but_rejects_dashboard_query(
    tmp_path: Path,
) -> None:
    channel = WebChatChannel()
    provider = _PluginUiProvider()
    app = create_chat_app(
        workspace=tmp_path,
        channel=channel,
        plugin_ui_provider=cast(Any, provider),
    )

    with TestClient(app) as client:
        catalog = client.get("/api/chat/plugin-ui/catalog")
        asset = client.get(
            "/api/chat/plugin-ui/asset",
            params={
                "plugin_id": "akasha",
                "plugin_revision": "revision-1",
                "kind": "module",
                "sha256": "b" * 64,
            },
        )
        query = client.post(
            "/api/chat/plugin-ui/query",
            json={
                "plugin_id": "akasha",
                "plugin_revision": "revision-1",
                "method": "recall.current",
                "payload": {"message_id": "assistant:turn-1"},
                "slot": "turn.before_reasoning",
                "session_id": "akashic:abc",
                "turn_id": "turn-1",
            },
        )
        dashboard_query = client.post(
            "/api/chat/plugin-ui/query",
            json={
                "plugin_id": "akasha",
                "plugin_revision": "revision-1",
                "method": "inspector.recent",
                "payload": {},
                "slot": "dashboard.main",
            },
        )

    assert catalog.status_code == 200
    assert catalog.json()["items"][0]["navigation"]["label"] == "Akasha Inspector"
    assert asset.status_code == 200
    assert asset.headers["content-type"] == "text/javascript; charset=utf-8"
    assert asset.headers["cache-control"] == "private, max-age=31536000, immutable"
    assert query.status_code == 200
    assert query.json() == {"left": [], "right": []}
    assert provider.queries == [{
        "plugin_id": "akasha",
        "plugin_revision": "revision-1",
        "method": "recall.current",
        "payload": {"message_id": "assistant:turn-1"},
        "session_id": "akashic:abc",
        "turn_id": "turn-1",
    }]
    assert dashboard_query.status_code == 422


@pytest.mark.asyncio
async def test_web_chat_message_send_can_create_session_without_persisting_empty_one(tmp_path: Path) -> None:
    bus = _Bus()
    ingress = _Inbound()
    session_manager = _SessionManager()
    channel = WebChatChannel()
    adapter = await _open_inbound_adapter(channel, ingress)
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
    assert bus.inbound == []
    assert len(ingress.messages) == 1
    assert ingress.messages[0].message.content == "你好"
    assert ingress.messages[0].message.chat_id
    await adapter.stop()


@pytest.mark.asyncio
async def test_web_chat_message_send_resolves_canonical_reply(tmp_path: Path) -> None:
    bus = _Bus()
    ingress = _Inbound()
    session_manager = _SessionManager()
    session_manager._store.messages["m1"] = {
        "id": "m1",
        "session_key": "akashic:abc",
        "role": "assistant",
        "content": "先前回答",
    }
    channel = WebChatChannel()
    adapter = await _open_inbound_adapter(channel, ingress)
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
                "session_id": "akashic:abc",
                "text": "继续说明",
                "media": [],
                "reply_to_message_id": "m1",
            })

    assert bus.inbound == []
    assert len(ingress.messages) == 1
    inbound = ingress.messages[0].message
    assert "先前回答" in inbound.content
    assert "继续说明" in inbound.content
    assert inbound.metadata == {
        "client_request_id": "reply-1",
        "display_content": "继续说明",
        "reply_to_message_id": "m1",
        "reply_role": "assistant",
        "reply_preview": "先前回答",
    }
    await adapter.stop()


@pytest.mark.asyncio
async def test_web_chat_message_send_rejects_invalid_reply_target(tmp_path: Path) -> None:
    bus = _Bus()
    session_manager = _SessionManager()
    session_manager._store.messages["other"] = {
        "id": "other",
        "session_key": "akashic:other",
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
                "session_id": "akashic:abc",
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


def test_chat_navigation_uses_same_origin_dashboard_path(
    tmp_path: Path,
) -> None:
    app = create_chat_app(workspace=tmp_path, channel=WebChatChannel())

    with TestClient(app) as client:
        response = client.get("/api/chat/navigation")

    assert response.json() == {"dashboard_path": "/"}


def test_chat_runtime_routes_share_read_only_inspection_projection(
    tmp_path: Path,
) -> None:
    class RuntimeInspection:
        def list_documents(self) -> dict[str, object]:
            return {"items": [{"id": "veda", "title": "VEDA 人格"}]}

        def get_document(self, document_id: str) -> dict[str, object]:
            return {"id": document_id, "markdown": "# VEDA"}

        def list_jobs(self) -> dict[str, object]:
            return {"items": [{"id": "morning", "name": "晨间提醒"}]}

        def get_job(self, job_id: str) -> dict[str, object]:
            return {"id": job_id, "markdown": "# 晨间提醒"}

        async def list_capabilities(self) -> dict[str, object]:
            return {
                "snapshot_id": "snapshot-1",
                "plugins": [],
                "skills": [],
                "mcp_servers": [
                    {"owner_id": "workspace", "name": "github", "tool_count": 14}
                ],
            }

        async def get_mcp(
            self,
            owner_id: str,
            server_name: str,
        ) -> dict[str, object]:
            return {
                "owner_id": owner_id,
                "name": server_name,
                "markdown": "# github",
            }

    app = create_chat_app(
        workspace=tmp_path,
        channel=WebChatChannel(),
        runtime_inspection=cast(Any, RuntimeInspection()),
    )

    with TestClient(app) as client:
        documents = client.get("/api/chat/runtime/documents")
        document = client.get("/api/chat/runtime/documents/veda")
        jobs = client.get("/api/chat/runtime/jobs")
        job = client.get("/api/chat/runtime/jobs/morning")
        capabilities = client.get("/api/chat/runtime/capabilities")
        mcp = client.get(
            "/api/chat/runtime/mcp",
            params={"owner_id": "workspace", "name": "github"},
        )

    assert documents.json()["items"][0]["id"] == "veda"
    assert document.json()["markdown"] == "# VEDA"
    assert jobs.json()["items"][0]["id"] == "morning"
    assert job.json()["markdown"] == "# 晨间提醒"
    assert capabilities.json()["snapshot_id"] == "snapshot-1"
    assert mcp.json()["owner_id"] == "workspace"


@pytest.mark.asyncio
async def test_web_chat_rejects_malformed_fields_without_closing_connection(tmp_path: Path) -> None:
    bus = _Bus()
    ingress = _Inbound()
    session_manager = _SessionManager()
    channel = WebChatChannel()
    adapter = await _open_inbound_adapter(channel, ingress)
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

    assert bus.inbound == []
    assert len(ingress.messages) == 1
    assert ingress.messages[0].message.content == "继续"
    await adapter.stop()


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


def test_chat_messages_default_to_latest_turn_order(tmp_path: Path) -> None:
    channel = WebChatChannel()
    session_manager = _SessionManager()
    channel._ctx = cast(Any, SimpleNamespace(session_manager=session_manager))
    app = create_chat_app(workspace=tmp_path, channel=channel)

    with TestClient(app) as client:
        response = client.get("/api/chat/sessions/akashic:abc/messages")

    payload = response.json()
    assert [item["role"] for item in payload["items"]] == ["user", "assistant"]
    assert payload["has_more"] is True
    assert payload["before_seq"] == 8
    assert session_manager._store.calls[0] == {
        "session_key": "akashic:abc",
        "page_size": 50,
        "before_seq": None,
    }


def test_chat_messages_project_durable_artifacts_without_paths(tmp_path: Path) -> None:
    ref = AttachmentRef(
        artifact_id="artifact-history",
        kind=V3AttachmentKind.IMAGE,
        filename="001.png",
        media_type="image/png",
        size_bytes=3,
        sha256="a" * 64,
    )

    class _ArtifactStore:
        def resolve_refs(
            self,
            artifact_ids: tuple[str, ...],
        ) -> tuple[AttachmentRef, ...]:
            assert artifact_ids == (ref.artifact_id,)
            return (ref,)

    channel = WebChatChannel()
    session_manager = _SessionManager()
    session_manager._store.list_chat_history_page = lambda **_kwargs: (
        [
            {"id": "m0", "seq": 0, "role": "user", "content": "问题"},
            {
                "id": "m1",
                "seq": 1,
                "role": "assistant",
                "content": "答复",
                "attachment_ids": [ref.artifact_id],
            },
        ],
        2,
        False,
    )
    channel._ctx = cast(Any, SimpleNamespace(session_manager=session_manager))
    channel._artifact_store = cast(Any, _ArtifactStore())
    app = create_chat_app(workspace=tmp_path, channel=channel)

    with TestClient(app) as client:
        payload = client.get("/api/chat/sessions/akashic:abc/messages").json()

    assistant = payload["items"][1]
    assert assistant["attachment_ids"] == [ref.artifact_id]
    assert assistant["attachments"] == [{
        "artifact_id": ref.artifact_id,
        "kind": "image",
        "filename": "001.png",
        "media_type": "image/png",
        "size_bytes": 3,
        "sha256": "a" * 64,
        "url": f"/api/chat/artifacts/{ref.artifact_id}",
    }]
    assert "/sandbox/" not in str(assistant)


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

    assert "akashic:abc" not in session_manager.sessions
    assert session_manager.appended == []


@pytest.mark.asyncio
async def test_web_final_preserves_full_outbound_projection(tmp_path: Path) -> None:
    channel = WebChatChannel()
    socket = _WebSocket()
    channel._connections["akashic:abc"] = {cast(Any, socket)}
    image = tmp_path / "result.png"
    image.write_bytes(b"image")

    receipt = await channel._deliver_message(
        ChannelMessage(
            channel="akashic",
            chat_id="abc",
            content="answer",
            thinking="reasoning",
            attachments=(ChannelAttachment(AttachmentKind.IMAGE, str(image)),),
            metadata={
                "render": "card",
                "turn_duration_ms": 17,
                "_channel_commit_role": "passive",
            },
            control_turn_id="turn-1",
            execution_attempt_id="attempt-1",
        )
    )
    assert receipt.succeeded

    assert socket.frames == [
        {
            "type": "message.final",
            "session_id": "akashic:abc",
            "turn_id": "attempt-1",
            "content": "answer",
            "thinking": "reasoning",
            "media": [str(image)],
            "duration_ms": 17,
            "metadata": {"render": "card", "turn_duration_ms": 17},
            "control_turn_id": "turn-1",
            "execution_attempt_id": "attempt-1",
        }
    ]
    assert channel.has_media(image)


@pytest.mark.asyncio
async def test_web_v3_native_delivery_projects_opaque_artifacts_and_semantics() -> None:
    channel = WebChatChannel()
    socket = _WebSocket()
    channel._connections["akashic:abc"] = {cast(Any, socket)}
    read = _AttachmentRead()
    ref = AttachmentRef(
        artifact_id="artifact-1",
        kind=V3AttachmentKind.IMAGE,
        filename="meme.png",
        media_type="image/png",
        size_bytes=7,
        sha256="a" * 64,
    )
    adapter = channel.build_v3_adapter(_v3_context(read))
    ready = await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token=ready.binding_token,
            delivery_id="delivery-1",
            recipient="abc",
            body="answer",
            attachments=(ref,),
            metadata=cast(Any, {
                "turn_duration_ms": 17,
                "render": {"kind": "card", "citations": ["mem_1"]},
            }),
            thinking="reasoning",
            reply_to="user-1",
            session_message_id="assistant-1",
            control_turn_id="turn-1",
            execution_attempt_id="attempt-1",
            commit_role=ChannelCommitRole.PASSIVE,
        )
    )

    assert receipt.status is V3DeliveryStatus.DELIVERED
    assert read.leases[0].closed is True
    assert socket.frames == [{
        "type": "message.final",
        "session_id": "akashic:abc",
        "turn_id": "attempt-1",
        "content": "answer",
        "thinking": "reasoning",
        "media": [{
            "artifact_id": "artifact-1",
            "kind": "image",
            "filename": "meme.png",
            "media_type": "image/png",
            "size_bytes": 7,
            "sha256": "a" * 64,
            "url": "/api/chat/artifacts/artifact-1",
        }],
        "metadata": {
            "turn_duration_ms": 17,
            "render": {"kind": "card", "citations": ["mem_1"]},
        },
        "reply_to": "user-1",
        "session_message_id": "assistant-1",
        "control_turn_id": "turn-1",
        "execution_attempt_id": "attempt-1",
        "duration_ms": 17,
    }]


@pytest.mark.asyncio
async def test_web_passive_message_push_uses_independent_projection_identity() -> None:
    channel = WebChatChannel()
    socket = _WebSocket()
    channel._connections["akashic:abc"] = {cast(Any, socket)}
    adapter = channel.build_v3_adapter(_v3_context())
    ready = await adapter.start()

    receipt = await adapter.deliver(ProviderDeliveryRequest(
        binding_token=ready.binding_token,
        delivery_id="push-1",
        recipient="abc",
        body="独立推送",
        metadata={"source": "message_push"},
        commit_role=ChannelCommitRole.PASSIVE,
    ))

    assert receipt.status is V3DeliveryStatus.DELIVERED
    assert socket.frames == [{
        "type": "message.final",
        "session_id": "akashic:abc",
        "turn_id": "delivery:push-1",
        "content": "独立推送",
        "thinking": "",
        "media": [],
        "metadata": {"source": "message_push"},
    }]


@pytest.mark.asyncio
async def test_web_v3_native_delivery_rejects_without_socket() -> None:
    channel = WebChatChannel()
    adapter = channel.build_v3_adapter(_v3_context())
    await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-1",
            recipient="missing",
            body="answer",
        )
    )

    assert receipt.status is V3DeliveryStatus.REJECTED
    assert receipt.error == "Web 会话没有可用连接"


@pytest.mark.asyncio
async def test_web_v3_terminal_without_socket_refills_after_session_attach() -> None:
    channel = WebChatChannel()
    adapter = channel.build_v3_adapter(_v3_context())
    await adapter.start()

    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-1",
            recipient="abc",
            body="answer",
            metadata=cast(Any, {
                "turn_duration_ms": None,
                "nested": {"ids": ["mem_1"]},
            }),
            control_turn_id="turn-1",
            execution_attempt_id="attempt-1",
            commit_role=ChannelCommitRole.PASSIVE,
        )
    )
    socket = _WebSocket()
    await channel._attach_session(
        cast(Any, socket),
        "attach-1",
        {"session_id": "akashic:abc"},
    )

    assert receipt.status is V3DeliveryStatus.DELIVERED
    assert socket.frames == [{
        "type": "message.final",
        "session_id": "akashic:abc",
        "turn_id": "attempt-1",
        "content": "answer",
        "thinking": "",
        "media": [],
        "metadata": {
            "turn_duration_ms": None,
            "nested": {"ids": ["mem_1"]},
        },
        "control_turn_id": "turn-1",
        "execution_attempt_id": "attempt-1",
    }]
    assert "duration_ms" not in socket.frames[0]
    assert "akashic:abc" not in channel._pending_terminal


@pytest.mark.asyncio
async def test_web_turn_lifecycle_projects_server_owned_turn_id() -> None:
    channel = WebChatChannel()
    socket = _WebSocket()
    channel._connections["akashic:abc"] = {cast(Any, socket)}

    await channel._on_turn_started(TurnStarted(
        session_key="akashic:abc",
        channel="akashic",
        chat_id="abc",
        content="question",
        timestamp=datetime.now(UTC),
        turn_id="attempt-1",
        control_turn_id="turn:server-owner",
        client_message_id="client-1",
    ))
    await channel._on_stream_delta(StreamDeltaReady(
        session_key="akashic:abc",
        channel="akashic",
        chat_id="abc",
        turn_id="attempt-1",
        content_delta="answer",
    ))
    await channel._on_output_completed(TurnOutputCompleted(
        session_key="akashic:abc",
        channel="akashic",
        chat_id="abc",
        turn_id="attempt-1",
        client_message_id="client-1",
    ))

    assert [frame["type"] for frame in socket.frames] == [
        "turn.started",
        "answer.delta",
        "turn.output.completed",
    ]
    assert socket.frames[0]["client_message_id"] == "client-1"
    assert {frame["turn_id"] for frame in socket.frames} == {"attempt-1"}
    assert socket.frames[0]["control_turn_id"] == "turn:server-owner"


@pytest.mark.asyncio
async def test_web_turn_started_rejects_missing_server_turn_id() -> None:
    channel = WebChatChannel()

    with pytest.raises(RuntimeError, match="缺少 Server 权威 turn_id"):
        await channel._on_turn_started(TurnStarted(
            session_key="akashic:abc",
            channel="akashic",
            chat_id="abc",
            content="question",
            timestamp=datetime.now(UTC),
        ))


@pytest.mark.asyncio
async def test_web_v3_native_delivery_marks_socket_failure_unknown() -> None:
    channel = WebChatChannel()
    channel._connections["akashic:abc"] = {cast(Any, _FailingWebSocket())}
    adapter = channel.build_v3_adapter(_v3_context())
    await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-1",
            recipient="abc",
            body="answer",
        )
    )

    assert receipt.status is V3DeliveryStatus.UNKNOWN


@pytest.mark.asyncio
async def test_web_v3_native_delivery_marks_partial_broadcast_unknown() -> None:
    channel = WebChatChannel()
    delivered_socket = _WebSocket()
    channel._connections["akashic:abc"] = {
        cast(Any, delivered_socket),
        cast(Any, _FailingWebSocket()),
    }
    adapter = channel.build_v3_adapter(_v3_context())
    await adapter.start()
    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token="binding-1",
            delivery_id="delivery-1",
            recipient="abc",
            body="answer",
        )
    )

    assert receipt.status is V3DeliveryStatus.UNKNOWN
    assert len(delivered_socket.frames) == 1


@pytest.mark.asyncio
async def test_web_artifact_api_returns_opaque_upload_and_bounded_readback(
    tmp_path: Path,
) -> None:
    from PIL import Image

    image_path = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image_path)
    image_bytes = image_path.read_bytes()
    session_manager = SessionManager(tmp_path)
    bus = _Bus()
    ingress = _Inbound()
    channel = WebChatChannel()
    adapter = await _open_inbound_adapter(channel, ingress)
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
        upload = client.post(
            "/api/chat/uploads",
            params={"filename": "meme"},
            content=image_bytes,
        )
        payload = upload.json()
        assert upload.status_code == 200
        assert payload["filename"] == "meme"
        assert payload["kind"] == "image"
        assert payload["media_type"] == "image/png"
        assert payload["artifact_id"]
        assert "upload_path" not in payload
        assert all(str(tmp_path) not in str(value) for value in payload.values())

        readback = client.get(payload["upload_url"])
        forged = client.post(
            "/api/chat/uploads",
            params={"filename": "forged.png"},
            content=b"not an image",
        ).json()
        assert forged["kind"] == "file"
        traversal = client.get("/api/chat/artifacts/../sessions.db")
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "session.create", "request_id": "create"})
            session_id = ws.receive_json()["session_id"]
            ws.send_json({
                "type": "message.send",
                "request_id": "send",
                "session_id": session_id,
                "text": "请查看附件",
                "media": [payload["artifact_id"]],
            })

    assert readback.status_code == 200
    assert readback.content == image_bytes
    assert traversal.status_code in {404, 405}
    assert bus.inbound == []
    assert len(ingress.messages) == 1
    inbound = ingress.messages[0].message
    assert inbound.attachments[0].artifact_id == payload["artifact_id"]
    assert inbound.metadata == {"client_request_id": "send"}
    await adapter.stop()


@pytest.mark.asyncio
async def test_web_v3_adapter_stop_closes_binding_without_stopping_provider() -> None:
    channel = WebChatChannel()
    adapter = channel.build_v3_adapter(_v3_context())
    ready = await adapter.start()
    receipt = await adapter.stop()

    assert receipt.binding_token == ready.binding_token
    assert receipt.resources_closed is True
    with pytest.raises(RuntimeError, match="尚未 start"):
        await adapter.deliver(
            ProviderDeliveryRequest(
                binding_token=ready.binding_token,
                delivery_id="delivery-after-stop",
                recipient="abc",
                body="answer",
            )
        )


@pytest.mark.asyncio
async def test_web_v3_closed_admission_rejects_message_without_legacy_bus_call(
    tmp_path: Path,
) -> None:
    bus = _Bus()
    ingress = _Inbound()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=bus,
        session_manager=_SessionManager(),
        event_bus=_EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    socket = _WebSocket()
    channel._connections["akashic:abc"] = {cast(Any, socket)}
    adapter = await _open_inbound_adapter(channel, ingress)
    adapter.close_admission()

    await channel._send_user_message(
        cast(Any, socket),
        "closed-1",
        {"session_id": "akashic:abc", "text": "拒绝", "media": []},
    )

    assert bus.inbound == []
    assert ingress.messages == []
    assert socket.frames[-1] == {
        "type": "error",
        "request_id": "closed-1",
        "message": "Web v3 ingress admission 已关闭",
    }
    await adapter.stop()


@pytest.mark.asyncio
async def test_web_v3_adapter_stop_drains_old_callback_before_unregistering(
    tmp_path: Path,
) -> None:
    ingress = _Inbound(block=True)
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=_Bus(),
        session_manager=_SessionManager(),
        event_bus=_EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    old = await _open_inbound_adapter(channel, ingress, binding_token="old-binding")
    socket = _WebSocket()
    channel._connections["akashic:abc"] = {cast(Any, socket)}

    send_task = asyncio.create_task(channel._send_user_message(
        cast(Any, socket),
        "old-message",
        {"session_id": "akashic:abc", "text": "旧 binding", "media": []},
    ))
    await ingress.started.wait()
    stop_task = asyncio.create_task(old.stop())
    await asyncio.sleep(0)
    assert not stop_task.done()
    ingress.release.set()
    await asyncio.gather(send_task, stop_task)
    assert old.binding_token not in channel._v3_adapters


@pytest.mark.asyncio
async def test_web_v3_old_inflight_callback_cannot_enter_new_binding(
    tmp_path: Path,
) -> None:
    old_ingress = _Inbound(block=True)
    new_ingress = _Inbound()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=_Bus(),
        session_manager=_SessionManager(),
        event_bus=_EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    old = await _open_inbound_adapter(
        channel,
        old_ingress,
        binding_token="old-binding",
    )
    socket = _WebSocket()
    channel._connections["akashic:abc"] = {cast(Any, socket)}
    add_started = asyncio.Event()
    add_release = asyncio.Event()
    original_add_connection = channel._add_connection

    async def block_add_connection(session_key: str, connection: WebSocket) -> None:
        add_started.set()
        await add_release.wait()
        await original_add_connection(session_key, connection)

    channel._add_connection = block_add_connection  # type: ignore[method-assign]
    send_task = asyncio.create_task(channel._send_user_message(
        cast(Any, socket),
        "old-message",
        {"session_id": "akashic:abc", "text": "旧消息", "media": []},
    ))
    await add_started.wait()

    old.close_admission()
    stop_task = asyncio.create_task(old.stop())
    await asyncio.sleep(0)
    new = await _open_inbound_adapter(
        channel,
        new_ingress,
        binding_token="new-binding",
    )
    add_release.set()
    await old_ingress.started.wait()
    assert new_ingress.messages == []
    old_ingress.release.set()
    await asyncio.gather(send_task, stop_task)
    assert [item.message.content for item in old_ingress.messages] == ["旧消息"]
    assert new_ingress.messages == []
    await new.stop()


@pytest.mark.asyncio
async def test_web_bus_closed_rolls_back_identity_session_and_connection(
    tmp_path: Path,
) -> None:
    session_manager = SessionManager(tmp_path / "workspace")
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        session_manager=session_manager,
        installed_cache_root=tmp_path / "cache",
    )
    bus = MessageBus()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=bus,
        session_manager=session_manager,
        event_bus=EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    manager.channel_generation_host.bind_inbound_publisher(bus.publish_channel_inbound)
    await manager.bind_core_channel_definitions(
        (build_core_channel_definition(channel),)
    )
    adapter = tuple(channel._v3_adapters.values())[0]
    existing = session_manager.get_or_create("akashic:existing")
    existing.metadata["marker"] = "before"
    session_manager.save(existing)
    existing_before = session_manager.control_store.get_session_meta("akashic:existing")
    existing_socket = _WebSocket()
    assert await channel._add_connection(
        "akashic:existing",
        cast(Any, existing_socket),
    ) is True
    await bus.aclose()
    try:
        socket = _WebSocket()
        with pytest.raises(RuntimeError, match="message bus 已关闭"):
            await channel._send_user_message(
                cast(Any, socket),
                "closed-bus",
                {"session_id": "akashic:abc", "text": "hello", "media": []},
            )

        assert session_manager.get_channel_identities("akashic") == {}
        assert session_manager.control_store.get_session_meta("akashic:abc") is None
        assert session_manager.channel_identity_migration_completed("akashic") is True
        assert "akashic:abc" not in session_manager._cache
        assert channel._connections.get("akashic:abc") is None
        assert adapter._in_flight == 0

        with pytest.raises(RuntimeError, match="message bus 已关闭"):
            await channel._send_user_message(
                cast(Any, existing_socket),
                "closed-bus-existing",
                {"session_id": "akashic:existing", "text": "hello", "media": []},
            )
        assert (
            session_manager.control_store.get_session_meta("akashic:existing")
            == existing_before
        )
        assert channel._connections["akashic:existing"] == {existing_socket}
        assert adapter._in_flight == 0
    finally:
        await manager.terminate_all()
        session_manager.close()


@pytest.mark.asyncio
async def test_web_cancelled_ingress_rolls_back_session_and_connection(
    tmp_path: Path,
) -> None:
    session_manager = SessionManager(tmp_path / "workspace")
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        session_manager=session_manager,
        installed_cache_root=tmp_path / "cache",
    )
    publish_started = asyncio.Event()
    publish_release = asyncio.Event()

    async def publish(_envelope: Any) -> None:
        publish_started.set()
        await publish_release.wait()

    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=_Bus(),
        session_manager=session_manager,
        event_bus=EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    manager.channel_generation_host.bind_inbound_publisher(publish)
    await manager.bind_core_channel_definitions(
        (build_core_channel_definition(channel),)
    )
    adapter = tuple(channel._v3_adapters.values())[0]
    try:
        socket = _WebSocket()
        send_task = asyncio.create_task(channel._send_user_message(
            cast(Any, socket),
            "cancelled-ingress",
            {"session_id": "akashic:cancel", "text": "hello", "media": []},
        ))
        await publish_started.wait()
        send_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send_task

        assert session_manager.get_channel_identities("akashic") == {}
        assert session_manager.control_store.get_session_meta("akashic:cancel") is None
        assert "akashic:cancel" not in session_manager._cache
        assert channel._connections.get("akashic:cancel") is None
        assert adapter._in_flight == 0
    finally:
        publish_release.set()
        await manager.terminate_all()
        session_manager.close()


@pytest.mark.asyncio
async def test_web_v3_ingress_persists_unprefixed_identity_for_exact_session(
    tmp_path: Path,
) -> None:
    """A Web identity and its envelope must name the same durable session."""

    session_manager = SessionManager(tmp_path / "workspace")
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        session_manager=session_manager,
        installed_cache_root=tmp_path / "cache",
    )
    bus = MessageBus()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=bus,
        session_manager=session_manager,
        event_bus=EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    manager.channel_generation_host.bind_inbound_publisher(bus.publish_channel_inbound)
    await manager.bind_core_channel_definitions(
        (build_core_channel_definition(channel),)
    )
    try:
        socket = _WebSocket()
        session_key = await channel._create_session(cast(Any, socket), "create-1")
        await channel._send_user_message(
            cast(Any, socket),
            "request-1",
            {"session_id": session_key, "text": "hello", "media": []},
        )
        envelope = await bus.consume_inbound()
        assert isinstance(envelope, InboundEnvelope)
        chat_id = session_key.removeprefix("akashic:")
        assert envelope.session_key == session_key
        assert envelope.message.chat_id == chat_id
        assert session_manager.get_channel_identities("akashic") == {chat_id: chat_id}
        _, admission_id = session_manager.admit_existing(session_key)
        session_manager.release_admission(admission_id)
        await bus.release_channel_inbound(envelope, InboundOwner.LANE)
    finally:
        await manager.terminate_all()
        await bus.aclose()
        session_manager.close()


@pytest.mark.asyncio
async def test_web_ingress_survives_unrelated_plugin_snapshot_promotion(
    tmp_path: Path,
) -> None:
    """普通插件晋升后，Web 入站继续通过新 stable snapshot。"""

    # 1. 先发布普通插件，再接入 Core Web Channel
    plugin_dir = tmp_path / "plugins" / "plain_probe"
    plugin_dir.mkdir(parents=True)
    plugin_source = (
        "api_version = 3\n"
        "name = 'plain_probe'\n"
        "version = {version!r}\n"
        "async def apply(ctx, config): pass\n"
    )
    (plugin_dir / "plugin.py").write_text(
        plugin_source.format(version="1.0.0"),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    session_manager = SessionManager(workspace)
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=workspace,
        session_manager=session_manager,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    bus = MessageBus()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=bus,
        session_manager=session_manager,
        event_bus=EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    manager.channel_generation_host.bind_inbound_publisher(
        bus.publish_channel_inbound
    )
    await manager.bind_core_channel_definitions(
        (build_core_channel_definition(channel),)
    )
    previous_snapshot = manager.current_snapshot
    assert previous_snapshot is not None
    previous_runtime = manager.channel_generation_host.get(previous_snapshot.snapshot_id)
    assert previous_runtime is not None

    try:
        # 2. 只晋升普通插件，Core Web Channel 声明保持完全相同
        (plugin_dir / "plugin.py").write_text(
            plugin_source.format(version="2.0.0"),
            encoding="utf-8",
        )
        assert await manager.prepare_candidate("plain_probe") is not None
        await manager.publish_prepared("plain_probe")
        current_snapshot = manager.current_snapshot
        assert current_snapshot is not None
        current_runtime = manager.channel_generation_host.get(current_snapshot.snapshot_id)
        assert current_runtime is not None
        assert current_runtime is not previous_runtime
        assert current_runtime.snapshot_id == current_snapshot.snapshot_id

        # 3. 真实 Web 发送必须进入 MessageBus，不得因旧 snapshot 断开
        socket = _WebSocket()
        await channel._send_user_message(
            cast(Any, socket),
            "request-after-promotion",
            {"session_id": "akashic:after-promotion", "text": "hello", "media": []},
        )
        envelope = await bus.consume_inbound()
        assert isinstance(envelope, InboundEnvelope)
        assert envelope.snapshot_id == current_runtime.snapshot_id
        assert envelope.message.content == "hello"
        await bus.release_channel_inbound(envelope, InboundOwner.LANE)
    finally:
        await manager.terminate_all()
        await bus.aclose()
        session_manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_id", "session_id", "expected_request_id", "expected_error"),
    [
        ("x" * 257, "akashic:abc", "", "request_id 格式无效"),
        ("bad\x01id", "akashic:abc", "", "request_id 格式无效"),
        ("safe", f"akashic:{'x' * 253}", "safe", "session_id 格式无效"),
        ("safe", "akashic:bad\x01id", "safe", "session_id 格式无效"),
    ],
)
async def test_web_rejects_invalid_external_ids_before_ingress_or_session_write(
    tmp_path: Path,
    request_id: str,
    session_id: str,
    expected_request_id: str,
    expected_error: str,
) -> None:
    session_manager = SessionManager(tmp_path / "workspace")
    ingress = _Inbound()
    channel = WebChatChannel()
    await channel.start(cast(Any, SimpleNamespace(
        bus=_Bus(),
        session_manager=session_manager,
        event_bus=_EventBus(),
        push_tool=_PushTool(),
        attachment_store=AttachmentStore(tmp_path / "uploads"),
        interrupt_controller=None,
    )))
    adapter = await _open_inbound_adapter(channel, ingress)
    socket = _WebSocket()
    try:
        await channel._handle_client_frame(cast(Any, socket), {
            "type": "message.send",
            "request_id": request_id,
            "session_id": session_id,
            "text": "hello",
            "media": [],
        })
        assert socket.frames == [{
            "type": "error",
            "request_id": expected_request_id,
            "message": expected_error,
        }]
        assert ingress.messages == []
        assert session_manager.get_channel_identities("akashic") == {}
        with pytest.raises(KeyError, match="session 不存在"):
            session_manager.admit_existing("akashic:abc")
    finally:
        await adapter.stop()
        session_manager.close()

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
import secrets
import threading
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from agent.config_models import MobileKeyEncryptionConfig, MobileRealtimeConfig
from agent.control.scoped_turn import TurnAcceptedReceipt
from agent.control.runtime import ConversationRuntime
from agent.plugin_composition.channels import (
    ChannelCommitRole,
    ChannelDeliveryReceipt,
    ChannelFactoryContext,
    ChannelInboundMessage,
    ChannelRuntimePorts,
    JsonValue,
    ProviderDeliveryRequest,
    RawInbound,
)
from agent.plugin_composition.durable_deliveries import (
    DurableBindingAttempt,
    DurableDeliveryRequest,
    PluginDurableDeliveries,
)
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from bootstrap.chat_api import create_chat_app
from bootstrap.core_channel_adapter import build_core_channel_definition
from bootstrap.passive_worker import PassiveMessageWorker
from bootstrap.tools import _dispatch_v3_durable_delivery
from bus.event_bus import EventBus
from bus.events import OutboundMessage, channel_message_from_outbound
from bus.events_lifecycle import StreamDeltaReady, TurnOutputCompleted, TurnStarted
from bus.queue import MessageBus
from infra.channels.base import AttachmentStore
from infra.channels.akashic_channel import AkashicChannel
from infra.channels.web_chat_channel import WebChatChannel
from infra.mobile_realtime.attachments import decode_attachment_chunk
from infra.mobile_realtime.auth import device_proof_signing_bytes
from infra.mobile_realtime.gateway import (
    MobileGatewayRuntime,
    build_mobile_gateway_runtime,
    create_mobile_gateway_app,
)
from infra.mobile_realtime.key_protection import KeyProtectionError
from infra.mobile_realtime.storage import DeviceRecord
from agent.plugins.manager import PluginManager
from session.manager import SessionManager


async def _attach_open_mobile_v3(
    channel: Any,
    ingress: Any,
    *,
    binding_token: str,
) -> Any:
    """Attach one exact v3 ingress and open admission for this fixture."""

    context = ChannelFactoryContext(
        snapshot_id="isolated-e2e-snapshot",
        generation_id="isolated-e2e-generation",
        binding_token=binding_token,
        config={},
        credentials={},
        provider_client_factory=cast(Any, object()),
        ingress=ingress,
        identity=None,
    )
    adapter = channel.build_v3_adapter(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    assert (await adapter.start()).binding_token == binding_token
    adapter.open_admission()
    return adapter


class _EphemeralMasterKeys:
    def __init__(self) -> None:
        self.keys: dict[str, bytes] = {}

    def create(self) -> tuple[str, bytes]:
        key_id = uuid4().hex
        key = secrets.token_bytes(32)
        self.keys[key_id] = key
        return key_id, key

    def load(self, master_key_id: str) -> bytes:
        try:
            return self.keys[master_key_id]
        except KeyError as error:
            raise KeyProtectionError("隔离测试 master key 不存在") from error


class _EventBus:
    def on(self, event_type: type[object], callback: object) -> None:
        return None


class _PushTool:
    pass


class _SharedSessionBus:
    """Persist public Web/Mobile ingress into one real SessionManager."""

    def __init__(self, manager: SessionManager, event_bus: EventBus) -> None:
        self._manager = manager
        self._event_bus = event_bus
        self._adapter: Any | None = None
        self._recovery = None
        self._count = 0
        self._changed = threading.Condition()

    def bind_adapter(self, adapter: Any) -> None:
        self._adapter = adapter

    def bind_mobile_channel_inbound_recoverer(self, callback: object) -> None:
        self._recovery = callback

    async def reserve_mobile_channel_handoff(self, raw: RawInbound) -> bool:
        assert raw.message.metadata.get("mobile_v3_handoff") is True
        return True

    async def defer_mobile_channel_handoff(self, handoff_id: str) -> None:
        raise AssertionError(f"unexpected deferred handoff: {handoff_id}")

    def has_pending_mobile_handoff(
        self,
        *,
        session_key: str,
        client_message_id: str,
    ) -> bool:
        return False

    async def admit(self, raw: RawInbound) -> bool:
        session_id = str(
            raw.message.metadata.get("session_key_override")
            or f"akashic:{raw.message.chat_id}"
        )
        session = self._manager.get_or_create(session_id)
        session.add_message(
            "user",
            raw.message.content,
            client_message_id=raw.message_id,
        )
        self._manager.save(session)
        turn_id = f"turn:shared:{self._count + 1}"
        await self._event_bus.fanout(
            TurnStarted(
                session_key=session_id,
                channel="akashic",
                chat_id=raw.message.chat_id,
                content=raw.message.content,
                timestamp=datetime.now(timezone.utc),
                turn_id=turn_id,
                control_turn_id=turn_id,
                client_message_id=raw.message_id,
            )
        )
        await self._event_bus.fanout(
            StreamDeltaReady(
                session_key=session_id,
                channel="akashic",
                chat_id=raw.message.chat_id,
                turn_id=turn_id,
                thinking_delta="共享思考",
                content_delta="共享回答",
            )
        )
        await self._event_bus.fanout(
            TurnOutputCompleted(
                session_key=session_id,
                channel="akashic",
                chat_id=raw.message.chat_id,
                turn_id=turn_id,
                client_message_id=raw.message_id,
            )
        )
        if self._adapter is None:
            raise RuntimeError("Shared Akashic fixture 尚未绑定 adapter")
        receipt = await self._adapter.deliver(
            ProviderDeliveryRequest(
                binding_token="shared-e2e-binding",
                delivery_id=f"reply:{raw.message_id}",
                recipient=raw.message.chat_id,
                body="共享回答",
                thinking="共享思考",
                metadata={"client_message_id": raw.message_id},
                commit_role=ChannelCommitRole.PASSIVE,
                control_turn_id=turn_id,
            )
        )
        assert receipt.status.value == "delivered"
        with self._changed:
            self._count += 1
            self._changed.notify_all()
        return True

    def wait_for_count(self, expected: int) -> None:
        with self._changed:
            assert self._changed.wait_for(
                lambda: self._count >= expected,
                timeout=3,
            )


class _DeterministicAgentBus:
    """把手机入站消息持久化，并返回一条带固定媒体的确定性回复。"""

    def __init__(self, manager: SessionManager, reply_media: Path) -> None:
        self._manager = manager
        self._reply_media = reply_media
        self._runtime: MobileGatewayRuntime | None = None
        self.inbound_count = 0
        self.legacy_publish_calls = 0

    def bind(self, runtime: MobileGatewayRuntime) -> None:
        self._runtime = runtime

    async def publish_inbound(self, message: object) -> None:
        self.legacy_publish_calls += 1
        raise AssertionError("Mobile v3 fixture 不得调用 legacy publish_inbound")

    async def reserve_mobile_channel_handoff(self, raw: RawInbound) -> bool:
        assert isinstance(raw, RawInbound)
        assert raw.message.metadata["mobile_v3_handoff"] is True
        return True

    async def defer_mobile_channel_handoff(self, handoff_id: str) -> None:
        raise AssertionError(f"unexpected deferred Mobile v3 handoff: {handoff_id}")

    def has_pending_mobile_handoff(
        self,
        *,
        session_key: str,
        client_message_id: str,
    ) -> bool:
        return False

    async def admit(self, raw: RawInbound) -> bool:
        """按真实持久化顺序生成 turn.started 与 message.final。"""

        # 1. 持久化同一个 client_message_id，模拟生命周期入库结果
        assert isinstance(raw, RawInbound)
        inbound = raw.message
        assert isinstance(inbound, ChannelInboundMessage)
        runtime = self._require_runtime()
        session_id = cast(str, inbound.metadata["session_key_override"])
        session = self._manager.get_or_create(session_id)
        client_message_id = cast(str, inbound.metadata["client_message_id"])
        session.add_message(
            "user",
            inbound.content,
            client_message_id=client_message_id,
        )
        turn_id = uuid4().hex
        session.add_message(
            "assistant",
            "隔离网关固定回复",
            media=[str(self._reply_media)],
        )
        self._manager.save(session)
        assistant_message_id = str(session.messages[-1]["id"])
        self.inbound_count += 1

        # 2. 通过真实移动渠道发布可恢复事件
        await runtime.channel._on_turn_started(
            TurnStarted(
                session_key=session_id,
                channel="akashic",
                chat_id=inbound.chat_id,
                content=inbound.content,
                timestamp=datetime.now(timezone.utc),
                turn_id=turn_id,
            )
        )
        receipt = await runtime.channel._deliver_message(
            channel_message_from_outbound(
                OutboundMessage(
                    channel="akashic",
                    chat_id=inbound.chat_id,
                    content="隔离网关固定回复",
                    media=[str(self._reply_media)],
                    control_turn_id=turn_id,
                    session_message_id=assistant_message_id,
                    metadata={"_channel_commit_role": "passive"},
                )
            )
        )
        assert receipt.succeeded
        return True

    def _require_runtime(self) -> MobileGatewayRuntime:
        if self._runtime is None:
            raise RuntimeError("隔离 Agent bus 尚未绑定 gateway runtime")
        return self._runtime


def _config(root: Path) -> MobileRealtimeConfig:
    return MobileRealtimeConfig(
        enabled=True,
        database=root / "gateway" / "mobile.db",
        lan_hostname="isolated-mobile.test",
        public_url="",
        key_encryption=MobileKeyEncryptionConfig(
            keyset_manifest=root / "gateway" / "keys" / "current.json"
        ),
    )


def _public_key(private_key: ec.EllipticCurvePrivateKey) -> str:
    encoded = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return base64.b64encode(encoded).decode("ascii")


def _proof(
    challenge: dict[str, object],
    device_id: str,
    device_key: ec.EllipticCurvePrivateKey,
) -> dict[str, object]:
    client_nonce = base64.urlsafe_b64encode(secrets.token_bytes(18)).decode("ascii")
    signing_bytes = device_proof_signing_bytes(
        server_id=str(challenge["server_id"]),
        challenge_id=str(challenge["challenge_id"]),
        challenge_nonce=str(challenge["nonce"]),
        device_id=device_id,
        client_nonce=client_nonce,
    )
    signature = device_key.sign(signing_bytes, ec.ECDSA(hashes.SHA256()))
    return {
        "v": 1,
        "kind": "control",
        "type": "device.proof",
        "payload": {
            "challenge_id": challenge["challenge_id"],
            "device_id": device_id,
            "client_nonce": client_nonce,
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    }


def _authenticate(
    websocket: Any,
    device_id: str,
    device_key: ec.EllipticCurvePrivateKey,
) -> int:
    challenge = websocket.receive_json()
    assert challenge["type"] == "server.challenge"
    websocket.send_json(_proof(challenge["payload"], device_id, device_key))
    accepted = websocket.receive_json()
    assert accepted["type"] == "auth.accepted"
    return int(accepted["connection_epoch"])


def _resume(websocket: Any, epoch: int, last_ack: int) -> list[dict[str, Any]]:
    websocket.send_json(
        {
            "v": 1,
            "kind": "control",
            "type": "resume",
            "connection_epoch": epoch,
            "payload": {"last_ack": last_ack, "active_turns": []},
        }
    )
    frames: list[dict[str, Any]] = []
    while True:
        frame = websocket.receive_json()
        frames.append(frame)
        if frame["type"] == "sync.completed":
            return frames


def _command(
    command_id: str,
    command_type: str,
    epoch: int,
    *,
    session_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    frame: dict[str, object] = {
        "v": 1,
        "kind": "command",
        "type": command_type,
        "id": command_id,
        "connection_epoch": epoch,
        "payload": payload or {},
    }
    if session_id is not None:
        frame["session_id"] = session_id
    return frame


def _history_identity(item: dict[str, Any]) -> str:
    if item["role"] == "user" and item.get("client_message_id"):
        return f"user:{item['client_message_id']}"
    return f"{item['role']}:{item['id']}"


def test_web_and_mobile_share_one_session_and_receive_one_delivery(
    tmp_path: Path,
) -> None:
    """Exercise both public protocols against one Session projection fixture."""

    root = tmp_path / "shared-akashic-e2e"
    manager = SessionManager(root / "workspace")

    async def build_runtime() -> tuple[MobileGatewayRuntime, object]:
        return build_mobile_gateway_runtime(
            _config(root),
            root,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build_runtime())
    web = WebChatChannel()
    channel = AkashicChannel(web, runtime.channel)
    event_bus = EventBus()
    bus = _SharedSessionBus(manager, event_bus)
    context = cast(
        Any,
        SimpleNamespace(
            bus=bus,
            session_manager=manager,
            event_bus=event_bus,
            push_tool=_PushTool(),
            interrupt_controller=None,
            attachment_store=AttachmentStore(root / "attachments"),
        ),
    )
    asyncio.run(channel.start(context))
    adapter = asyncio.run(
        _attach_open_mobile_v3(
            channel,
            bus,
            binding_token="shared-e2e-binding",
        )
    )
    bus.bind_adapter(adapter)

    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_public_key(device_key),
            display_name="Shared Akashic Harness",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )
    app = create_chat_app(workspace=root, channel=web)
    app.mount("/mobile", create_mobile_gateway_app(runtime))

    try:
        with TestClient(app) as client:
            with (
                client.websocket_connect("/ws") as web_socket,
                client.websocket_connect("/mobile/ws") as mobile_socket,
            ):
                epoch = _authenticate(mobile_socket, device_id, device_key)
                assert [
                    frame["type"] for frame in _resume(mobile_socket, epoch, 0)
                ] == ["sync.completed"]

                # 1. Web allocates the identity and writes through the shared ingress.
                web_socket.send_json(
                    {"type": "session.create", "request_id": "web-create"}
                )
                created = web_socket.receive_json()
                session_id = cast(str, created["session_id"])
                assert re.fullmatch(r"akashic:[0-9a-f]{32}", session_id)
                web_socket.send_json(
                    {
                        "type": "message.send",
                        "request_id": "web-message",
                        "session_id": session_id,
                        "text": "Web 写入同一个会话",
                        "media": [],
                    }
                )
                bus.wait_for_count(1)
                web_live = [web_socket.receive_json() for _ in range(5)]
                mobile_live = [mobile_socket.receive_json() for _ in range(4)]
                assert [frame["type"] for frame in web_live] == [
                    "turn.started",
                    "react.thinking.delta",
                    "answer.delta",
                    "turn.output.completed",
                    "message.final",
                ]
                assert [frame["type"] for frame in mobile_live] == [
                    "turn.started",
                    "react.thinking.delta",
                    "answer.delta",
                    "message.final",
                ]
                assert web_live[0]["client_message_id"] == "web-message"

                # 2. Mobile lists and reads the exact same durable Session.
                mobile_socket.send_json(
                    _command("01J00000000000000000000020", "session.list", epoch)
                )
                listed = mobile_socket.receive_json()
                assert listed["type"] == "session.list"
                assert mobile_socket.receive_json()["type"] == "session.list.ok"
                assert [item["session_id"] for item in listed["payload"]["items"]] == [
                    session_id
                ]
                mobile_socket.send_json(
                    _command(
                        "01J00000000000000000000021",
                        "history.get",
                        epoch,
                        session_id=session_id,
                        payload={"page": 1, "page_size": 50},
                    )
                )
                first_page = mobile_socket.receive_json()
                assert mobile_socket.receive_json()["type"] == "history.get.ok"
                assert [item["content"] for item in first_page["payload"]["items"]] == [
                    "Web 写入同一个会话"
                ]

                # 3. Mobile writes back; Web HTTP history sees both messages once.
                mobile_socket.send_json(
                    _command(
                        "01J00000000000000000000022",
                        "message.send",
                        epoch,
                        session_id=session_id,
                        payload={
                            "client_message_id": "01J00000000000000000000022",
                            "session_id": session_id,
                            "text": "Mobile 写回同一个会话",
                            "media_refs": [],
                            "client_created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
                mobile_reply_frames = [mobile_socket.receive_json() for _ in range(5)]
                assert [frame["type"] for frame in mobile_reply_frames] == [
                    "turn.started",
                    "react.thinking.delta",
                    "answer.delta",
                    "message.final",
                    "message.send.ok",
                ]
                bus.wait_for_count(2)
                web_reply_frames = [web_socket.receive_json() for _ in range(5)]
                assert [frame["type"] for frame in web_reply_frames] == [
                    "turn.started",
                    "react.thinking.delta",
                    "answer.delta",
                    "turn.output.completed",
                    "message.final",
                ]
                assert web_reply_frames[0]["client_message_id"] == (
                    "01J00000000000000000000022"
                )
                assert web_reply_frames[0]["content"] == "Mobile 写回同一个会话"
                history = client.get(f"/api/chat/sessions/{session_id}/messages").json()
                assert [item["content"] for item in history["items"]] == [
                    "Web 写入同一个会话",
                    "Mobile 写回同一个会话",
                ]
                sessions = client.get("/api/chat/sessions").json()
                assert [item["key"] for item in sessions["items"]] == [session_id]
                assert not any(
                    item["key"].startswith(("web:", "mobile:"))
                    for item in sessions["items"]
                )

                # 4. One logical proactive/schedule result fans out to both UIs.
                delivery_id = "schedule-delivery-e2e"

                async def sender(request: DurableDeliveryRequest, started: Any) -> Any:
                    started(
                        DurableBindingAttempt(
                            delivery_id,
                            "isolated-e2e-snapshot",
                            "isolated-e2e-generation",
                            "shared-e2e-binding",
                        )
                    )
                    delivery_metadata = cast(
                        dict[str, JsonValue], dict(request.metadata)
                    )
                    delivery_metadata["delivery_id"] = request.logical_delivery_id
                    session_message_id = await manager.append_durable_delivery(
                        session_key=request.projection_session_id,
                        content=request.body,
                        delivery_id=request.logical_delivery_id,
                        control_turn_id=request.accepted_turn.turn_id,
                    )
                    provider = await adapter.deliver(
                        ProviderDeliveryRequest(
                            binding_token="shared-e2e-binding",
                            delivery_id=request.logical_delivery_id,
                            recipient=request.recipient,
                            body=request.body,
                            metadata=delivery_metadata,
                            commit_role=ChannelCommitRole.DIRECT,
                            control_turn_id=request.accepted_turn.turn_id,
                            session_message_id=session_message_id,
                        )
                    )
                    return ChannelDeliveryReceipt(
                        provider.delivery_id,
                        provider.status,
                        provider.provider_ids,
                        provider.error,
                    )

                async def project(request: DurableDeliveryRequest) -> str:
                    return await manager.append_durable_delivery(
                        session_key=request.projection_session_id,
                        content=request.body,
                        delivery_id=request.logical_delivery_id,
                        control_turn_id=request.accepted_turn.turn_id,
                    )

                durable = PluginDurableDeliveries(
                    DurableDeliveryStore(root / "runtime" / "settlements.sqlite"),
                    sender,
                    project,
                )
                request = DurableDeliveryRequest(
                    logical_delivery_id=delivery_id,
                    accepted_turn=TurnAcceptedReceipt(
                        "scheduler:morning",
                        "turn:schedule-e2e",
                    ),
                    target_service="scheduler.delivery.v1",
                    channel="akashic",
                    recipient=session_id.removeprefix("akashic:"),
                    projection_session_id=session_id,
                    body="定时任务完成",
                    metadata={"source": "schedule"},
                )
                receipt = client.portal.call(durable.submit, request)
                assert receipt.state == "projected"
                web_delivery = web_socket.receive_json()
                mobile_delivery = mobile_socket.receive_json()
                assert web_delivery["type"] == "message.final"
                assert mobile_delivery["type"] == "session.updated"
                assert web_delivery["session_id"] == session_id
                assert mobile_delivery["session_id"] == session_id
                assert web_delivery["content"] == "定时任务完成"
                assert mobile_delivery["payload"]["head_seq"] == 2
                assert web_delivery["metadata"]["delivery_id"] == delivery_id
                assert mobile_delivery["payload"]["message_id"] == f"{session_id}:2"
                projected = manager.control_store.fetch_session_messages(session_id)
                assert [item["content"] for item in projected] == [
                    "Web 写入同一个会话",
                    "Mobile 写回同一个会话",
                    "定时任务完成",
                ]
                assert projected[-1]["delivery_id"] == delivery_id

                duplicate = client.portal.call(durable.submit, request)
                assert duplicate.projection_message_id == receipt.projection_message_id
                assert (
                    len(manager.control_store.fetch_session_messages(session_id)) == 3
                )
    finally:
        asyncio.run(adapter.stop())
        asyncio.run(channel.stop())
        manager.close()
        runtime.close()


def test_production_channel_binding_persists_ingress_and_routes_durable_delivery(
    tmp_path: Path,
) -> None:
    """Run Web ingress and durable output through the committed Core binding."""

    root = tmp_path / "production-akashic-e2e"
    manager = SessionManager(root / "workspace")

    async def build_runtime() -> tuple[MobileGatewayRuntime, object]:
        return build_mobile_gateway_runtime(
            _config(root),
            root,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build_runtime())
    web = WebChatChannel()
    channel = AkashicChannel(web, runtime.channel)
    bus = MessageBus()
    event_bus = EventBus()
    plugin_manager = PluginManager(
        plugin_dirs=[root / "plugins"],
        event_bus=event_bus,
        workspace=root / "workspace",
        session_manager=manager,
        installed_cache_root=root / "cache",
    )

    async def execute(request: Any) -> str:
        return f"Core 回复：{request.input}"

    conversation = ConversationRuntime(manager.control_store, execute)
    worker = PassiveMessageWorker(
        bus,
        conversation,
        cast(Any, SimpleNamespace(session_manager=manager)),
    )
    context = cast(
        Any,
        SimpleNamespace(
            bus=bus,
            session_manager=manager,
            event_bus=event_bus,
            push_tool=_PushTool(),
            interrupt_controller=None,
            attachment_store=AttachmentStore(root / "attachments"),
        ),
    )
    tasks: tuple[asyncio.Task[None], asyncio.Task[None]] | None = None

    async def start() -> None:
        nonlocal tasks
        bus.bind_durable_inbound_store(manager.control_store)
        plugin_manager.channel_generation_host.bind_inbound_publisher(
            bus.publish_channel_inbound
        )
        plugin_manager.bind_durable_delivery_sender(
            lambda request, started: _dispatch_v3_durable_delivery(
                plugin_manager,
                bus,
                request,
                started,
                session_manager=manager,
            )
        )
        bus.bind_channel_outbound_dispatcher(
            plugin_manager.channel_generation_host.dispatch_outbound
        )
        await channel.start(context)
        await plugin_manager.bind_core_channel_definitions(
            (build_core_channel_definition(channel),)
        )
        tasks = (
            asyncio.create_task(worker.run()),
            asyncio.create_task(bus.dispatch_outbound()),
        )

    async def stop() -> None:
        worker.stop()
        bus.stop()
        if tasks is not None:
            await asyncio.gather(*tasks)
        await conversation.shutdown()
        await plugin_manager.terminate_all()
        await channel.stop()

    app = create_chat_app(workspace=root, channel=web)
    try:
        with TestClient(app) as client:
            client.portal.call(start)
            with client.websocket_connect("/ws") as socket:
                socket.send_json({"type": "session.create", "request_id": "create"})
                session_id = cast(str, socket.receive_json()["session_id"])
                socket.send_json(
                    {
                        "type": "message.send",
                        "request_id": "message",
                        "session_id": session_id,
                        "text": "穿过生产 Core",
                        "media": [],
                    }
                )
                terminal = socket.receive_json()
                assert terminal["type"] == "message.final"
                assert terminal["content"] == "Core 回复：穿过生产 Core"
                turns = manager.control_store.list_turns(session_id)
                assert len(turns) == 1
                assert turns[0].input == "穿过生产 Core"
                assert turns[0].final_response == "Core 回复：穿过生产 Core"
                assert turns[0].metadata["channel"] == "akashic"

                durable = plugin_manager._formal_durable_deliveries()
                request = DurableDeliveryRequest(
                    logical_delivery_id="wake:production-e2e",
                    accepted_turn=TurnAcceptedReceipt(
                        "wake:production",
                        "turn:wake-production",
                    ),
                    target_service="wake.delivery.v1",
                    channel="akashic",
                    recipient=session_id.removeprefix("akashic:"),
                    projection_session_id=session_id,
                    body="主动结果穿过生产 Core",
                )
                receipt = client.portal.call(durable.submit, request)
                pushed = socket.receive_json()
                assert receipt.state == "projected"
                assert pushed["type"] == "message.final"
                assert pushed["content"] == request.body
                assert [
                    item["content"]
                    for item in manager.control_store.fetch_session_messages(session_id)
                ] == ["主动结果穿过生产 Core"]
            client.portal.call(stop)
    finally:
        if tasks is not None and any(not task.done() for task in tasks):
            for task in tasks:
                task.cancel()
        with suppress(Exception):
            asyncio.run(plugin_manager.terminate_all())
        with suppress(Exception):
            asyncio.run(channel.stop())
        manager.close()
        runtime.close()


def test_isolated_gateway_recovers_lost_frames_and_keeps_history_idempotent(
    tmp_path: Path,
) -> None:
    """覆盖隔离存储、重复历史同步、断线补发与固定媒体下载。"""

    # 1. 只在 pytest 临时根目录创建 Gateway、会话库和附件目录
    root = tmp_path / "isolated-mobile-e2e"
    manager = SessionManager(root / "workspace")
    reply_media = root / "fixtures" / "gateway-reply.gif"
    reply_media.parent.mkdir(parents=True)
    reply_bytes = b"GIF89a" + bytes(range(256)) * 128
    reply_media.write_bytes(reply_bytes)

    async def build_runtime() -> tuple[MobileGatewayRuntime, object]:
        return build_mobile_gateway_runtime(
            _config(root),
            root,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build_runtime())
    bus = _DeterministicAgentBus(manager, reply_media)
    bus.bind(runtime)
    asyncio.run(
        runtime.channel.start(
            cast(
                Any,
                SimpleNamespace(
                    bus=bus,
                    session_manager=manager,
                    event_bus=_EventBus(),
                    push_tool=_PushTool(),
                    interrupt_controller=None,
                    attachment_store=AttachmentStore(root / "attachments"),
                ),
            )
        )
    )
    adapter = asyncio.run(
        _attach_open_mobile_v3(
            runtime.channel,
            bus,
            binding_token="isolated-e2e-fixture",
        )
    )

    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_public_key(device_key),
            display_name="Isolated Android Harness",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1", "attachments-v1"),
        )
    )
    client = TestClient(create_mobile_gateway_app(runtime))
    try:
        # 2. 连续拉取同一历史页两次，按 canonical identity 合并后不增长
        with client.websocket_connect("/ws") as websocket:
            epoch = _authenticate(websocket, device_id, device_key)
            initial = _resume(websocket, epoch, last_ack=0)
            assert [frame["type"] for frame in initial] == ["sync.completed"]
            websocket.send_json(
                {
                    "v": 1,
                    "kind": "ack",
                    "type": "event.ack",
                    "connection_epoch": epoch,
                    "payload": {"through_event_seq": initial[-1]["event_seq"]},
                }
            )
            last_ack = int(initial[-1]["event_seq"])

            # 3. Mobile 与 Web 使用同一个 Core 分配规则，不在测试里手写 Session ID。
            websocket.send_json(
                _command(
                    "01J00000000000000000000010",
                    "session.create",
                    epoch,
                )
            )
            created = websocket.receive_json()
            assert created["type"] == "session.created"
            session_id = cast(str, created["session_id"])
            assert re.fullmatch(r"akashic:[0-9a-f]{32}", session_id)
            assert created["payload"] == {"session_id": session_id}
            assert not manager.session_exists(session_id)
            historical = manager.get_or_create(session_id)
            historical.add_message(
                "user",
                "隔离历史问题",
                client_message_id="01J00000000000000000000000",
            )
            historical.add_message("assistant", "隔离历史回答")
            manager.save(historical)

            mirror: dict[str, dict[str, Any]] = {}
            history_pages: list[list[dict[str, Any]]] = []
            for command_id in (
                "01J00000000000000000000001",
                "01J00000000000000000000002",
            ):
                websocket.send_json(
                    _command(
                        command_id,
                        "history.get",
                        epoch,
                        session_id=session_id,
                        payload={"page": 1, "page_size": 50},
                    )
                )
                page = websocket.receive_json()
                reply = websocket.receive_json()
                assert page["type"] == "history.page"
                assert reply["type"] == "history.get.ok"
                items = cast(list[dict[str, Any]], page["payload"]["items"])
                history_pages.append(items)
                mirror.update({_history_identity(item): item for item in items})
                last_ack = int(page["event_seq"])
            websocket.send_json(
                {
                    "v": 1,
                    "kind": "ack",
                    "type": "event.ack",
                    "connection_epoch": epoch,
                    "payload": {"through_event_seq": last_ack},
                }
            )
            assert len(mirror) == 2
            assert [item["id"] for item in history_pages[0]] == [
                item["id"] for item in history_pages[1]
            ]
            assert history_pages[0][0]["client_message_id"] == (
                "01J00000000000000000000000"
            )

            # 4. 发送后只读到 turn.started 即断线，模拟移动网络丢帧
            live_command_id = "01J00000000000000000000003"
            websocket.send_json(
                _command(
                    live_command_id,
                    "message.send",
                    epoch,
                    session_id=session_id,
                    payload={
                        "client_message_id": live_command_id,
                        "session_id": session_id,
                        "text": "请返回固定媒体",
                        "media_refs": [],
                        "client_created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )
            first_live = websocket.receive_json()
            assert first_live["type"] == "turn.started"
            dropped_final = websocket.receive_json()
            dropped_reply = websocket.receive_json()
            assert dropped_final["type"] == "message.final"
            assert dropped_reply["type"] == "message.send.ok"

        # 4. 新 epoch 从上一个已处理历史页补发，最终回复和附件均不丢失
        with client.websocket_connect("/ws") as websocket:
            epoch = _authenticate(websocket, device_id, device_key)
            replay = _resume(websocket, epoch, last_ack=last_ack)
            assert [frame["type"] for frame in replay] == [
                "turn.started",
                "message.final",
                "sync.completed",
            ]
            final = replay[1]
            assert final["payload"]["content"] == "隔离网关固定回复"
            descriptor = final["payload"]["attachments"][0]
            terminal_seq = int(replay[-1]["event_seq"])
            websocket.send_json(
                {
                    "v": 1,
                    "kind": "ack",
                    "type": "event.ack",
                    "connection_epoch": epoch,
                    "payload": {"through_event_seq": terminal_seq},
                }
            )

            download_id = "01J00000000000000000000004"
            websocket.send_json(
                _command(
                    download_id,
                    "attachment.download",
                    epoch,
                    session_id=session_id,
                    payload={
                        "attachment_id": descriptor["attachment_id"],
                        "offset": 0,
                    },
                )
            )
            chunk = decode_attachment_chunk(websocket.receive_bytes())
            download_reply = websocket.receive_json()
            assert download_reply["type"] == "attachment.download.ok"
            assert chunk.data == reply_bytes
            assert hashlib.sha256(chunk.data).hexdigest() == descriptor["sha256"]

            # 5. 重连后的全量历史仍只对应四条 canonical message
            websocket.send_json(
                _command(
                    "01J00000000000000000000005",
                    "history.get",
                    epoch,
                    session_id=session_id,
                    payload={"page": 1, "page_size": 50},
                )
            )
            refreshed = websocket.receive_json()
            assert websocket.receive_json()["type"] == "history.get.ok"
            refreshed_items = cast(list[dict[str, Any]], refreshed["payload"]["items"])
            mirror.update({_history_identity(item): item for item in refreshed_items})
            assert len(refreshed_items) == 4
            assert len(mirror) == 4
            assert bus.inbound_count == 1
            assert bus.legacy_publish_calls == 0

        # 6. 所有持久化路径必须位于 pytest 隔离根目录
        assert (root / "gateway" / "mobile.db").is_file()
        assert (root / "workspace" / "sessions.db").is_file()
        assert (root / "attachments").is_dir()
        assert all(root in path.parents for path in root.rglob("*"))
    finally:
        asyncio.run(adapter.stop())
        asyncio.run(runtime.channel.stop())
        manager.close()
        runtime.close()

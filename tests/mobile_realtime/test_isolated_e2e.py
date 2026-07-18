from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from agent.config_models import MobileKeyEncryptionConfig, MobileRealtimeConfig
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import TurnStarted
from infra.channels.base import AttachmentStore
from infra.mobile_realtime.attachments import decode_attachment_chunk
from infra.mobile_realtime.auth import device_proof_signing_bytes
from infra.mobile_realtime.gateway import (
    MobileGatewayRuntime,
    build_mobile_gateway_runtime,
    create_mobile_gateway_app,
)
from infra.mobile_realtime.key_protection import KeyProtectionError
from infra.mobile_realtime.storage import DeviceRecord
from session.manager import SessionManager


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
    def register_channel(self, channel: str, **senders: object) -> None:
        assert channel == "mobile"


class _DeterministicAgentBus:
    """把手机入站消息持久化，并返回一条带固定媒体的确定性回复。"""

    def __init__(self, manager: SessionManager, reply_media: Path) -> None:
        self._manager = manager
        self._reply_media = reply_media
        self._runtime: MobileGatewayRuntime | None = None
        self.inbound_count = 0

    def bind(self, runtime: MobileGatewayRuntime) -> None:
        self._runtime = runtime

    def subscribe_outbound(self, channel: str, callback: object) -> None:
        assert channel == "mobile"

    async def publish_inbound(self, message: object) -> None:
        """按真实持久化顺序生成 turn.started 与 message.final。"""

        # 1. 持久化同一个 client_message_id，模拟生命周期入库结果
        inbound = cast(InboundMessage, message)
        runtime = self._require_runtime()
        session = self._manager.get_or_create(inbound.session_key)
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
                session_key=inbound.session_key,
                channel="mobile",
                chat_id=inbound.chat_id,
                content=inbound.content,
                timestamp=datetime.now(timezone.utc),
                turn_id=turn_id,
            )
        )
        await runtime.channel._on_response(
            OutboundMessage(
                channel="mobile",
                chat_id=inbound.chat_id,
                content="隔离网关固定回复",
                media=[str(self._reply_media)],
                control_turn_id=turn_id,
                session_message_id=assistant_message_id,
            )
        )

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
    session_id = f"mobile:{uuid4()}"
    historical = manager.get_or_create(session_id)
    historical.add_message(
        "user",
        "隔离历史问题",
        client_message_id="01J00000000000000000000000",
    )
    historical.add_message("assistant", "隔离历史回答")
    manager.save(historical)

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

            # 3. 发送后只读到 turn.started 即断线，模拟移动网络丢帧
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
            refreshed_items = cast(
                list[dict[str, Any]], refreshed["payload"]["items"]
            )
            mirror.update(
                {_history_identity(item): item for item in refreshed_items}
            )
            assert len(refreshed_items) == 4
            assert len(mirror) == 4
            assert bus.inbound_count == 1

        # 6. 所有持久化路径必须位于 pytest 隔离根目录
        assert (root / "gateway" / "mobile.db").is_file()
        assert (root / "workspace" / "sessions.db").is_file()
        assert (root / "attachments").is_dir()
        assert all(root in path.parents for path in root.rglob("*"))
    finally:
        asyncio.run(runtime.channel.stop())
        manager.close()
        runtime.close()

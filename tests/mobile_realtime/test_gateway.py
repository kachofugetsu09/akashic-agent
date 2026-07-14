from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent.config_models import MobileKeyEncryptionConfig, MobileRealtimeConfig
from bus.events import OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from infra.channels.base import AttachmentStore
from infra.mobile_realtime.attachments import AttachmentChunk, encode_attachment_chunk
from infra.mobile_realtime.auth import device_proof_signing_bytes
from infra.mobile_realtime.gateway import (
    build_mobile_gateway_runtime,
    build_mobile_gateway_server,
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
            raise KeyProtectionError("测试 master key 不存在") from error


def _config() -> MobileRealtimeConfig:
    return MobileRealtimeConfig(
        enabled=True,
        database=Path("data/mobile.db"),
        lan_hostname="akashic.local",
        public_url="wss://agent.example.com/ws",
        key_encryption=MobileKeyEncryptionConfig(
            keyset_manifest=Path("data/mobile/keys/current.json")
        ),
    )


def _device_public_key(private_key: ec.EllipticCurvePrivateKey) -> str:
    encoded = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return base64.b64encode(encoded).decode("ascii")


def _device_proof(
    *,
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


def test_authenticated_gateway_requires_resume_and_acks_durable_sync(
    tmp_path: Path,
) -> None:
    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    import asyncio

    runtime, keyset = asyncio.run(build())
    server = build_mobile_gateway_server(runtime, keyset)
    assert server.config.loaded is True
    assert server.config.is_ssl is True
    assert server.config.ssl is not None
    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name="Pixel Emulator",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )
    client = TestClient(create_mobile_gateway_app(runtime))

    with client.websocket_connect("/ws") as websocket:
        challenge_frame = websocket.receive_json()
        assert challenge_frame["type"] == "server.challenge"
        websocket.send_json(
            _device_proof(
                challenge=challenge_frame["payload"],
                device_id=device_id,
                device_key=device_key,
            )
        )
        accepted = websocket.receive_json()
        epoch = accepted["connection_epoch"]
        assert accepted["type"] == "auth.accepted"

        websocket.send_json(
            {
                "v": 1,
                "kind": "control",
                "type": "resume",
                "connection_epoch": epoch,
                "payload": {"last_ack": 0, "active_turns": []},
            }
        )
        synced = websocket.receive_json()
        assert synced["type"] == "sync.completed"
        assert synced["event_seq"] == 1
        websocket.send_json(
            {
                "v": 1,
                "kind": "ack",
                "type": "event.ack",
                "connection_epoch": epoch,
                "payload": {"through_event_seq": 1},
            }
        )
        websocket.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "ping",
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "connection_epoch": epoch,
                "payload": {},
            }
        )
        reply = websocket.receive_json()
        assert reply["type"] == "ping.ok"

    cursor = runtime.storage.read_cursor(device_id)
    assert cursor.acknowledged_event_seq == 1
    assert (
        runtime.storage.read_durable_events(
            device_id,
            after_event_seq=0,
            limit=10,
        )
        == ()
    )
    runtime.close()


def test_gateway_restart_allocates_epoch_newer_than_previous_connection(
    tmp_path: Path,
) -> None:
    import asyncio

    master_keys = _EphemeralMasterKeys()

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=master_keys,
        )

    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime, _ = asyncio.run(build())
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name="Pixel Emulator",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )
    with TestClient(create_mobile_gateway_app(runtime)).websocket_connect("/ws") as ws:
        challenge = ws.receive_json()
        ws.send_json(
            _device_proof(
                challenge=challenge["payload"],
                device_id=device_id,
                device_key=device_key,
            )
        )
        first_epoch = ws.receive_json()["connection_epoch"]
    runtime.close()

    restarted, _ = asyncio.run(build())
    with TestClient(create_mobile_gateway_app(restarted)).websocket_connect(
        "/ws"
    ) as ws:
        challenge = ws.receive_json()
        ws.send_json(
            _device_proof(
                challenge=challenge["payload"],
                device_id=device_id,
                device_key=device_key,
            )
        )
        restarted_epoch = ws.receive_json()["connection_epoch"]

    assert restarted_epoch > first_epoch
    restarted.close()


def test_gateway_rejects_business_frame_before_device_authentication(
    tmp_path: Path,
) -> None:
    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    import asyncio

    runtime, _ = asyncio.run(build())
    client = TestClient(create_mobile_gateway_app(runtime))
    with client.websocket_connect("/ws") as websocket:
        _ = websocket.receive_json()
        websocket.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "ping",
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "connection_epoch": 1,
                "payload": {},
            }
        )
        error = websocket.receive_json()
        assert error["type"] == "protocol.error"
        assert error["payload"]["code"] == 4401
    runtime.close()


def test_authenticated_gateway_closes_poison_message_command(
    tmp_path: Path,
) -> None:
    """验证超长消息可归属拒绝，客户端不会无限重放同一命令。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    import asyncio

    runtime, _ = asyncio.run(build())
    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name="Pixel Emulator",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )
    session_id = f"mobile:{uuid4()}"

    with TestClient(create_mobile_gateway_app(runtime)).websocket_connect("/ws") as ws:
        challenge = ws.receive_json()
        ws.send_json(
            _device_proof(
                challenge=challenge["payload"],
                device_id=device_id,
                device_key=device_key,
            )
        )
        epoch = ws.receive_json()["connection_epoch"]
        ws.send_json(
            {
                "v": 1,
                "kind": "control",
                "type": "resume",
                "connection_epoch": epoch,
                "payload": {"last_ack": 0, "active_turns": []},
            }
        )
        assert ws.receive_json()["type"] == "sync.completed"
        ws.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "message.send",
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "connection_epoch": epoch,
                "session_id": session_id,
                "payload": {
                    "client_message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                    "session_id": session_id,
                    "text": "x" * 65_537,
                    "media_refs": [],
                    "client_created_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        )
        error = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()

    assert error["type"] == "protocol.error"
    assert error["payload"] == {"code": 4410, "message": "协议字段无效"}
    assert closed.value.code == 4410
    assert closed.value.reason == "协议字段无效"
    runtime.close()


def test_resume_atomically_rebases_when_client_ack_is_ahead(
    tmp_path: Path,
) -> None:
    """验证服务端 DB 回退后以客户端序号续写 durable reset。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    import asyncio
    import json

    runtime, _ = asyncio.run(build())
    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name="Pixel Emulator",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )

    with TestClient(create_mobile_gateway_app(runtime)).websocket_connect("/ws") as ws:
        challenge = ws.receive_json()
        ws.send_json(
            _device_proof(
                challenge=challenge["payload"],
                device_id=device_id,
                device_key=device_key,
            )
        )
        epoch = ws.receive_json()["connection_epoch"]
        ws.send_json(
            {
                "v": 1,
                "kind": "control",
                "type": "resume",
                "connection_epoch": epoch,
                "payload": {"last_ack": 5, "active_turns": []},
            }
        )
        reset = ws.receive_json()

    assert reset["type"] == "sync.reset_required"
    assert reset["event_seq"] == 6
    assert reset["payload"]["reason"] == "client_ack_ahead_of_server_cursor"
    cursor = runtime.storage.read_cursor(device_id)
    assert (cursor.next_event_seq, cursor.sent_event_seq, cursor.acknowledged_event_seq) == (
        7,
        6,
        5,
    )
    events = runtime.storage.read_durable_events(
        device_id,
        after_event_seq=5,
        limit=10,
    )
    assert [event.event_seq for event in events] == [6]
    assert json.loads(events[0].envelope_json)["type"] == "sync.reset_required"
    runtime.close()


def test_authenticated_gateway_rejects_malformed_attachment_binary(
    tmp_path: Path,
) -> None:
    """验证坏二进制帧以明确协议错误关闭，而不是逃出 ASGI。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    import asyncio

    runtime, _ = asyncio.run(build())
    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name="Pixel Emulator",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("attachments-v1",),
        )
    )

    with TestClient(create_mobile_gateway_app(runtime)).websocket_connect("/ws") as ws:
        challenge = ws.receive_json()
        ws.send_json(
            _device_proof(
                challenge=challenge["payload"],
                device_id=device_id,
                device_key=device_key,
            )
        )
        epoch = ws.receive_json()["connection_epoch"]
        ws.send_json(
            {
                "v": 1,
                "kind": "control",
                "type": "resume",
                "connection_epoch": epoch,
                "payload": {"last_ack": 0, "active_turns": []},
            }
        )
        assert ws.receive_json()["type"] == "sync.completed"
        ws.send_bytes(b"\x00")
        error = ws.receive_json()

    assert error["type"] == "protocol.error"
    assert error["payload"]["code"] == 4400
    runtime.close()


def test_offline_proactive_event_is_durable_and_replayed_with_session(
    tmp_path: Path,
) -> None:
    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    import asyncio

    runtime, _ = asyncio.run(build())
    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name="Pixel Emulator",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )
    chat_id = str(uuid4())
    session_id = f"mobile:{chat_id}"
    runtime.storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    asyncio.run(runtime.channel.send(chat_id, "后台任务完成"))
    assert runtime.storage.count_durable_events(device_id) == 1

    client = TestClient(create_mobile_gateway_app(runtime))
    with client.websocket_connect("/ws") as websocket:
        challenge_frame = websocket.receive_json()
        websocket.send_json(
            _device_proof(
                challenge=challenge_frame["payload"],
                device_id=device_id,
                device_key=device_key,
            )
        )
        accepted = websocket.receive_json()
        epoch = accepted["connection_epoch"]
        websocket.send_json(
            {
                "v": 1,
                "kind": "control",
                "type": "resume",
                "connection_epoch": epoch,
                "payload": {"last_ack": 0, "active_turns": []},
            }
        )
        proactive = websocket.receive_json()
        synced = websocket.receive_json()

    assert proactive["type"] == "message.proactive"
    assert proactive["session_id"] == session_id
    assert proactive["payload"]["content"] == "后台任务完成"
    assert synced["type"] == "sync.completed"
    runtime.close()


def test_authenticated_message_send_reaches_agent_event_path_once(
    tmp_path: Path,
) -> None:
    """验证 WSS command 进入 InboundMessage，并按顺序返回完整事件流。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    class LoopbackBus:
        def __init__(self) -> None:
            self.inbound: list[object] = []

        def subscribe_outbound(self, channel: str, callback: object) -> None:
            assert channel == "mobile"

        async def publish_inbound(self, message: object) -> None:
            from bus.events import InboundMessage

            assert isinstance(message, InboundMessage)
            self.inbound.append(message)
            turn_id = uuid4().hex
            await runtime.channel._on_turn_started(
                TurnStarted(
                    session_key=message.session_key,
                    channel=message.channel,
                    chat_id=message.chat_id,
                    content=message.content,
                    timestamp=message.timestamp,
                    turn_id=turn_id,
                )
            )
            await runtime.channel._on_stream_delta(
                StreamDeltaReady(
                    session_key=message.session_key,
                    channel=message.channel,
                    chat_id=message.chat_id,
                    turn_id=turn_id,
                    thinking_delta="先检查",
                )
            )
            await runtime.channel._on_tool_call_started(
                ToolCallStarted(
                    session_key=message.session_key,
                    channel=message.channel,
                    chat_id=message.chat_id,
                    iteration=1,
                    call_id="call-1",
                    tool_name="shell",
                    arguments={"command": "pwd"},
                    turn_id=turn_id,
                )
            )
            await runtime.channel._on_tool_call_completed(
                ToolCallCompleted(
                    session_key=message.session_key,
                    channel=message.channel,
                    chat_id=message.chat_id,
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
            await runtime.channel._on_stream_delta(
                StreamDeltaReady(
                    session_key=message.session_key,
                    channel=message.channel,
                    chat_id=message.chat_id,
                    turn_id=turn_id,
                    thinking_delta="工具后继续",
                )
            )
            await runtime.channel._on_response(
                OutboundMessage(
                    channel="mobile",
                    chat_id=message.chat_id,
                    content="完成",
                    thinking="先检查",
                    control_turn_id=turn_id,
                )
            )

    class FakeEventBus:
        def on(self, event_type: type[object], callback: object) -> None:
            return None

    class FakePushTool:
        def register_channel(self, channel: str, **senders: object) -> None:
            assert channel == "mobile"

    import asyncio

    runtime, _ = asyncio.run(build())
    bus = LoopbackBus()
    asyncio.run(
        runtime.channel.start(
            cast(
                Any,
                SimpleNamespace(
                    bus=bus,
                    session_manager=SessionManager(tmp_path / "sessions"),
                    event_bus=FakeEventBus(),
                    push_tool=FakePushTool(),
                    interrupt_controller=None,
                    attachment_store=AttachmentStore(tmp_path / "uploads"),
                ),
            )
        )
    )
    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name="Pixel Emulator",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )
    session_id = f"mobile:{uuid4()}"
    command_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    command = {
        "v": 1,
        "kind": "command",
        "type": "message.send",
        "id": command_id,
        "connection_epoch": 1,
        "session_id": session_id,
        "payload": {
            "client_message_id": command_id,
            "session_id": session_id,
            "text": "帮我检查",
            "media_refs": [],
            "client_created_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    client = TestClient(create_mobile_gateway_app(runtime))
    with client.websocket_connect("/ws") as websocket:
        challenge_frame = websocket.receive_json()
        websocket.send_json(
            _device_proof(
                challenge=challenge_frame["payload"],
                device_id=device_id,
                device_key=device_key,
            )
        )
        accepted = websocket.receive_json()
        epoch = accepted["connection_epoch"]
        websocket.send_json(
            {
                "v": 1,
                "kind": "control",
                "type": "resume",
                "connection_epoch": epoch,
                "payload": {"last_ack": 0, "active_turns": []},
            }
        )
        assert websocket.receive_json()["type"] == "sync.completed"
        command["connection_epoch"] = epoch
        websocket.send_json(command)
        frames = [websocket.receive_json() for _ in range(7)]
        assert [frame["type"] for frame in frames] == [
            "turn.started",
            "react.thinking.delta",
            "react.tool.started",
            "react.tool.completed",
            "react.thinking.delta",
            "message.final",
            "message.send.ok",
        ]
        first_thinking, tool_started, tool_completed, second_thinking = (
            frames[1],
            frames[2],
            frames[3],
            frames[4],
        )
        assert first_thinking["payload"]["ordinal"] == 0
        assert tool_started["payload"]["ordinal"] == 1
        assert tool_completed["payload"]["ordinal"] == 1
        assert (
            tool_completed["payload"]["block_id"] == tool_started["payload"]["block_id"]
        )
        assert second_thinking["payload"]["ordinal"] == 2

        websocket.send_json(command)
        websocket.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "ping",
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FAX",
                "connection_epoch": epoch,
                "payload": {},
            }
        )
        assert websocket.receive_json()["type"] == "message.send.ok"
        assert websocket.receive_json()["type"] == "ping.ok"

    assert len(bus.inbound) == 1
    asyncio.run(runtime.channel.stop())
    runtime.close()


def test_attachment_upload_resumes_and_reaches_agent_media(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """验证二进制上传跨连接续传，并以 media_refs 进入 Agent。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    class CaptureBus:
        def __init__(self) -> None:
            self.inbound: list[object] = []

        def subscribe_outbound(self, channel: str, callback: object) -> None:
            assert channel == "mobile"

        async def publish_inbound(self, message: object) -> None:
            self.inbound.append(message)

    class FakeEventBus:
        def on(self, event_type: type[object], callback: object) -> None:
            return None

    class FakePushTool:
        def register_channel(self, channel: str, **senders: object) -> None:
            assert channel == "mobile"

    import asyncio

    runtime, _ = asyncio.run(build())
    request.addfinalizer(runtime.close)
    bus = CaptureBus()
    asyncio.run(
        runtime.channel.start(
            cast(
                Any,
                SimpleNamespace(
                    bus=bus,
                    session_manager=SessionManager(tmp_path / "sessions"),
                    event_bus=FakeEventBus(),
                    push_tool=FakePushTool(),
                    interrupt_controller=None,
                    attachment_store=AttachmentStore(tmp_path / "uploads"),
                ),
            )
        )
    )
    request.addfinalizer(lambda: asyncio.run(runtime.channel.stop()))
    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name="Pixel Emulator",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1", "attachments-v1"),
        )
    )
    session_id = f"mobile:{uuid4()}"
    attachment_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    confirmed_offset = 1024 * 1024
    content = b"a" * confirmed_offset + b"resumed payload"
    digest = hashlib.sha256(content).hexdigest()
    client = TestClient(create_mobile_gateway_app(runtime))

    with client.websocket_connect("/ws") as websocket:
        challenge = websocket.receive_json()
        websocket.send_json(
            _device_proof(
                challenge=challenge["payload"],
                device_id=device_id,
                device_key=device_key,
            )
        )
        epoch = websocket.receive_json()["connection_epoch"]
        websocket.send_json(
            {
                "v": 1,
                "kind": "control",
                "type": "resume",
                "connection_epoch": epoch,
                "payload": {"last_ack": 0, "active_turns": []},
            }
        )
        synced = websocket.receive_json()
        assert synced["type"] == "sync.completed"
        websocket.send_json(
            {
                "v": 1,
                "kind": "ack",
                "type": "event.ack",
                "connection_epoch": epoch,
                "payload": {"through_event_seq": synced["event_seq"]},
            }
        )
        websocket.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "attachment.begin",
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FAW",
                "connection_epoch": epoch,
                "session_id": session_id,
                "payload": {
                    "attachment_id": attachment_id,
                    "filename": "meme.png",
                    "content_type": "image/png",
                    "size_bytes": len(content),
                    "sha256": digest,
                },
            }
        )
        begin = websocket.receive_json()
        assert begin["type"] == "attachment.begin.ok"
        assert begin["payload"]["next_offset"] == 0
        for offset in range(0, confirmed_offset, 128 * 1024):
            websocket.send_bytes(
                encode_attachment_chunk(
                    AttachmentChunk(
                        attachment_id,
                        offset,
                        content[offset : offset + 128 * 1024],
                    )
                )
            )
        confirmed = websocket.receive_json()
        assert confirmed["type"] == "attachment.progress"
        assert confirmed["payload"]["transferred_bytes"] == confirmed_offset

    with client.websocket_connect("/ws") as websocket:
        challenge = websocket.receive_json()
        websocket.send_json(
            _device_proof(
                challenge=challenge["payload"],
                device_id=device_id,
                device_key=device_key,
            )
        )
        epoch = websocket.receive_json()["connection_epoch"]
        websocket.send_json(
            {
                "v": 1,
                "kind": "control",
                "type": "resume",
                "connection_epoch": epoch,
                "payload": {
                    "last_ack": confirmed["event_seq"],
                    "active_turns": [],
                },
            }
        )
        assert websocket.receive_json()["type"] == "sync.completed"
        websocket.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "attachment.begin",
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FAX",
                "connection_epoch": epoch,
                "session_id": session_id,
                "payload": {
                    "attachment_id": attachment_id,
                    "filename": "meme.png",
                    "content_type": "image/png",
                    "size_bytes": len(content),
                    "sha256": digest,
                },
            }
        )
        resumed = websocket.receive_json()
        assert resumed["payload"]["next_offset"] == confirmed_offset
        websocket.send_bytes(
            encode_attachment_chunk(
                AttachmentChunk(
                    attachment_id,
                    confirmed_offset,
                    content[confirmed_offset:],
                )
            )
        )
        progress = websocket.receive_json()
        assert progress["type"] == "attachment.progress"
        assert progress["payload"]["transferred_bytes"] == len(content)
        websocket.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "attachment.finish",
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FAY",
                "connection_epoch": epoch,
                "session_id": session_id,
                "payload": {"attachment_id": attachment_id},
            }
        )
        assert websocket.receive_json()["type"] == "attachment.ready"
        assert websocket.receive_json()["type"] == "attachment.finish.ok"
        websocket.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "message.send",
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FAZ",
                "connection_epoch": epoch,
                "session_id": session_id,
                "payload": {
                    "client_message_id": "01ARZ3NDEKTSV4RRFFQ69G5FAZ",
                    "session_id": session_id,
                    "text": "",
                    "media_refs": [attachment_id],
                    "client_created_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        )
        assert websocket.receive_json()["type"] == "message.send.ok"

    from bus.events import InboundMessage

    assert len(bus.inbound) == 1
    assert isinstance(bus.inbound[0], InboundMessage)
    assert bus.inbound[0].content == ""
    assert len(bus.inbound[0].media) == 1
    assert Path(bus.inbound[0].media[0]).read_bytes() == content

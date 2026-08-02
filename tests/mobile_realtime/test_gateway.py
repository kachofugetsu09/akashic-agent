from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import sqlite3
from collections import deque
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
import infra.mobile_realtime.gateway as gateway_module
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent.config_models import MobileKeyEncryptionConfig, MobileRealtimeConfig
from agent.plugins.mobile_ui import MobileUiRpcExecutionError
from bus.events import OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from infra.channels.base import AttachmentStore
from infra.mobile_realtime.attachments import (
    AttachmentChunk,
    AttachmentTransferService,
    MAX_ATTACHMENT_CHUNK_BYTES,
    attachment_descriptor,
    decode_attachment_chunk,
    encode_attachment_chunk,
)
from infra.mobile_realtime.auth import device_proof_signing_bytes
from infra.mobile_realtime.gateway import (
    ActiveMobileConnection,
    build_mobile_gateway_runtime,
    build_mobile_gateway_server,
    create_mobile_gateway_app,
)
from infra.mobile_realtime.key_protection import KeyProtectionError
from infra.mobile_realtime.plugin_ui_http import (
    PluginUiHttpTicketError,
    PluginUiHttpTicketIssuer,
)
from infra.mobile_realtime.protocol import AttachmentDownloadCommand, parse_frame
from infra.mobile_realtime.storage import DeviceRecord, MobileStorageError
from session.manager import SessionManager


@pytest.mark.asyncio
async def test_gateway_atomically_publishes_proactive_attachment(
    tmp_path: Path,
) -> None:
    runtime, _keyset = build_mobile_gateway_runtime(
        _config(),
        tmp_path,
        master_keys=_EphemeralMasterKeys(),
    )
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key="test-public-key",
            display_name="Pixel",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("attachments",),
        )
    )
    source = tmp_path / "report.pdf"
    source.write_bytes(b"report")
    service = AttachmentTransferService(
        runtime.storage,
        AttachmentStore(tmp_path / "uploads"),
        max_attachment_bytes=1024,
    )
    candidates = service.snapshot_outbound_batch(
        session_id="mobile:session-1",
        local_media_paths=(source,),
    )
    try:
        resolved = await runtime.publish_event_with_outbound_attachments(
            candidates=candidates,
            session_id="mobile:session-1",
            payload_builder=lambda records: {
                "content": "报告",
                "attachments": [
                    attachment_descriptor(record) for record in records
                ],
                "metadata": {"source": "message_push"},
                "delivery_id": "delivery-1",
            },
        )

        replay = runtime.storage.read_durable_events(
            device_id,
            after_event_seq=0,
            limit=10,
        )
        assert len(resolved) == 1
        assert runtime.storage.read_attachment(resolved[0].attachment_id) == resolved[0]
        assert len(replay) == 1
        envelope = json.loads(replay[0].envelope_json)
        assert envelope["payload"]["attachments"][0]["attachment_id"] == (
            resolved[0].attachment_id
        )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_gateway_rejects_attachment_commit_after_last_device_disappears(
    tmp_path: Path,
) -> None:
    runtime, _keyset = build_mobile_gateway_runtime(
        _config(),
        tmp_path,
        master_keys=_EphemeralMasterKeys(),
    )
    source = tmp_path / "report.pdf"
    source.write_bytes(b"report")
    service = AttachmentTransferService(
        runtime.storage,
        AttachmentStore(tmp_path / "uploads"),
        max_attachment_bytes=1024,
    )
    candidates = service.snapshot_outbound_batch(
        session_id="mobile:session-1",
        local_media_paths=(source,),
    )
    try:
        with pytest.raises(MobileStorageError, match="没有可提交的目标设备"):
            await runtime.publish_event_with_outbound_attachments(
                candidates=candidates,
                session_id="mobile:session-1",
                payload_builder=lambda _records: {
                    "content": "报告",
                    "attachments": [],
                },
            )
    finally:
        service.cleanup_outbound_candidates(candidates)
        runtime.close()


class _ControlledWebSocket:
    def __init__(
        self,
        *,
        send_gate: asyncio.Event | None = None,
        close_gate: asyncio.Event | None = None,
        bytes_gate: asyncio.Event | None = None,
    ) -> None:
        self.send_gate = send_gate
        self.close_gate = close_gate
        self.bytes_gate = bytes_gate
        self.send_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.bytes_started = asyncio.Event()
        self.sent_text: list[str] = []
        self.wire_order: list[str] = []
        self.close_calls: list[tuple[int, str]] = []

    async def send_text(self, text: str) -> None:
        self.send_started.set()
        if self.send_gate is not None:
            await self.send_gate.wait()
        self.sent_text.append(text)
        self.wire_order.append(str(json.loads(text)["kind"]))

    async def send_bytes(self, data: bytes) -> None:
        self.bytes_started.set()
        self.wire_order.append("bytes")
        if self.bytes_gate is not None:
            await self.bytes_gate.wait()

    async def send_json(self, data: object) -> None:
        self.wire_order.append("reply")

    async def close(self, *, code: int, reason: str) -> None:
        self.close_started.set()
        self.close_calls.append((code, reason))
        if self.close_gate is not None:
            await self.close_gate.wait()


def _register_test_device(runtime: Any, device_id: str) -> None:
    device_key = ec.generate_private_key(ec.SECP256R1())
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name=f"Device {device_id}",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )


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


def test_resume_reset_terminal_explicitly_bridges_client_sequence_gap(
    tmp_path: Path,
) -> None:
    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build())
    device_id = uuid4().hex
    _register_test_device(runtime, device_id)
    try:
        for index in range(3):
            runtime._enqueue_event(
                device_id=device_id,
                event_type="message.final",
                payload={"content": f"message-{index}"},
            )
        runtime.inbox.mark_sent(device_id, through_event_seq=3)
        runtime.inbox.acknowledge(device_id, through_event_seq=3)

        replay_after, replay_through, terminal = runtime._prepare_resume(
            device_id=device_id,
            last_ack=1,
        )

        assert (replay_after, replay_through) == (1, 1)
        assert terminal.event_seq == 4
        stored = json.loads(terminal.envelope_json)
        assert stored["type"] == "sync.reset_required"
        assert stored["payload"]["reason"] == "client_ack_behind_server_cursor"
    finally:
        runtime.close()


def test_resume_rebases_when_authenticated_client_ack_is_ahead(
    tmp_path: Path,
) -> None:
    """服务端 durable DB 回退后，下一帧继续客户端序号并要求全量重建。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build())
    device_id = uuid4().hex
    _register_test_device(runtime, device_id)
    try:
        runtime._enqueue_event(
            device_id=device_id,
            event_type="message.final",
            payload={"content": "rolled-back"},
        )

        replay_after, replay_through, terminal = runtime._prepare_resume(
            device_id=device_id,
            last_ack=5,
        )

        assert (replay_after, replay_through) == (5, 5)
        assert terminal.event_seq == 6
        stored = json.loads(terminal.envelope_json)
        assert stored["type"] == "sync.reset_required"
        assert stored["payload"]["reason"] == "client_ack_ahead_of_server_cursor"
        cursor = runtime.storage.read_cursor(device_id)
        assert cursor.next_event_seq == 7
        assert cursor.sent_event_seq == 5
        assert cursor.acknowledged_event_seq == 5
        assert [event.event_seq for event in runtime.storage.read_durable_events(
            device_id,
            after_event_seq=5,
            limit=10,
        )] == [6]
    finally:
        runtime.close()


def test_rebased_reset_survives_runtime_restart(tmp_path: Path) -> None:
    """游标重定位提交后即使进程退出，重连也只能先收到已落盘 reset。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=master_keys,
        )

    master_keys = _EphemeralMasterKeys()
    device_id = uuid4().hex
    runtime, _ = asyncio.run(build())
    _register_test_device(runtime, device_id)
    runtime._enqueue_event(
        device_id=device_id,
        event_type="message.final",
        payload={"content": "rolled-back"},
    )
    _, _, reset = runtime._prepare_resume(device_id=device_id, last_ack=5)
    assert reset.event_seq == 6
    runtime.close()

    restarted, _ = asyncio.run(build())
    try:
        replay_after, replay_through, terminal = restarted._prepare_resume(
            device_id=device_id,
            last_ack=5,
        )

        assert (replay_after, replay_through) == (5, 5)
        assert terminal.event_seq == 6
        assert json.loads(terminal.envelope_json)["type"] == "sync.reset_required"
        assert restarted.storage.read_cursor(device_id).next_event_seq == 7
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_rebased_reset_replays_events_written_before_reconnect(
    tmp_path: Path,
) -> None:
    """reset 后离线写入的事件必须随首次重连连续送达。"""

    def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=master_keys,
        )

    master_keys = _EphemeralMasterKeys()
    device_id = uuid4().hex
    runtime, _ = build()
    _register_test_device(runtime, device_id)
    runtime._enqueue_event(
        device_id=device_id,
        event_type="message.final",
        payload={"content": "rolled-back"},
    )
    _, _, reset = runtime._prepare_resume(device_id=device_id, last_ack=5)
    assert reset.event_seq == 6
    runtime.close()

    restarted, _ = build()
    try:
        offline = restarted._enqueue_event(
            device_id=device_id,
            event_type="message.final",
            payload={"content": "offline-after-reset"},
        )
        assert offline.event_seq == 7
        websocket = _ControlledWebSocket()

        await restarted._resume_and_register(
            cast(Any, websocket),
            device_id=device_id,
            connection_epoch=1,
            last_ack=5,
        )
        await restarted.publish_event(
            event_type="message.final",
            payload={"content": "live-after-resume"},
            device_id=device_id,
        )

        async def wait_for_live_event() -> None:
            while len(websocket.sent_text) < 3:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_live_event(), timeout=1)
        frames = [json.loads(text) for text in websocket.sent_text]
        assert [frame["event_seq"] for frame in frames] == [6, 7, 8]
        assert [frame["type"] for frame in frames] == [
            "sync.reset_required",
            "message.final",
            "message.final",
        ]
    finally:
        restarted.close()


def test_rebase_storage_rejects_ack_outside_sqlite_range(tmp_path: Path) -> None:
    """存储 owner 不接受无法继续分配 reset 的客户端序号。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build())
    device_id = uuid4().hex
    _register_test_device(runtime, device_id)
    try:
        with pytest.raises(ValueError, match="SQLite 序号范围"):
            runtime.storage.rebase_cursor_with_durable_event(
                device_id,
                through_event_seq=(1 << 63) - 2,
                event_id=uuid4().hex,
                envelope_json='{"type":"sync.reset_required"}',
                created_at=datetime.now(timezone.utc),
            )
    finally:
        runtime.close()


def test_maximum_rebase_ack_can_complete_next_resume(tmp_path: Path) -> None:
    """最大合法恢复 ACK 仍为 reset 确认后的完成帧保留充足序号空间。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build())
    device_id = uuid4().hex
    _register_test_device(runtime, device_id)
    maximum_ack = 1 << 62
    try:
        _, _, reset = runtime._prepare_resume(
            device_id=device_id,
            last_ack=maximum_ack,
        )
        assert reset.event_seq == maximum_ack + 1
        runtime.storage.mark_events_sent(
            device_id,
            through_event_seq=reset.event_seq,
        )
        runtime.storage.acknowledge_durable_events(
            device_id,
            through_event_seq=reset.event_seq,
        )

        replay_after, replay_through, completed = runtime._prepare_resume(
            device_id=device_id,
            last_ack=reset.event_seq,
        )

        assert (replay_after, replay_through) == (reset.event_seq, reset.event_seq)
        assert completed.event_seq == maximum_ack + 2
        assert json.loads(completed.envelope_json)["type"] == "sync.completed"
    finally:
        runtime.close()


def test_ahead_ack_rebases_before_expired_inbox_check(tmp_path: Path) -> None:
    """服务端回退与旧事件过期并存时，仍按客户端下一序号原子 reset。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build())
    device_id = uuid4().hex
    _register_test_device(runtime, device_id)
    runtime.storage.append_durable_event(
        device_id=device_id,
        event_id=uuid4().hex,
        envelope_json='{"type":"message.final"}',
        created_at=datetime.now(timezone.utc) - timedelta(days=8),
    )
    try:
        replay_after, replay_through, terminal = runtime._prepare_resume(
            device_id=device_id,
            last_ack=5,
        )

        assert (replay_after, replay_through) == (5, 5)
        assert terminal.event_seq == 6
        assert json.loads(terminal.envelope_json)["type"] == "sync.reset_required"
        assert [event.event_seq for event in runtime.storage.read_durable_events(
            device_id,
            after_event_seq=5,
            limit=10,
        )] == [6]
    finally:
        runtime.close()


def test_resume_resets_when_durable_inbox_has_sequence_gap(
    tmp_path: Path,
) -> None:
    """持久化窗口缺号时直接重建，避免客户端永久重连。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build())
    device_id = uuid4().hex
    _register_test_device(runtime, device_id)
    try:
        for index in range(3):
            runtime._enqueue_event(
                device_id=device_id,
                event_type="message.final",
                payload={"content": f"message-{index}"},
            )
        with closing(sqlite3.connect(runtime.storage.db_path)) as db, db:
            db.execute(
                "DELETE FROM mobile_device_inbox WHERE device_id = ? AND event_seq = ?",
                (device_id, 1),
            )

        replay_after, replay_through, terminal = runtime._prepare_resume(
            device_id=device_id,
            last_ack=0,
        )

        assert (replay_after, replay_through) == (0, 0)
        assert terminal.event_seq == 4
        stored = json.loads(terminal.envelope_json)
        assert stored["type"] == "sync.reset_required"
        assert stored["payload"]["reason"] == "inbox_sequence_gap"
        assert runtime.storage.count_durable_events(device_id) == 3

        runtime.inbox.mark_sent(device_id, through_event_seq=terminal.event_seq)
        runtime.inbox.acknowledge(device_id, through_event_seq=terminal.event_seq)
        replay_after, replay_through, completed = runtime._prepare_resume(
            device_id=device_id,
            last_ack=terminal.event_seq,
        )

        assert (replay_after, replay_through) == (4, 4)
        assert json.loads(completed.envelope_json)["type"] == "sync.completed"
    finally:
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


def test_authenticated_gateway_reports_validation_without_echoing_payload(
    tmp_path: Path,
) -> None:
    """验证协议字段错误可定位，但不会把用户载荷写回手机或关闭原因。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

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
                    "text": "private-message-body",
                    "media_refs": [],
                    "client_created_at": datetime.now(timezone.utc).isoformat(),
                    "reply_to": "private-reference-body",
                },
            }
        )
        error = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()

    assert error["type"] == "protocol.error"
    assert error["payload"]["code"] == 4410
    assert error["payload"]["message"] == "协议字段无效"
    assert "private-message-body" not in error["payload"]["message"]
    assert "private-reference-body" not in error["payload"]["message"]
    assert closed.value.code == 4410
    assert closed.value.reason == "协议字段无效"
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
    second_device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=second_device_id,
            public_key=_device_public_key(ec.generate_private_key(ec.SECP256R1())),
            display_name="Second Pixel",
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
    assert runtime.storage.count_durable_events(second_device_id) == 1

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
                    status="success",
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


def test_plugin_ui_failure_keeps_authenticated_websocket_available(
    tmp_path: Path,
) -> None:
    """插件面板失败后，同一连接仍能继续处理命令。"""

    class FailedMobileUiProvider:
        def catalog(self) -> dict[str, object]:
            return {"catalog_revision": "a" * 64, "items": []}

        async def query(self, *args: object, **kwargs: object) -> dict[str, object]:
            raise MobileUiRpcExecutionError(
                "插件 mobile UI RPC 执行失败: fitbit@mobile-lab.fitbit.overview"
            )

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build())
    runtime.channel.bind_mobile_ui_provider(cast(Any, FailedMobileUiProvider()))
    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name="Pixel7",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )

    # 1. 完成认证，并在活动连接上触发插件失败
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
        websocket.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "plugin.ui.query",
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FB5",
                "connection_epoch": epoch,
                "payload": {
                    "owner_id": "dashboard:fitbit",
                    "plugin_id": "fitbit@mobile-lab",
                    "plugin_revision": "revision-1",
                    "method": "fitbit.overview",
                    "payload": {},
                    "slot": "dashboard.main",
                },
            }
        )
        failed = websocket.receive_json()
        assert failed["type"] == "plugin.ui.query.error"
        assert failed["payload"]["code"] == "plugin_failed"

        # 2. 错误回复不能改变 epoch，也不能阻断后续命令
        websocket.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "ping",
                "id": "01ARZ3NDEKTSV4RRFFQ69G5FB6",
                "connection_epoch": epoch,
                "payload": {},
            }
        )
        assert websocket.receive_json()["type"] == "ping.ok"

    runtime.close()


def test_plugin_ui_https_data_plane_uses_signed_request_bound_ticket(
    tmp_path: Path,
) -> None:
    query_calls: list[dict[str, object]] = []

    class MobileUiProvider:
        def catalog(self) -> dict[str, object]:
            return {"catalog_revision": "a" * 64, "items": []}

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
            query_calls.append(payload)
            return {
                "schema": "akasha.recall-card.v1",
                "plugin_id": plugin_id,
                "plugin_revision": plugin_revision,
                "method": method,
                "payload": payload,
                "session_id": session_id,
                "turn_id": turn_id,
            }

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, _ = asyncio.run(build())
    runtime.channel.bind_mobile_ui_provider(cast(Any, MobileUiProvider()))
    device_key = ec.generate_private_key(ec.SECP256R1())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(device_key),
            display_name="Pixel7",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )
    command_id = "01ARZ3NDEKTSV4RRFFQ69G5FB7"
    query_payload = {
        "owner_id": "turn:akasha",
        "plugin_id": "akasha@builtin",
        "plugin_revision": "revision-1",
        "method": "recall.current",
        "payload": {"message_id": "message:446"},
        "slot": "turn.before_reasoning",
    }
    request_body = {
        "request_id": command_id,
        **query_payload,
        "session_id": None,
        "turn_id": None,
    }

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
        websocket.send_json(
            {
                "v": 1,
                "kind": "command",
                "type": "plugin.ui.query.prepare",
                "id": command_id,
                "connection_epoch": epoch,
                "payload": query_payload,
            }
        )
        ready = websocket.receive_json()
        assert ready["type"] == "plugin.ui.query.ready", ready
        assert len(json.dumps(ready).encode("utf-8")) < 2 * 1024

        response = client.post(
            ready["payload"]["path"],
            headers={
                "Authorization": f"Bearer {ready['payload']['ticket']}",
            },
            json=request_body,
        )
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert query_calls == [{"message_id": "message:446"}]
        assert response.json() == {
            "schema": "akasha.recall-card.v1",
            "plugin_id": "akasha@builtin",
            "plugin_revision": "revision-1",
            "method": "recall.current",
            "payload": {"message_id": "message:446"},
            "session_id": None,
            "turn_id": None,
        }

        tampered = client.post(
            ready["payload"]["path"],
            headers={
                "Authorization": f"Bearer {ready['payload']['ticket']}",
            },
            json={**request_body, "turn_id": "turn-other"},
        )
        assert tampered.status_code == 401
        assert tampered.json()["code"] == "invalid_ticket"
        assert len(query_calls) == 1

    disconnected = client.post(
        ready["payload"]["path"],
        headers={
            "Authorization": f"Bearer {ready['payload']['ticket']}",
        },
        json=request_body,
    )
    assert disconnected.status_code == 401
    assert disconnected.json()["code"] == "invalid_ticket"
    assert len(query_calls) == 1

    runtime.close()


def test_plugin_ui_https_ticket_expires_before_query_execution(
    tmp_path: Path,
) -> None:
    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    runtime, keyset = asyncio.run(build())
    device_id = uuid4().hex
    _register_test_device(runtime, device_id)
    current = [datetime(2026, 7, 28, tzinfo=timezone.utc)]
    issuer = PluginUiHttpTicketIssuer(
        keyset,
        runtime.storage,
        clock=lambda: current[0],
    )
    body: dict[str, object] = {
        "request_id": "01ARZ3NDEKTSV4RRFFQ69G5FB8",
        "owner_id": "owner",
    }
    grant = issuer.issue(
        device_id=device_id,
        connection_epoch=1,
        request_body=body,
    )

    runtime.storage.revoke_device(
        device_id,
        revoked_at=current[0],
    )
    with pytest.raises(PluginUiHttpTicketError, match="设备无效"):
        issuer.verify(grant.ticket, request_body=body)

    current[0] += timedelta(seconds=31)

    with pytest.raises(PluginUiHttpTicketError, match="已过期"):
        issuer.verify(grant.ticket, request_body=body)

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


def test_outbound_attachment_download_replays_binary_before_reply(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """验证出站附件只暴露描述符，并以可重复 offset 下载二进制。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    class CapturePushTool:
        def register_channel(self, channel: str, **senders: object) -> None:
            assert channel == "mobile"

    import asyncio

    runtime, _ = asyncio.run(build())
    request.addfinalizer(runtime.close)
    push = CapturePushTool()
    asyncio.run(
        runtime.channel.start(
            cast(
                Any,
                SimpleNamespace(
                    bus=SimpleNamespace(subscribe_outbound=lambda *_: None),
                    session_manager=SessionManager(tmp_path / "sessions"),
                    event_bus=SimpleNamespace(on=lambda *_: None),
                    push_tool=push,
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
            capabilities=("attachments-v1",),
        )
    )
    chat_id = str(uuid4())
    session_id = f"mobile:{chat_id}"
    runtime.storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    content = b"outbound" * 20_000
    source = tmp_path / "agent-result.bin"
    source.write_bytes(content)
    turn_id = uuid4().hex
    persisted = runtime.channel._require_ctx().session_manager.get_or_create(session_id)
    persisted.add_message(
        "assistant",
        "文件已生成",
        media=[str(source)],
        id=uuid4().hex,
    )
    runtime.channel._require_ctx().session_manager.save(persisted)
    asyncio.run(
        runtime.channel._on_turn_started(
            TurnStarted(
                session_key=session_id,
                channel="mobile",
                chat_id=chat_id,
                content="生成文件",
                timestamp=datetime.now(timezone.utc),
                turn_id=turn_id,
            )
        )
    )
    asyncio.run(
        runtime.channel._on_response(
            OutboundMessage(
                channel="mobile",
                chat_id=chat_id,
                content="文件已生成",
                media=[str(source)],
                control_turn_id=turn_id,
                session_message_id=str(persisted.messages[-1]["id"]),
            )
        )
    )

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
        assert websocket.receive_json()["type"] == "turn.started"
        final = websocket.receive_json()
        assert final["type"] == "message.final"
        descriptor = final["payload"]["attachments"][0]
        assert "local_path" not in descriptor
        assert websocket.receive_json()["type"] == "sync.completed"

        command = {
            "v": 1,
            "kind": "command",
            "type": "attachment.download",
            "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "connection_epoch": epoch,
            "session_id": session_id,
            "payload": {
                "attachment_id": descriptor["attachment_id"],
                "offset": 0,
            },
        }
        websocket.send_json(command)
        first = decode_attachment_chunk(websocket.receive_bytes())
        reply = websocket.receive_json()
        assert first.data == content[:MAX_ATTACHMENT_CHUNK_BYTES]
        assert reply["type"] == "attachment.download.ok"
        assert reply["payload"]["next_offset"] == len(first.data)

        websocket.send_json(command)
        duplicate = decode_attachment_chunk(websocket.receive_bytes())
        assert duplicate == first
        assert websocket.receive_json() == reply


def test_slow_device_delivery_does_not_block_other_device(tmp_path: Path) -> None:
    """验证慢设备只阻塞自身队列，其他设备仍能实时收到事件。"""

    async def scenario() -> None:
        # 1. 注册两个在线设备，其中一个 socket 人为阻塞
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        slow_id = uuid4().hex
        fast_id = uuid4().hex
        _register_test_device(runtime, slow_id)
        _register_test_device(runtime, fast_id)
        slow_gate = asyncio.Event()
        slow_socket = _ControlledWebSocket(send_gate=slow_gate)
        fast_socket = _ControlledWebSocket()
        slow_connection = ActiveMobileConnection(
            websocket=cast(Any, slow_socket),
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )
        fast_connection = ActiveMobileConnection(
            websocket=cast(Any, fast_socket),
            connection_epoch=2,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )
        runtime._connections[slow_id] = slow_connection
        runtime._connections[fast_id] = fast_connection

        # 2. fanout 返回后，快设备应在慢设备解除阻塞前完成写入
        await runtime.publish_event(
            event_type="connection.degraded",
            payload={"reason": "fanout-test"},
        )
        await asyncio.wait_for(slow_socket.send_started.wait(), timeout=1)
        await asyncio.wait_for(fast_socket.send_started.wait(), timeout=1)
        assert len(fast_socket.sent_text) == 1
        assert slow_socket.sent_text == []

        # 3. 解除慢设备后，其 durable 序号仍按顺序推进
        slow_task = slow_connection.delivery_task
        assert slow_task is not None
        slow_gate.set()
        await asyncio.wait_for(slow_task, timeout=1)
        assert len(slow_socket.sent_text) == 1
        assert runtime.storage.read_cursor(slow_id).sent_event_seq == 1
        assert runtime.storage.read_cursor(fast_id).sent_event_seq == 1
        runtime.close()

    asyncio.run(scenario())


def test_connection_control_only_reaches_matching_current_connection(tmp_path: Path) -> None:
    """验证临时控制帧不入箱，且只投递给匹配的当前 epoch。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        socket = _ControlledWebSocket()
        runtime._connections[device_id] = ActiveMobileConnection(
            websocket=cast(Any, socket),
            connection_epoch=2,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )

        await runtime.publish_connection_control(
            device_id=device_id,
            connection_epoch=1,
            control_type="plugin.ui.changed",
            payload={},
        )

        assert runtime.storage.read_cursor(device_id).next_event_seq == 1
        assert socket.sent_text == []

        await runtime.publish_connection_control(
            device_id=device_id,
            connection_epoch=2,
            control_type="plugin.ui.changed",
            payload={},
        )

        assert runtime.storage.read_cursor(device_id).next_event_seq == 1
        assert json.loads(socket.sent_text[0]) == {
            "connection_epoch": 2,
            "kind": "control",
            "payload": {},
            "type": "plugin.ui.changed",
            "v": 1,
        }
        runtime.close()

    asyncio.run(scenario())


def test_connection_control_timeout_only_removes_slow_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证控制帧写超时不会卡住调用方或污染 durable cursor。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        socket = _ControlledWebSocket(send_gate=asyncio.Event())
        runtime._connections[device_id] = ActiveMobileConnection(
            websocket=cast(Any, socket),
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )
        monkeypatch.setattr(
            gateway_module,
            "_CONNECTION_CONTROL_SEND_TIMEOUT_SECONDS",
            0.01,
        )

        await runtime.publish_connection_control(
            device_id=device_id,
            connection_epoch=1,
            control_type="plugin.ui.changed",
            payload={},
        )

        assert device_id not in runtime._connections
        await asyncio.wait_for(socket.close_started.wait(), timeout=1)
        assert socket.close_calls == [
            (4408, "连接控制帧投递失败，请重新连接恢复"),
        ]
        assert runtime.storage.read_cursor(device_id).next_event_seq == 1
        runtime.close()

    asyncio.run(scenario())


def test_connection_control_waits_for_inflight_frame_without_removing_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证正常在途帧占锁超过控制帧超时也不会被误判为慢连接。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        socket = _ControlledWebSocket()
        send_lock = asyncio.Lock()
        await send_lock.acquire()
        connection = ActiveMobileConnection(
            websocket=cast(Any, socket),
            connection_epoch=1,
            send_lock=send_lock,
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )
        runtime._connections[device_id] = connection
        monkeypatch.setattr(
            gateway_module,
            "_CONNECTION_CONTROL_SEND_TIMEOUT_SECONDS",
            0.01,
        )

        delivery = asyncio.create_task(
            runtime.publish_connection_control(
                device_id=device_id,
                connection_epoch=1,
                control_type="plugin.ui.changed",
                payload={},
            )
        )
        await asyncio.sleep(0.03)

        assert not delivery.done()
        assert runtime._connections[device_id] is connection
        assert socket.close_calls == []

        send_lock.release()
        await asyncio.wait_for(delivery, timeout=1)
        assert runtime._connections[device_id] is connection
        assert len(socket.sent_text) == 1
        runtime.close()

    asyncio.run(scenario())


def test_connection_control_lock_timeout_removes_stalled_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证长期持锁的失活连接不会永久阻塞插件目录刷新。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        socket = _ControlledWebSocket()
        send_lock = asyncio.Lock()
        await send_lock.acquire()
        runtime._connections[device_id] = ActiveMobileConnection(
            websocket=cast(Any, socket),
            connection_epoch=1,
            send_lock=send_lock,
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )
        monkeypatch.setattr(
            gateway_module,
            "_CONNECTION_CONTROL_LOCK_TIMEOUT_SECONDS",
            0.01,
        )

        await runtime.publish_connection_control(
            device_id=device_id,
            connection_epoch=1,
            control_type="plugin.ui.changed",
            payload={},
        )

        assert device_id not in runtime._connections
        assert socket.sent_text == []
        await asyncio.wait_for(socket.close_started.wait(), timeout=1)
        assert socket.close_calls == [
            (4408, "连接控制帧投递失败，请重新连接恢复"),
        ]
        assert send_lock.locked()
        send_lock.release()
        runtime.close()

    asyncio.run(scenario())


def test_resume_window_queues_concurrent_event_after_sync_terminal(
    tmp_path: Path,
) -> None:
    """验证 resume 阻塞期间发布的事件不会漏发、重发或越过终止帧。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        await runtime.publish_event(
            device_id=device_id,
            event_type="connection.degraded",
            payload={"reason": "before-resume"},
        )
        send_gate = asyncio.Event()
        websocket = _ControlledWebSocket(send_gate=send_gate)

        # 1. 卡住历史事件写入，并在 resume 窗口发布实时事件
        resume_task = asyncio.create_task(
            runtime._resume_and_register(
                cast(Any, websocket),
                device_id=device_id,
                connection_epoch=1,
                last_ack=0,
            )
        )
        await asyncio.wait_for(websocket.send_started.wait(), timeout=1)
        await runtime.publish_event(
            device_id=device_id,
            event_type="connection.degraded",
            payload={"reason": "during-resume"},
        )
        send_gate.set()
        await asyncio.wait_for(resume_task, timeout=1)

        # 2. 等待独立投递任务排空重放期间的实时事件
        connection = runtime._connections[device_id]
        delivery_task = connection.delivery_task
        if delivery_task is not None:
            await asyncio.wait_for(delivery_task, timeout=1)
        frames = [json.loads(text) for text in websocket.sent_text]
        assert [frame["type"] for frame in frames] == [
            "connection.degraded",
            "sync.completed",
            "connection.degraded",
        ]
        assert frames[0]["payload"]["reason"] == "before-resume"
        assert frames[2]["payload"]["reason"] == "during-resume"
        assert [frame["event_seq"] for frame in frames] == [1, 2, 3]
        assert runtime.storage.read_cursor(device_id).sent_event_seq == 3
        runtime.close()

    asyncio.run(scenario())


def test_replaced_socket_close_does_not_block_other_device(
    tmp_path: Path,
) -> None:
    """验证旧连接关闭卡住时，不会占用其他设备的投递路径。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        replaced_id = uuid4().hex
        other_id = uuid4().hex
        _register_test_device(runtime, replaced_id)
        _register_test_device(runtime, other_id)
        close_gate = asyncio.Event()
        old_socket = _ControlledWebSocket(close_gate=close_gate)
        runtime._connections[replaced_id] = ActiveMobileConnection(
            websocket=cast(Any, old_socket),
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )
        other_socket = _ControlledWebSocket()
        runtime._connections[other_id] = ActiveMobileConnection(
            websocket=cast(Any, other_socket),
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )

        # 1. 新连接完成 resume，但旧 socket 的 close 持续阻塞
        new_socket = _ControlledWebSocket()
        await runtime._resume_and_register(
            cast(Any, new_socket),
            device_id=replaced_id,
            connection_epoch=2,
            last_ack=0,
        )
        await asyncio.wait_for(old_socket.close_started.wait(), timeout=1)

        # 2. 另一设备仍能收到实时事件
        await runtime.publish_event(
            device_id=other_id,
            event_type="connection.degraded",
            payload={"reason": "other-device"},
        )
        await asyncio.wait_for(other_socket.send_started.wait(), timeout=1)
        assert len(other_socket.sent_text) == 1
        close_gate.set()
        await asyncio.sleep(0)
        runtime.close()

    asyncio.run(scenario())


def test_binary_reply_is_atomic_against_event_delivery(tmp_path: Path) -> None:
    """验证下载二进制与 reply 之间不会插入实时事件。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        bytes_gate = asyncio.Event()
        websocket = _ControlledWebSocket(bytes_gate=bytes_gate)
        connection = ActiveMobileConnection(
            websocket=cast(Any, websocket),
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )
        runtime._connections[device_id] = connection

        class BinaryReplyChannel:
            async def handle_command(self, **_: object) -> object:
                return SimpleNamespace(
                    binary=AttachmentChunk("01ARZ3NDEKTSV4RRFFQ69G5FAV", 0, b"data"),
                    type="attachment.download.ok",
                    payload={"next_offset": 4},
                    session_id=None,
                    turn_id=None,
                )

        runtime._channel = cast(Any, BinaryReplyChannel())
        frame = parse_frame(
            json.dumps(
                {
                    "v": 1,
                    "kind": "command",
                    "type": "attachment.download",
                    "id": "01ARZ3NDEKTSV4RRFFQ69G5FAW",
                    "connection_epoch": 1,
                    "session_id": f"mobile:{uuid4()}",
                    "payload": {
                        "attachment_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                        "offset": 0,
                    },
                }
            )
        )
        assert isinstance(frame, AttachmentDownloadCommand)

        # 1. 二进制写入持锁期间发布事件，事件任务只能等待
        command_task = asyncio.create_task(
            runtime._handle_command(
                cast(Any, websocket),
                frame,
                connection_epoch=1,
                device_id=device_id,
            )
        )
        await asyncio.wait_for(websocket.bytes_started.wait(), timeout=1)
        await runtime.publish_event(
            device_id=device_id,
            event_type="connection.degraded",
            payload={"reason": "atomic-order"},
        )
        await asyncio.sleep(0)
        assert websocket.wire_order == ["bytes"]

        # 2. 解锁后必须先 reply，再发送排队事件
        bytes_gate.set()
        await asyncio.wait_for(command_task, timeout=1)
        for _ in range(20):
            if len(websocket.wire_order) == 3:
                break
            await asyncio.sleep(0)
        assert websocket.wire_order == ["bytes", "reply", "event"]
        runtime.close()

    asyncio.run(scenario())


def test_slow_connection_queue_overflow_keeps_durable_events(
    tmp_path: Path,
) -> None:
    """验证实时队列超限会断开慢连接，但 durable inbox 完整保留。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        websocket = _ControlledWebSocket()
        connection = ActiveMobileConnection(
            websocket=cast(Any, websocket),
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=False,
            delivery_task=None,
        )
        runtime._connections[device_id] = connection

        # 1. resume 尚未 ready 时持续排队，第 65 个事件触发慢消费者降级
        for index in range(65):
            await runtime.publish_event(
                device_id=device_id,
                event_type="connection.degraded",
                payload={"reason": f"queued-{index}"},
            )
        await asyncio.wait_for(websocket.close_started.wait(), timeout=1)
        assert device_id not in runtime._connections

        # 2. 网络队列被丢弃，但 65 个事件仍可从 durable inbox 恢复
        durable = runtime.storage.read_durable_events(
            device_id,
            after_event_seq=0,
            limit=100,
        )
        assert len(durable) == 65
        assert [event.event_seq for event in durable] == list(range(1, 66))
        assert runtime.storage.read_cursor(device_id).sent_event_seq == 0
        runtime.close()

    asyncio.run(scenario())


def test_command_reply_stops_at_causal_event_barrier(tmp_path: Path) -> None:
    """验证后续高频事件不会让当前命令 reply 等到整队列排空。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        send_gate = asyncio.Event()
        websocket = _ControlledWebSocket(send_gate=send_gate)
        connection = ActiveMobileConnection(
            websocket=cast(Any, websocket),
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )
        runtime._connections[device_id] = connection

        class CausalEventChannel:
            async def handle_command(self, **_: object) -> object:
                await runtime.publish_event(
                    device_id=device_id,
                    event_type="connection.degraded",
                    payload={"reason": "causal"},
                )
                return SimpleNamespace(
                    binary=None,
                    type="session.list.ok",
                    payload={"sessions": []},
                    session_id=None,
                    turn_id=None,
                )

        runtime._channel = cast(Any, CausalEventChannel())
        frame = parse_frame(
            json.dumps(
                {
                    "v": 1,
                    "kind": "command",
                    "type": "session.list",
                    "id": "01ARZ3NDEKTSV4RRFFQ69G5FAX",
                    "connection_epoch": 1,
                    "payload": {},
                }
            )
        )

        # 1. 命令因果事件卡在 socket，同时追加一批后续事件
        command_task = asyncio.create_task(
            runtime._handle_command(
                cast(Any, websocket),
                cast(Any, frame),
                connection_epoch=1,
                device_id=device_id,
            )
        )
        await asyncio.wait_for(websocket.send_started.wait(), timeout=1)
        for index in range(20):
            await runtime.publish_event(
                device_id=device_id,
                event_type="connection.degraded",
                payload={"reason": f"later-{index}"},
            )

        # 2. 放行后 reply 紧跟因果事件，不等待后续 20 个事件排空
        send_gate.set()
        await asyncio.wait_for(command_task, timeout=1)
        assert websocket.wire_order[:2] == ["event", "reply"]
        for _ in range(100):
            if len(websocket.wire_order) == 22:
                break
            await asyncio.sleep(0)
        assert websocket.wire_order == ["event", "reply"] + ["event"] * 20
        runtime.close()

    asyncio.run(scenario())


def test_resume_pages_only_to_frozen_high_watermark(tmp_path: Path) -> None:
    """验证大于单页的 backlog 分页重放，并在冻结上限后发送 terminal。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        for index in range(600):
            await runtime.publish_event(
                device_id=device_id,
                event_type="connection.degraded",
                payload={"reason": f"backlog-{index}"},
            )
        websocket = _ControlledWebSocket()

        await runtime._resume_and_register(
            cast(Any, websocket),
            device_id=device_id,
            connection_epoch=1,
            last_ack=0,
        )
        frames = [json.loads(text) for text in websocket.sent_text]
        assert len(frames) == 601
        assert frames[-1]["type"] == "sync.completed"
        assert frames[-1]["payload"]["replayed_events"] == 600
        assert [frame["event_seq"] for frame in frames] == list(range(1, 602))
        runtime.close()

    asyncio.run(scenario())


def test_replaced_connection_ack_cannot_delete_new_resume_window(
    tmp_path: Path,
) -> None:
    """验证旧 epoch 的迟到 ACK 不能删除新连接需要重放的 durable 前缀。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        await runtime.publish_event(
            device_id=device_id,
            event_type="connection.degraded",
            payload={"reason": "must-replay"},
        )
        _ = runtime.inbox.mark_sent(device_id, through_event_seq=1)
        old_socket = _ControlledWebSocket()
        new_socket = _ControlledWebSocket()
        runtime._connections[device_id] = ActiveMobileConnection(
            websocket=cast(Any, new_socket),
            connection_epoch=2,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=False,
            delivery_task=None,
        )

        # 1. 新连接已占住活动代际后，旧连接 ACK 必须被拒绝
        acknowledged = await runtime._acknowledge_active_connection(
            device_id=device_id,
            websocket=cast(Any, old_socket),
            connection_epoch=1,
            through_event_seq=1,
        )
        assert acknowledged is False
        assert runtime.storage.read_cursor(device_id).acknowledged_event_seq == 0
        assert len(
            runtime.storage.read_durable_events(
                device_id,
                after_event_seq=0,
                limit=10,
            )
        ) == 1

        # 2. 只有当前 websocket 与 epoch 的 ACK 可以删除该前缀
        acknowledged = await runtime._acknowledge_active_connection(
            device_id=device_id,
            websocket=cast(Any, new_socket),
            connection_epoch=2,
            through_event_seq=1,
        )
        assert acknowledged is True
        assert runtime.storage.read_cursor(device_id).acknowledged_event_seq == 1
        assert runtime.storage.read_durable_events(
            device_id,
            after_event_seq=0,
            limit=10,
        ) == ()
        runtime.close()

    asyncio.run(scenario())

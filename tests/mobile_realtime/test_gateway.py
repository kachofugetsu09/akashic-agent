from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import sqlite3
from collections import deque
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import infra.mobile_realtime.gateway as gateway_module
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent.config_models import MobileKeyEncryptionConfig, MobileRealtimeConfig
from agent.plugins.mobile_ui import MobileUiRpcExecutionError
from agent.plugin_composition.channels import (
    ChannelFactoryContext,
    ChannelInboundMessage,
    ChannelRuntimePorts,
    RawInbound,
)
from bus.events import OutboundMessage, channel_message_from_outbound
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from infra.channels.base import AttachmentStore
from infra.channels.artifacts import ChannelAttachmentArtifactStore
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
from infra.mobile_realtime.protocol import (
    AttachmentDownloadCommand,
    TURN_OUTPUT_COMPLETED_CAPABILITY,
    parse_frame,
)
from infra.mobile_realtime.storage import DeviceRecord, MobileStorageError
from session.manager import SessionManager


async def _attach_open_mobile_v3(
    channel: Any,
    ingress: Any,
    *,
    binding_token: str,
) -> Any:
    """Attach one exact v3 ingress and open admission for this fixture."""

    context = ChannelFactoryContext(
        snapshot_id="gateway-test-snapshot",
        generation_id="gateway-test-generation",
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


@pytest.mark.asyncio
async def test_gateway_publish_event_reports_zero_after_device_race(
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
            capabilities=(),
        )
    )
    original_list_active_devices = runtime.storage.list_active_devices
    calls = 0

    def list_active_devices_with_race() -> tuple[DeviceRecord, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_list_active_devices()
        return ()

    runtime.storage.list_active_devices = (  # type: ignore[method-assign]
        list_active_devices_with_race
    )
    try:
        assert runtime.storage.list_active_devices()
        recipient_count = await runtime.publish_event(
            event_type="session.updated",
            session_id="akashic:race",
            payload={
                "session_id": "akashic:race",
                "message_id": "akashic:race:0",
                "head_seq": 0,
            },
        )
        assert recipient_count == 0
        assert calls == 2
        assert runtime.storage.count_durable_events(device_id) == 0
    finally:
        runtime.close()


class _ControlledWebSocket:
    def __init__(
        self,
        *,
        send_gate: asyncio.Event | None = None,
        close_gate: asyncio.Event | None = None,
        bytes_gate: asyncio.Event | None = None,
        fail_send: bool = False,
    ) -> None:
        self.send_gate = send_gate
        self.close_gate = close_gate
        self.bytes_gate = bytes_gate
        self.fail_send = fail_send
        self.receive_hang = asyncio.Event()
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
        if self.fail_send:
            raise RuntimeError("socket send failed")
        self.sent_text.append(text)
        self.wire_order.append(str(json.loads(text)["kind"]))

    async def receive_text(self) -> str:
        # 模拟断了 send 但客户端 receive 仍挂起的旧 socket
        await self.receive_hang.wait()
        raise RuntimeError("socket closed by peer")

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


@pytest.mark.asyncio
async def test_gateway_file_provider_persists_identity_across_restart(
    tmp_path: Path,
) -> None:
    config = MobileRealtimeConfig(
        enabled=True,
        database=Path("data/mobile.db"),
        lan_hostname="akashic.local",
        public_url="wss://agent.example.com/ws",
        key_encryption=MobileKeyEncryptionConfig(
            provider="file",
            master_key_file=Path("data/mobile/master-keys.json"),
            keyset_manifest=Path("data/mobile/keys/current.json"),
        ),
    )

    first_runtime, first_keyset = build_mobile_gateway_runtime(config, tmp_path)
    first_runtime.close()
    second_runtime, second_keyset = build_mobile_gateway_runtime(config, tmp_path)
    try:
        assert second_keyset.server_fingerprint == first_keyset.server_fingerprint
        assert (tmp_path / "data/mobile/master-keys.json").is_file()
    finally:
        second_runtime.close()


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
    stale_turn_id = "01ARZ3NDEKTSV4RRFFQ69G5FAQ"

    async def reconcile(*, device_id: str, active_turns: tuple[str, ...]) -> None:
        assert active_turns == (stale_turn_id,)
        await runtime.publish_event(
            device_id=device_id,
            event_type="turn.interrupted",
            turn_id=stale_turn_id,
            payload={"status": "cancelled"},
        )

    runtime.channel.reconcile_active_turns = AsyncMock(side_effect=reconcile)
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
                "payload": {"last_ack": 0, "active_turns": [stale_turn_id]},
            }
        )
        terminal = websocket.receive_json()
        assert terminal["type"] == "turn.interrupted"
        assert terminal["turn_id"] == stale_turn_id
        assert terminal["event_seq"] == 1
        synced = websocket.receive_json()
        assert synced["type"] == "sync.completed"
        assert synced["event_seq"] == 2
        websocket.send_json(
            {
                "v": 1,
                "kind": "ack",
                "type": "event.ack",
                "connection_epoch": epoch,
                "payload": {"through_event_seq": 2},
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
    assert cursor.acknowledged_event_seq == 2
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
        assert [
            event.event_seq
            for event in runtime.storage.read_durable_events(
                device_id,
                after_event_seq=5,
                limit=10,
            )
        ] == [6]
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
        assert [
            event.event_seq
            for event in runtime.storage.read_durable_events(
                device_id,
                after_event_seq=5,
                limit=10,
            )
        ] == [6]
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
    session_id = f"akashic:{uuid4()}"

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


def test_offline_session_update_is_durable_and_replayed(
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
    session_id = f"akashic:{chat_id}"
    runtime.storage.claim_session(
        device_id=device_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
    )
    asyncio.run(
        runtime.publish_event(
            event_type="session.updated",
            session_id=session_id,
            payload={
                "session_id": session_id,
                "message_id": f"{session_id}:7",
                "head_seq": 7,
            },
        )
    )
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
        updated = websocket.receive_json()
        synced = websocket.receive_json()

    assert updated["type"] == "session.updated"
    assert updated["session_id"] == session_id
    assert updated["payload"]["head_seq"] == 7
    assert synced["type"] == "sync.completed"
    runtime.close()


def test_publish_event_respects_required_capability(tmp_path: Path) -> None:
    """output.completed 只入箱声明了能力的设备，旧客户端不收到该事件。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    import asyncio

    runtime, _ = asyncio.run(build())
    capable_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=capable_id,
            public_key=_device_public_key(ec.generate_private_key(ec.SECP256R1())),
            display_name="New Client",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1", TURN_OUTPUT_COMPLETED_CAPABILITY),
        )
    )
    legacy_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=legacy_id,
            public_key=_device_public_key(ec.generate_private_key(ec.SECP256R1())),
            display_name="Legacy Client",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )
    asyncio.run(
        runtime.publish_event(
            event_type="turn.output.completed",
            payload={"client_message_id": "cmid-1"},
            session_id="akashic:abc",
            turn_id="turn-1",
            required_capability=TURN_OUTPUT_COMPLETED_CAPABILITY,
        )
    )
    assert runtime.storage.count_durable_events(capable_id) == 1
    assert runtime.storage.count_durable_events(legacy_id) == 0
    runtime.close()


def test_device_update_refreshes_capabilities_and_unlocks_event(
    tmp_path: Path,
) -> None:
    """已配对旧客户端升级后 device.update 刷新能力，无需重新配对即可收到新事件。"""

    async def build():
        return build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )

    import asyncio

    runtime, _ = asyncio.run(build())
    device_id = uuid4().hex
    runtime.storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=_device_public_key(ec.generate_private_key(ec.SECP256R1())),
            display_name="Upgraded Client",
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )
    # 升级前：旧能力收不到 output.completed
    asyncio.run(
        runtime.publish_event(
            event_type="turn.output.completed",
            payload={"client_message_id": "cmid-0"},
            session_id="akashic:abc",
            turn_id="turn-0",
            required_capability=TURN_OUTPUT_COMPLETED_CAPABILITY,
        )
    )
    assert runtime.storage.count_durable_events(device_id) == 0

    # device.update 刷新能力声明
    asyncio.run(
        runtime.refresh_device_capabilities(
            device_id=device_id,
            capabilities=("stream-v1", TURN_OUTPUT_COMPLETED_CAPABILITY),
        )
    )
    assert runtime.storage.read_device(device_id).capabilities == (
        "stream-v1",
        TURN_OUTPUT_COMPLETED_CAPABILITY,
    )

    # 升级后：新能力收到 output.completed
    asyncio.run(
        runtime.publish_event(
            event_type="turn.output.completed",
            payload={"client_message_id": "cmid-1"},
            session_id="akashic:abc",
            turn_id="turn-1",
            required_capability=TURN_OUTPUT_COMPLETED_CAPABILITY,
        )
    )
    assert runtime.storage.count_durable_events(device_id) == 1
    runtime.close()


def test_authenticated_message_send_reaches_agent_event_path_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
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
            self.inbound: list[RawInbound] = []
            self.legacy_publish_calls = 0

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
            assert isinstance(raw, RawInbound)
            inbound = raw.message
            assert isinstance(inbound, ChannelInboundMessage)
            self.inbound.append(raw)
            turn_id = uuid4().hex
            await runtime.channel._on_turn_started(
                TurnStarted(
                    session_key=cast(str, inbound.metadata["session_key_override"]),
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    content=inbound.content,
                    timestamp=inbound.timestamp,
                    turn_id=turn_id,
                )
            )
            await runtime.channel._on_stream_delta(
                StreamDeltaReady(
                    session_key=cast(str, inbound.metadata["session_key_override"]),
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    turn_id=turn_id,
                    thinking_delta="先检查",
                )
            )
            await runtime.channel._on_tool_call_started(
                ToolCallStarted(
                    session_key=cast(str, inbound.metadata["session_key_override"]),
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    iteration=1,
                    call_id="call-1",
                    tool_name="shell",
                    arguments={"command": "pwd"},
                    turn_id=turn_id,
                )
            )
            await runtime.channel._on_tool_call_completed(
                ToolCallCompleted(
                    session_key=cast(str, inbound.metadata["session_key_override"]),
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
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
                    session_key=cast(str, inbound.metadata["session_key_override"]),
                    channel=inbound.channel,
                    chat_id=inbound.chat_id,
                    turn_id=turn_id,
                    thinking_delta="工具后继续",
                )
            )
            receipt = await runtime.channel._deliver_message(
                channel_message_from_outbound(
                        OutboundMessage(
                            channel="akashic",
                            chat_id=inbound.chat_id,
                            content="完成",
                            thinking="先检查",
                            control_turn_id=turn_id,
                            execution_attempt_id=turn_id,
                            metadata={"_channel_commit_role": "passive"},
                    )
                )
            )
            assert receipt.succeeded
            return True

    class FakeEventBus:
        def on(self, event_type: type[object], callback: object) -> None:
            return None

    class FakePushTool:
        pass

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
    adapter = asyncio.run(
        _attach_open_mobile_v3(
            runtime.channel,
            bus,
            binding_token="gateway-event-fixture",
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
    session_id = f"akashic:{uuid4()}"
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

    caplog.set_level(logging.INFO, logger="infra.mobile_realtime.gateway")
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
    assert bus.legacy_publish_calls == 0
    reply_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event") == "tl:send.reply_sent"
        and record.akashic_fields.get("client_message_id") == command_id
    ]
    assert len(reply_records) == 2
    assert [record.akashic_fields["outcome"] for record in reply_records] == [
        "sent",
        "receipt_replayed",
    ]
    assert [record.akashic_fields["receipt_replayed"] for record in reply_records] == [
        False,
        True,
    ]
    assert all(
        record.akashic_fields["device_id"] == device_id for record in reply_records
    )
    assert all(
        record.akashic_fields["connection_epoch"] == epoch for record in reply_records
    )
    assert all(
        record.akashic_fields["reply_type"] == "message.send.ok"
        for record in reply_records
    )
    asyncio.run(adapter.stop())
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
            self.inbound: list[RawInbound] = []
            self.legacy_publish_calls = 0

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
            assert isinstance(raw, RawInbound)
            assert isinstance(raw.message, ChannelInboundMessage)
            self.inbound.append(raw)
            return True

    class FakeEventBus:
        def on(self, event_type: type[object], callback: object) -> None:
            return None

    class FakePushTool:
        pass

    import asyncio

    runtime, _ = asyncio.run(build())
    request.addfinalizer(runtime.close)
    bus = CaptureBus()
    session_manager = SessionManager(tmp_path / "sessions")
    runtime.channel.bind_channel_attachment_store(
        ChannelAttachmentArtifactStore(
            workspace=session_manager.workspace,
            session_store=session_manager.control_store,
        )
    )
    asyncio.run(
        runtime.channel.start(
            cast(
                Any,
                SimpleNamespace(
                    bus=bus,
                    session_manager=session_manager,
                    event_bus=FakeEventBus(),
                    push_tool=FakePushTool(),
                    interrupt_controller=None,
                    attachment_store=AttachmentStore(tmp_path / "uploads"),
                ),
            )
        )
    )
    adapter = asyncio.run(
        _attach_open_mobile_v3(
            runtime.channel,
            bus,
            binding_token="gateway-attachment-fixture",
        )
    )
    request.addfinalizer(lambda: asyncio.run(runtime.channel.stop()))
    request.addfinalizer(lambda: asyncio.run(adapter.stop()))
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
    session_id = f"akashic:{uuid4()}"
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

    assert len(bus.inbound) == 1
    assert bus.legacy_publish_calls == 0
    raw = bus.inbound[0]
    assert raw.message.content == ""
    artifact_ids = cast(tuple[str, ...], raw.message.metadata["attachment_ids"])
    assert len(artifact_ids) == 1
    assert tuple(ref.artifact_id for ref in raw.message.attachments) == artifact_ids
    artifact = session_manager.control_store.get_attachment(artifact_ids[0])
    assert artifact is not None
    assert artifact.state == "ready"
    assert artifact.size_bytes == len(content)
    assert (session_manager.workspace / artifact.storage_key).read_bytes() == content


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
        pass

    import asyncio

    runtime, _ = asyncio.run(build())
    request.addfinalizer(runtime.close)
    push = CapturePushTool()
    asyncio.run(
        runtime.channel.start(
            cast(
                Any,
                SimpleNamespace(
                    bus=SimpleNamespace(),
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
    session_id = f"akashic:{chat_id}"
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
                channel="akashic",
                chat_id=chat_id,
                content="生成文件",
                timestamp=datetime.now(timezone.utc),
                turn_id=turn_id,
            )
        )
    )
    asyncio.run(
        runtime.channel._deliver_message(
            channel_message_from_outbound(
                OutboundMessage(
                    channel="akashic",
                    chat_id=chat_id,
                    content="文件已生成",
                        media=[str(source)],
                        control_turn_id=turn_id,
                        execution_attempt_id=turn_id,
                        session_message_id=str(persisted.messages[-1]["id"]),
                    metadata={"_channel_commit_role": "passive"},
                )
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


def test_live_drain_sends_plain_and_terminal_events_with_identity(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证在线排空普通与终态事件：均写入 socket、cursor 推进、身份日志不崩。"""

    async def scenario() -> str:
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
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )

        # 1. 普通事件经真实 _drain_connection 写入并推进 sent cursor
        await runtime.publish_event(
            device_id=device_id,
            event_type="connection.degraded",
            payload={"reason": "plain-live"},
        )
        plain_task = runtime._connections[device_id].delivery_task
        assert plain_task is not None
        await asyncio.wait_for(plain_task, timeout=5)
        assert len(socket.sent_text) == 1
        assert runtime.storage.read_cursor(device_id).sent_event_seq == 1

        # 2. 终态事件带 session/turn，identity 观测路径正常执行
        await runtime.publish_event(
            device_id=device_id,
            event_type="message.final",
            payload={"content": "done"},
            session_id="akashic:s1",
            turn_id="akashic:t1",
        )
        terminal_task = runtime._connections[device_id].delivery_task
        assert terminal_task is not None
        await asyncio.wait_for(terminal_task, timeout=5)
        frames = [json.loads(text) for text in socket.sent_text]
        assert [frame["type"] for frame in frames] == [
            "connection.degraded",
            "message.final",
        ]
        assert runtime.storage.read_cursor(device_id).sent_event_seq == 2
        runtime.close()
        return device_id

    caplog.set_level(logging.INFO, logger="infra.mobile_realtime.gateway")
    registered_device_id = asyncio.run(scenario())
    sent_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event") == "tl:event.sent"
    ]
    assert len(sent_records) == 1
    assert sent_records[0].akashic_fields["session_id"] == "akashic:s1"
    assert sent_records[0].akashic_fields["turn_id"] == "akashic:t1"
    assert sent_records[0].akashic_fields["client_message_id"] == ""
    assert sent_records[0].akashic_fields["counts"] == (
        f"event_type=message.final device_id={registered_device_id} "
        f"event_seq=2 connection_epoch=1"
    )
    queued_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event") == "tl:event.queued"
    ]
    assert len(queued_records) == 1
    assert queued_records[0].akashic_fields["session_id"] == "akashic:s1"
    assert queued_records[0].akashic_fields["turn_id"] == "akashic:t1"
    assert queued_records[0].akashic_fields["counts"] == (
        f"event_type=message.final device_id={registered_device_id} event_seq=2"
    )


def test_broken_socket_send_failure_closes_socket_and_resume_replays_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证 send 失败摘除连接后主动 close 旧 socket、cursor 不推进、不记 sent，
    resume 恰好重放一次终态事件，并在新 epoch 记一次 sent。"""

    async def scenario() -> str:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        broken_socket = _ControlledWebSocket(fail_send=True)
        runtime._connections[device_id] = ActiveMobileConnection(
            websocket=cast(Any, broken_socket),
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )

        # 1. 在线投递终态事件时 send_text 失败（客户端 receive 仍挂起）
        await runtime.publish_event(
            device_id=device_id,
            event_type="message.final",
            payload={"content": "done", "client_message_id": "cmid-t"},
            session_id="akashic:s1",
            turn_id="akashic:t1",
        )
        await asyncio.wait_for(broken_socket.send_started.wait(), timeout=5)

        # 2. 等待 drain 摘除当前连接、并主动 close 旧 socket，cursor 未推进
        for _ in range(500):
            if device_id not in runtime._connections:
                break
            await asyncio.sleep(0.01)
        assert device_id not in runtime._connections
        await asyncio.wait_for(broken_socket.close_started.wait(), timeout=5)
        assert broken_socket.close_calls == [
            (4408, "连接投递失败，请重新连接恢复"),
        ]
        assert broken_socket.sent_text == []
        cursor = runtime.storage.read_cursor(device_id)
        assert cursor.sent_event_seq == 0
        assert cursor.next_event_seq == 2

        # 3. 新连接 resume 后同一 durable 终态事件恰好重放一次并推进 cursor
        new_socket = _ControlledWebSocket()
        await runtime._resume_and_register(
            cast(Any, new_socket),
            device_id=device_id,
            connection_epoch=2,
            last_ack=0,
        )
        frames = [json.loads(text) for text in new_socket.sent_text]
        assert [frame["type"] for frame in frames] == [
            "message.final",
            "sync.completed",
        ]
        assert [frame["event_seq"] for frame in frames] == [1, 2]
        assert runtime.storage.read_cursor(device_id).sent_event_seq == 2
        runtime.close()
        return device_id

    caplog.set_level(logging.INFO, logger="infra.mobile_realtime.gateway")
    registered_device_id = asyncio.run(scenario())
    sent_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event") == "tl:event.sent"
    ]
    # 失败 epoch 绝不记 sent；新 epoch 重放同 seq 成功只记一次。
    assert len(sent_records) == 1
    assert sent_records[0].akashic_fields["session_id"] == "akashic:s1"
    assert sent_records[0].akashic_fields["turn_id"] == "akashic:t1"
    assert sent_records[0].akashic_fields["client_message_id"] == "cmid-t"
    assert sent_records[0].akashic_fields["counts"] == (
        f"event_type=message.final device_id={registered_device_id} "
        f"event_seq=1 connection_epoch=2"
    )


def test_replaced_epoch_during_send_records_no_sent_and_replay_records_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """旧连接 send_text 阻塞期间被新 epoch 替换：旧 owner 未推进 cursor 绝不记
    sent，新 epoch resume 重放同 seq 后只记一条 sent。"""

    async def scenario() -> str:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        old_socket = _ControlledWebSocket(send_gate=asyncio.Event())
        runtime._connections[device_id] = ActiveMobileConnection(
            websocket=cast(Any, old_socket),
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )

        # 1. 在线投递终态事件：旧 epoch 的 drain 阻塞在真实 send_text 写锁内
        await runtime.publish_event(
            device_id=device_id,
            event_type="message.final",
            payload={"content": "done", "client_message_id": "cmid-replace"},
            session_id="akashic:s1",
            turn_id="akashic:t1",
        )
        await asyncio.wait_for(old_socket.send_started.wait(), timeout=5)
        old_task = runtime._connections[device_id].delivery_task
        assert old_task is not None

        # 2. 阻塞期间新 epoch 经真实 _resume_and_register 替换旧连接
        new_socket = _ControlledWebSocket()
        await runtime._resume_and_register(
            cast(Any, new_socket),
            device_id=device_id,
            connection_epoch=2,
            last_ack=0,
        )
        assert runtime._connections[device_id].connection_epoch == 2

        # 3. 释放旧 gate：旧 send 物理完成但 cursor 不推进、不记 sent
        old_socket.send_gate.set()
        await asyncio.wait_for(old_task, timeout=5)
        cursor = runtime.storage.read_cursor(device_id)
        assert cursor.sent_event_seq == 2

        # 4. 新 epoch 重放同 seq 的 message.final 与 sync.completed
        frames = [json.loads(text) for text in new_socket.sent_text]
        finals = [frame for frame in frames if frame["type"] == "message.final"]
        assert [frame["event_seq"] for frame in finals] == [1]
        runtime.close()
        return device_id

    caplog.set_level(logging.INFO, logger="infra.mobile_realtime.gateway")
    registered_device_id = asyncio.run(scenario())
    sent_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event") == "tl:event.sent"
    ]
    # 旧 epoch 不记 sent；新 epoch 重放同 seq 只记一条。
    assert len(sent_records) == 1
    assert sent_records[0].akashic_fields["session_id"] == "akashic:s1"
    assert sent_records[0].akashic_fields["turn_id"] == "akashic:t1"
    assert sent_records[0].akashic_fields["client_message_id"] == "cmid-replace"
    assert sent_records[0].akashic_fields["counts"] == (
        f"event_type=message.final device_id={registered_device_id} "
        f"event_seq=1 connection_epoch=2"
    )


def test_terminal_queued_records_zero_device_without_false_report(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """无活动设备时终态事件不入任何 inbox，queued 绝不虚报。"""

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        await runtime.publish_event(
            event_type="message.final",
            payload={"content": "done"},
            session_id="akashic:s1",
            turn_id="akashic:t1",
        )
        assert runtime.storage.list_active_devices() == ()
        runtime.close()

    caplog.set_level(logging.INFO, logger="infra.mobile_realtime.gateway")
    asyncio.run(scenario())
    queued_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event") == "tl:event.queued"
    ]
    assert queued_records == []


def test_terminal_queued_records_one_milestone_per_device(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """终态事件按 enqueue_many 真实返回的设备副本逐设备记录 queued。"""

    async def scenario() -> tuple[str, str]:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_a = uuid4().hex
        device_b = uuid4().hex
        _register_test_device(runtime, device_a)
        _register_test_device(runtime, device_b)
        await runtime.publish_event(
            event_type="turn.interrupted",
            payload={"status": "interrupted", "reason": "test"},
            session_id="akashic:s1",
            turn_id="akashic:t1",
        )
        runtime.close()
        return device_a, device_b

    caplog.set_level(logging.INFO, logger="infra.mobile_realtime.gateway")
    device_a, device_b = asyncio.run(scenario())
    queued_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event") == "tl:event.queued"
    ]
    assert len(queued_records) == 2
    assert {record.akashic_fields["counts"] for record in queued_records} == {
        f"event_type=turn.interrupted device_id={device_a} event_seq=1",
        f"event_type=turn.interrupted device_id={device_b} event_seq=1",
    }
    for record in queued_records:
        assert record.akashic_fields["session_id"] == "akashic:s1"
        assert record.akashic_fields["turn_id"] == "akashic:t1"


def test_terminal_without_identity_still_advances_cursor_and_delivery_task(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """缺 session/turn/client 身份时 sent 观测缺省为 missing，不破坏投递任务与 cursor。"""

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
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )
        await runtime.publish_event(
            device_id=device_id,
            event_type="message.final",
            payload={"content": "done"},
        )
        terminal_task = runtime._connections[device_id].delivery_task
        assert terminal_task is not None
        await asyncio.wait_for(terminal_task, timeout=5)
        assert runtime.storage.read_cursor(device_id).sent_event_seq == 1
        runtime.close()

    caplog.set_level(logging.INFO, logger="infra.mobile_realtime.gateway")
    asyncio.run(scenario())
    sent_records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event") == "tl:event.sent"
    ]
    assert len(sent_records) == 1
    assert sent_records[0].akashic_fields["session_id"] == ""
    assert sent_records[0].akashic_fields["turn_id"] == ""
    assert sent_records[0].akashic_fields["client_message_id"] == ""


def test_terminal_milestone_logger_contract_is_no_throw(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """观测使用的 turn_milestone 契约 no-throw：真实 logger 下相同字段形状不抛错，
    且结构化字段完整落入记录。"""

    caplog.set_level(logging.INFO, logger="infra.mobile_realtime.gateway")
    logger = logging.getLogger("infra.mobile_realtime.gateway")
    gateway_module.turn_milestone(
        logger,
        "tl:event.sent",
        session_id="akashic:s1",
        turn_id="akashic:t1",
        client_message_id="cmid-t",
        counts=(
            "event_type=message.final device_id=d1 " "event_seq=2 connection_epoch=3"
        ),
    )
    records = [
        record
        for record in caplog.records
        if getattr(record, "akashic_fields", {}).get("event") == "tl:event.sent"
    ]
    assert len(records) == 1
    assert records[0].akashic_fields["session_id"] == "akashic:s1"
    assert records[0].akashic_fields["turn_id"] == "akashic:t1"
    assert records[0].akashic_fields["client_message_id"] == "cmid-t"
    assert records[0].akashic_fields["counts"] == (
        "event_type=message.final device_id=d1 event_seq=2 connection_epoch=3"
    )


def test_live_observation_failure_keeps_cursor_and_epoch_after_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """sent 观测抛错时：frame 已送、cursor 已推进、连接/epoch 保持、同 seq 不重发。"""

    def broken_milestone(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("milestone logger broken")

    monkeypatch.setattr(gateway_module, "turn_milestone", broken_milestone)
    caplog.set_level(logging.INFO, logger="infra.mobile_realtime.gateway")

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
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )

        # 1. 终态事件真实 send；观测抛错不得回滚 cursor、摘除连接或杀死任务
        await runtime.publish_event(
            device_id=device_id,
            event_type="message.final",
            payload={"content": "done", "client_message_id": "cmid-t"},
            session_id="akashic:s1",
            turn_id="akashic:t1",
        )
        terminal_task = runtime._connections[device_id].delivery_task
        assert terminal_task is not None
        await asyncio.wait_for(terminal_task, timeout=5)
        assert [json.loads(text)["type"] for text in socket.sent_text] == [
            "message.final"
        ]
        assert [json.loads(text)["event_seq"] for text in socket.sent_text] == [1]
        cursor = runtime.storage.read_cursor(device_id)
        assert cursor.sent_event_seq == 1
        assert device_id in runtime._connections
        assert runtime._connections[device_id].connection_epoch == 1

        # 2. 同一 epoch 的后续投递照常工作：cursor 继续推进，同 seq 绝不重发
        await runtime.publish_event(
            device_id=device_id,
            event_type="connection.degraded",
            payload={"reason": "after-failure"},
        )
        second_task = runtime._connections[device_id].delivery_task
        assert second_task is not None
        await asyncio.wait_for(second_task, timeout=5)
        frames = [json.loads(text) for text in socket.sent_text]
        assert [frame["event_seq"] for frame in frames] == [1, 2]
        assert len(frames) == len({frame["event_seq"] for frame in frames})
        assert runtime.storage.read_cursor(device_id).sent_event_seq == 2

        # 3. 客户端按已推进 cursor resume：无任何旧 seq 重放
        await runtime._resume_and_register(
            cast(Any, socket),
            device_id=device_id,
            connection_epoch=3,
            last_ack=2,
        )
        replay_frames = [json.loads(text) for text in socket.sent_text]
        assert [frame["type"] for frame in replay_frames] == [
            "message.final",
            "connection.degraded",
            "sync.completed",
        ]
        assert all(frame["event_seq"] <= 3 for frame in replay_frames)
        assert [frame["event_seq"] for frame in replay_frames].count(1) == 1
        assert [frame["event_seq"] for frame in replay_frames].count(2) == 1
        runtime.close()

    asyncio.run(scenario())
    failure_records = [
        record for record in caplog.records if "sent 观测失败" in record.getMessage()
    ]
    assert len(failure_records) == 1
    assert "event_seq=1" in failure_records[0].getMessage()


def test_resume_observation_failure_keeps_cursor_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """resume 重放终态后观测抛错：帧已送、cursor 已推进、再次 resume 不重放同 seq。"""

    def broken_milestone(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("milestone logger broken")

    monkeypatch.setattr(gateway_module, "turn_milestone", broken_milestone)
    caplog.set_level(logging.INFO, logger="infra.mobile_realtime.gateway")

    async def scenario() -> None:
        runtime, _ = build_mobile_gateway_runtime(
            _config(),
            tmp_path,
            master_keys=_EphemeralMasterKeys(),
        )
        device_id = uuid4().hex
        _register_test_device(runtime, device_id)
        await runtime.publish_event(
            event_type="message.final",
            payload={"content": "done", "client_message_id": "cmid-t"},
            session_id="akashic:s1",
            turn_id="akashic:t1",
        )

        # 1. 无在线连接时事件只入箱；resume 重放终态 + sync.completed
        socket = _ControlledWebSocket()
        await runtime._resume_and_register(
            cast(Any, socket),
            device_id=device_id,
            connection_epoch=2,
            last_ack=0,
        )
        frames = [json.loads(text) for text in socket.sent_text]
        assert [frame["type"] for frame in frames] == [
            "message.final",
            "sync.completed",
        ]
        assert [frame["event_seq"] for frame in frames] == [1, 2]
        cursor = runtime.storage.read_cursor(device_id)
        assert cursor.sent_event_seq == 2
        assert device_id in runtime._connections
        assert runtime._connections[device_id].connection_epoch == 2

        # 2. 观测虽失败但 cursor 已提交：按 sent cursor 再次 resume 不重放同 seq
        second_socket = _ControlledWebSocket()
        await runtime._resume_and_register(
            cast(Any, second_socket),
            device_id=device_id,
            connection_epoch=3,
            last_ack=2,
        )
        replay_frames = [json.loads(text) for text in second_socket.sent_text]
        assert [frame["type"] for frame in replay_frames] == ["sync.completed"]
        assert all(frame["event_seq"] != 1 for frame in replay_frames)
        assert all(frame["event_seq"] != 2 for frame in replay_frames)
        runtime.close()

    asyncio.run(scenario())
    failure_records = [
        record for record in caplog.records if "sent 观测失败" in record.getMessage()
    ]
    assert len(failure_records) == 1
    assert "event_seq=1" in failure_records[0].getMessage()


def test_connection_control_only_reaches_matching_current_connection(
    tmp_path: Path,
) -> None:
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
                    "session_id": f"akashic:{uuid4()}",
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
        assert (
            len(
                runtime.storage.read_durable_events(
                    device_id,
                    after_event_seq=0,
                    limit=10,
                )
            )
            == 1
        )

        # 2. 只有当前 websocket 与 epoch 的 ACK 可以删除该前缀
        acknowledged = await runtime._acknowledge_active_connection(
            device_id=device_id,
            websocket=cast(Any, new_socket),
            connection_epoch=2,
            through_event_seq=1,
        )
        assert acknowledged is True
        assert runtime.storage.read_cursor(device_id).acknowledged_event_seq == 1
        assert (
            runtime.storage.read_durable_events(
                device_id,
                after_event_seq=0,
                limit=10,
            )
            == ()
        )
        runtime.close()

    asyncio.run(scenario())

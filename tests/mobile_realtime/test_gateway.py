from __future__ import annotations

import base64
import secrets
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from agent.config_models import MobileKeyEncryptionConfig, MobileRealtimeConfig
from infra.mobile_realtime.auth import device_proof_signing_bytes
from infra.mobile_realtime.gateway import (
    build_mobile_gateway_runtime,
    build_mobile_gateway_server,
    create_mobile_gateway_app,
)
from infra.mobile_realtime.key_protection import KeyProtectionError
from infra.mobile_realtime.storage import DeviceRecord


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

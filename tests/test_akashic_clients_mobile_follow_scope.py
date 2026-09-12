"""用真实 Mobile WebSocket 验证 follow scope 的任务归属和重连释放。"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager, closing, contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from agent.config_models import MobileRealtimeConfig
from agent.plugin_composition.messages import MESSAGE_CATALOG
from plugins.akashic_clients.mobile_realtime.auth import (
    DeviceAuthenticator,
    device_proof_signing_bytes,
)
from plugins.akashic_clients.mobile_realtime.channel import MobileRealtimeChannel
from plugins.akashic_clients.mobile_realtime.gateway import (
    MobileGatewayRuntime,
    PairingApprovalRegistry,
    create_mobile_gateway_app,
)
from plugins.akashic_clients.mobile_realtime.inbox import DurableInboxManager
from plugins.akashic_clients.mobile_realtime.key_protection import (
    FileMasterKeyStore,
    KeysetManager,
)
from plugins.akashic_clients.mobile_realtime.pairing import PairingService
from plugins.akashic_clients.mobile_realtime.storage import (
    DeviceRecord,
    MobileRealtimeStorage,
)
from plugins.akashic_clients.scoped_capabilities import open_request_scope
from session.log import MessageLog
from session.message import ContentPart, Input
from tests.test_mobile_message_log import append


class _HostScope:
    def __init__(self, catalog: Any) -> None:
        self._catalog = catalog

    def require(self, key: Any) -> Any:
        if key != MESSAGE_CATALOG:
            raise AssertionError(f"unexpected test capability: {key!r}")
        return self._catalog


@pytest.fixture
def gateway(tmp_path):
    loop = asyncio.new_event_loop()
    scope_tasks: list[tuple[str, asyncio.Task[Any] | None]] = []
    with (
        closing(MessageLog(tmp_path / "sessions.db")) as log,
        closing(MobileRealtimeStorage(tmp_path / "mobile.db")) as storage,
    ):
        keyset = KeysetManager(
            tmp_path / "keys", FileMasterKeyStore(tmp_path / "master.json")
        ).initialize(lan_hostname="localhost")
        private = ec.generate_private_key(ec.SECP256R1())
        public = base64.b64encode(
            private.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        ).decode()
        device = uuid4().hex
        storage.register_device(
            DeviceRecord(
                device,
                public,
                "fixture",
                datetime.now(timezone.utc),
                None,
                (),
            )
        )
        runtime = MobileGatewayRuntime(
            config=MobileRealtimeConfig(),
            storage=storage,
            pairing=PairingService(
                storage, keyset, lan_endpoints=(), tunnel_endpoints=()
            ),
            authenticator=DeviceAuthenticator(storage, keyset),
            inbox=DurableInboxManager(storage),
            approvals=PairingApprovalRegistry(loop),
            keyset=keyset,
        )
        channel = MobileRealtimeChannel(runtime)

        @asynccontextmanager
        async def host_scope():
            task = asyncio.current_task()
            scope_tasks.append(("enter", task))
            try:
                yield _HostScope(log.catalog())
            finally:
                scope_tasks.append(("exit", asyncio.current_task()))

        @asynccontextmanager
        async def message_scope():
            async with open_request_scope(host_scope) as scope:
                yield scope.require(MESSAGE_CATALOG)

        channel.bind_message_scope(message_scope)
        runtime.bind_channel(channel)
        with TestClient(create_mobile_gateway_app(runtime)) as client:
            yield log, runtime, client, device, private
        assert not runtime._message_followers
        assert not log._listeners

    assert len(scope_tasks) % 2 == 0
    for enter, exit_ in zip(scope_tasks[::2], scope_tasks[1::2]):
        assert enter[0] == "enter"
        assert exit_[0] == "exit"
        assert enter[1] is exit_[1]
    loop.close()


@contextmanager
def connected(gateway):
    _log, runtime, client, device, private = gateway
    with client.websocket_connect("/ws") as websocket:
        challenge = websocket.receive_json()["payload"]
        nonce = uuid4().hex
        signature = private.sign(
            device_proof_signing_bytes(
                server_id=challenge["server_id"],
                challenge_id=challenge["challenge_id"],
                challenge_nonce=challenge["nonce"],
                device_id=device,
                client_nonce=nonce,
            ),
            ec.ECDSA(hashes.SHA256()),
        )
        websocket.send_json(
            {
                "v": 1,
                "kind": "control",
                "type": "device.proof",
                "payload": {
                    "challenge_id": challenge["challenge_id"],
                    "device_id": device,
                    "client_nonce": nonce,
                    "signature": base64.b64encode(signature).decode(),
                },
            }
        )
        accepted = websocket.receive_json()
        assert accepted["type"] == "auth.accepted"
        epoch = accepted["connection_epoch"]
        replay_through = runtime.storage.read_cursor(device).next_event_seq - 1
        websocket.send_json(
            {
                "v": 1,
                "kind": "control",
                "type": "resume",
                "connection_epoch": epoch,
                "payload": {"last_ack": 0, "active_turns": []},
            }
        )
        while True:
            frame = websocket.receive_json()
            if frame["type"] == "sync.completed" and frame["event_seq"] > replay_through:
                break
        yield websocket, epoch


def _follow(websocket, epoch: int, session_id: str, command_id: str) -> None:
    websocket.send_json(
        {
            "v": 1,
            "kind": "command",
            "type": "session.follow",
            "id": command_id,
            "connection_epoch": epoch,
            "session_id": session_id,
            "payload": {"message_log_version": 2, "after_seq": -1},
        }
    )
    while True:
        reply = websocket.receive_json()
        if reply.get("type") == "session.follow.ok":
            return
        assert reply.get("type") == "session.message", reply


def test_mobile_follow_scope_stays_in_child_and_releases_on_reload(gateway):
    log, runtime, _client, _device, _private = gateway
    session_id = f"akashic:{uuid4()}"
    append(log, session_id, "one", Input((ContentPart("text", "hello"),)))

    with connected(gateway) as (websocket, epoch):
        _follow(websocket, epoch, session_id, "01ARZ3NDEKTSV4RRFFQ69G5FAV")
        # Replacing the follow cancels its child and opens a new exact scope.
        _follow(websocket, epoch, session_id, "01ARZ3NDEKTSV4RRFFQ69G5FAW")

    assert not runtime._message_followers

    # A fresh websocket is a real reload boundary; the next follow must open
    # a new scope after the previous connection has released its own scope.
    with connected(gateway) as (websocket, epoch):
        _follow(websocket, epoch, session_id, "01ARZ3NDEKTSV4RRFFQ69G5FAX")

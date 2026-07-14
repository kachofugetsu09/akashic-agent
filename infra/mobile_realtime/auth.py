from __future__ import annotations

import base64
import binascii
import json
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pydantic import BaseModel, ConfigDict, Field

from infra.mobile_realtime.key_protection import LoadedKeyset
from infra.mobile_realtime.pairing import parse_device_public_key
from infra.mobile_realtime.storage import MobileRealtimeStorage, UnknownDeviceError

_CHALLENGE_TTL = timedelta(seconds=30)


class AuthenticationError(RuntimeError):
    pass


class UnknownAuthenticationChallenge(AuthenticationError):
    pass


class AuthenticationChallengeExpired(AuthenticationError):
    pass


class DeviceAuthenticationError(AuthenticationError):
    pass


class DeviceRevokedError(AuthenticationError):
    pass


NonEmptyText = Annotated[str, Field(min_length=1, max_length=512)]


class DeviceProofPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    challenge_id: NonEmptyText
    device_id: NonEmptyText
    client_nonce: str = Field(min_length=16, max_length=128)
    signature: str = Field(min_length=64, max_length=512)


@dataclass(frozen=True, slots=True)
class ServerChallenge:
    challenge_id: str
    server_id: str
    server_public_key: str
    server_fingerprint: str
    nonce: str
    expires_at: datetime
    signature: str

    def payload(self) -> dict[str, object]:
        return {
            "challenge_id": self.challenge_id,
            "server_id": self.server_id,
            "server_public_key": self.server_public_key,
            "server_fingerprint": self.server_fingerprint,
            "nonce": self.nonce,
            "expires_at": _format_datetime(self.expires_at),
            "signature": self.signature,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedDevice:
    device_id: str
    display_name: str
    capabilities: tuple[str, ...]
    connection_epoch: int


@dataclass(frozen=True, slots=True)
class _ChallengeRecord:
    connection_id: str
    challenge_id: str
    nonce: str
    expires_at: datetime


class DeviceAuthenticator:
    """签发服务端 challenge，并校验已配对设备的单次连接证明。"""

    def __init__(
        self,
        storage: MobileRealtimeStorage,
        keyset: LoadedKeyset,
        *,
        challenge_ttl: timedelta = _CHALLENGE_TTL,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if challenge_ttl <= timedelta(0):
            raise ValueError("challenge_ttl 必须大于零")
        self._storage = storage
        self._keyset = keyset
        self._challenge_ttl = challenge_ttl
        self._clock = clock
        self._challenges: dict[str, _ChallengeRecord] = {}
        self._lock = threading.RLock()

    def create_challenge(self, connection_id: str) -> ServerChallenge:
        """为一条 TLS 连接签发短时、单次使用的应用层 challenge。"""

        if not connection_id:
            raise ValueError("connection_id 不能为空")
        now = self._now()
        record = _ChallengeRecord(
            connection_id=connection_id,
            challenge_id=uuid4().hex,
            nonce=_b64url(secrets.token_bytes(32)),
            expires_at=now + self._challenge_ttl,
        )
        public_key_bytes = self._keyset.identity_private_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        public_key = base64.b64encode(public_key_bytes).decode("ascii")
        signing_bytes = _server_challenge_signing_bytes(
            challenge_id=record.challenge_id,
            server_id=self._keyset.manifest.server_id,
            server_fingerprint=self._keyset.server_fingerprint,
            server_public_key=public_key,
            nonce=record.nonce,
            expires_at=record.expires_at,
        )
        signature = self._keyset.identity_private_key.sign(
            signing_bytes,
            ec.ECDSA(hashes.SHA256()),
        )
        with self._lock:
            self._purge_expired_locked(now)
            if connection_id in self._challenges:
                raise AuthenticationError("同一连接已存在待验证 challenge")
            self._challenges[connection_id] = record
        return ServerChallenge(
            challenge_id=record.challenge_id,
            server_id=self._keyset.manifest.server_id,
            server_public_key=public_key,
            server_fingerprint=self._keyset.server_fingerprint,
            nonce=record.nonce,
            expires_at=record.expires_at,
            signature=base64.b64encode(signature).decode("ascii"),
        )

    def authenticate(
        self,
        connection_id: str,
        proof: DeviceProofPayload,
    ) -> AuthenticatedDevice:
        """消费 challenge、验签设备证明，并分配新的 connection epoch。"""

        # 1. challenge 由连接拥有，任何一次证明尝试都会消费它
        now = self._now()
        with self._lock:
            record = self._challenges.pop(connection_id, None)
        if record is None or record.challenge_id != proof.challenge_id:
            raise UnknownAuthenticationChallenge("设备证明未匹配当前连接 challenge")
        if record.expires_at <= now:
            raise AuthenticationChallengeExpired("设备认证 challenge 已过期")

        # 2. 设备状态和公钥只从持久化配对记录读取
        device = self._storage.read_device(proof.device_id)
        if device is None:
            raise UnknownDeviceError(f"设备不存在: {proof.device_id}")
        if device.revoked_at is not None:
            raise DeviceRevokedError(f"设备已撤销: {proof.device_id}")
        signing_bytes = device_proof_signing_bytes(
            server_id=self._keyset.manifest.server_id,
            challenge_id=record.challenge_id,
            challenge_nonce=record.nonce,
            device_id=proof.device_id,
            client_nonce=proof.client_nonce,
        )
        try:
            parse_device_public_key(device.public_key).verify(
                _decode_base64(proof.signature),
                signing_bytes,
                ec.ECDSA(hashes.SHA256()),
            )
        except (InvalidSignature, ValueError) as error:
            raise DeviceAuthenticationError("设备 challenge 签名无效") from error

        # 3. epoch 由 SQLite 原子分配，服务重启后仍拒绝旧代连接
        connection_epoch = self._storage.allocate_connection_epoch()
        return AuthenticatedDevice(
            device_id=device.device_id,
            display_name=device.display_name,
            capabilities=device.capabilities,
            connection_epoch=connection_epoch,
        )

    def _purge_expired_locked(self, now: datetime) -> None:
        expired = [
            connection_id
            for connection_id, challenge in self._challenges.items()
            if challenge.expires_at <= now
        ]
        for connection_id in expired:
            del self._challenges[connection_id]

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("auth clock 必须返回带时区的 datetime")
        return now


def server_challenge_signing_bytes(challenge: ServerChallenge) -> bytes:
    return _server_challenge_signing_bytes(
        challenge_id=challenge.challenge_id,
        server_id=challenge.server_id,
        server_fingerprint=challenge.server_fingerprint,
        server_public_key=challenge.server_public_key,
        nonce=challenge.nonce,
        expires_at=challenge.expires_at,
    )


def device_proof_signing_bytes(
    *,
    server_id: str,
    challenge_id: str,
    challenge_nonce: str,
    device_id: str,
    client_nonce: str,
) -> bytes:
    return _canonical_json(
        {
            "challenge_id": challenge_id,
            "challenge_nonce": challenge_nonce,
            "client_nonce": client_nonce,
            "device_id": device_id,
            "protocol_version": 1,
            "server_id": server_id,
        }
    )


def _server_challenge_signing_bytes(
    *,
    challenge_id: str,
    server_id: str,
    server_fingerprint: str,
    server_public_key: str,
    nonce: str,
    expires_at: datetime,
) -> bytes:
    return _canonical_json(
        {
            "challenge_id": challenge_id,
            "expires_at": _format_datetime(expires_at),
            "nonce": nonce,
            "protocol_version": 1,
            "server_fingerprint": server_fingerprint,
            "server_id": server_id,
            "server_public_key": server_public_key,
        }
    )


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise DeviceAuthenticationError("设备签名不是合法 Base64") from error


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _format_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "AuthenticatedDevice",
    "AuthenticationChallengeExpired",
    "AuthenticationError",
    "DeviceAuthenticationError",
    "DeviceAuthenticator",
    "DeviceProofPayload",
    "DeviceRevokedError",
    "ServerChallenge",
    "UnknownAuthenticationChallenge",
    "device_proof_signing_bytes",
    "server_challenge_signing_bytes",
]

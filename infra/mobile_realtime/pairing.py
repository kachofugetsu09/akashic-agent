from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from infra.mobile_realtime.key_protection import LoadedKeyset
from infra.mobile_realtime.storage import (
    DeviceRecord,
    MobileRealtimeStorage,
    PairingSessionRecord,
    PairingStateError,
)

_PAIRING_SECRET_BYTES = 32
_PAIRING_SECRET_DOMAIN = b"akasic-mobile-pairing-secret-v1\x00"
_PAIRING_TTL = timedelta(minutes=8)
_MAX_ENDPOINTS = 16


class PairingError(RuntimeError):
    pass


class PairingSecretError(PairingError):
    pass


class PairingSignatureError(PairingError):
    pass


class PairingConfirmationError(PairingError):
    pass


NonEmptyText = Annotated[str, Field(min_length=1, max_length=512)]


class PairClaimPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pairing_id: NonEmptyText
    one_time_secret: str = Field(pattern=r"^[A-Za-z0-9_-]{43}$")
    device_public_key: str = Field(min_length=80, max_length=2048)
    device_name: str = Field(min_length=1, max_length=128)
    capabilities: list[NonEmptyText] = Field(max_length=128)
    client_nonce: str = Field(min_length=16, max_length=128)
    signature: str = Field(min_length=64, max_length=512)

    @field_validator("capabilities")
    @classmethod
    def require_unique_capabilities(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("capabilities 不能重复")
        return value


@dataclass(frozen=True, slots=True)
class PairingOffer:
    protocol_version: int
    server_id: str
    server_application_key_fingerprint: str
    server_application_public_key: str
    lan_endpoints: tuple[str, ...]
    tunnel_endpoints: tuple[str, ...]
    tls_spki_pins: tuple[str, ...]
    pairing_id: str
    one_time_secret: str
    expires_at: datetime

    def qr_payload(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "server_id": self.server_id,
            "server_application_key_fingerprint": (
                self.server_application_key_fingerprint
            ),
            "server_application_public_key": self.server_application_public_key,
            "lan_endpoints": list(self.lan_endpoints),
            "tunnel_endpoints": list(self.tunnel_endpoints),
            "tls_spki_pins": list(self.tls_spki_pins),
            "pairing_id": self.pairing_id,
            "one_time_secret": self.one_time_secret,
            "expires_at": _format_datetime(self.expires_at),
        }


@dataclass(frozen=True, slots=True)
class PendingPairingClaim:
    pairing_id: str
    device_public_key: str
    device_name: str
    capabilities: tuple[str, ...]
    client_nonce: str
    confirmation_code: str


class PairingService:
    """创建一次性 QR，校验手机 claim，并在电脑确认后登记设备。"""

    def __init__(
        self,
        storage: MobileRealtimeStorage,
        keyset: LoadedKeyset,
        *,
        lan_endpoints: tuple[str, ...],
        tunnel_endpoints: tuple[str, ...],
        ttl: timedelta = _PAIRING_TTL,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("pairing ttl 必须大于零")
        _validate_endpoints(lan_endpoints, "lan_endpoints")
        _validate_endpoints(tunnel_endpoints, "tunnel_endpoints")
        self._storage = storage
        self._keyset = keyset
        self._lan_endpoints = lan_endpoints
        self._tunnel_endpoints = tunnel_endpoints
        self._ttl = ttl
        self._clock = clock
        self._claims: dict[str, PendingPairingClaim] = {}
        self._lock = threading.RLock()

    def create_offer(self) -> PairingOffer:
        """创建只在本机 WebChat 展示的一次性 QR payload。"""

        # 1. 只把不可逆 secret hash 写入数据库
        pairing_id = uuid4().hex
        one_time_secret = _b64url(secrets.token_bytes(_PAIRING_SECRET_BYTES))
        expires_at = self._now() + self._ttl
        self._storage.create_pairing_session(
            PairingSessionRecord(
                pairing_id=pairing_id,
                secret_hash=_pairing_secret_hash(one_time_secret),
                expires_at=expires_at,
                status="pending",
            )
        )

        # 2. QR 携带验证应用身份所需的公钥与稳定 fingerprint
        public_key = self._keyset.identity_private_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return PairingOffer(
            protocol_version=1,
            server_id=self._keyset.manifest.server_id,
            server_application_key_fingerprint=self._keyset.server_fingerprint,
            server_application_public_key=base64.b64encode(public_key).decode("ascii"),
            lan_endpoints=self._lan_endpoints,
            tunnel_endpoints=self._tunnel_endpoints,
            tls_spki_pins=(self._keyset.tls_spki_fingerprint,),
            pairing_id=pairing_id,
            one_time_secret=one_time_secret,
            expires_at=expires_at,
        )

    def claim(self, payload: PairClaimPayload) -> PendingPairingClaim:
        """验证一次性 secret 和设备签名，并生成双方一致的确认码。"""

        # 1. 数据库拥有 secret 生命周期和过期状态
        session = self._storage.read_pairing_session(payload.pairing_id)
        if session is None:
            raise PairingSecretError("配对会话不存在")
        if session.status not in {"pending", "confirmed"}:
            raise PairingStateError(f"配对会话不能 claim: status={session.status}")
        if session.expires_at <= self._now():
            raise PairingSecretError("配对会话已过期")
        if session.secret_hash is None or not hmac.compare_digest(
            session.secret_hash,
            _pairing_secret_hash(payload.one_time_secret),
        ):
            raise PairingSecretError("一次性 pairing secret 无效")

        # 2. 设备必须证明持有其声明的 P-256 私钥
        transcript = _pair_claim_transcript(
            server_id=self._keyset.manifest.server_id,
            pairing_id=payload.pairing_id,
            secret_hash=session.secret_hash,
            device_public_key=payload.device_public_key,
            device_name=payload.device_name,
            capabilities=payload.capabilities,
            client_nonce=payload.client_nonce,
        )
        device_public_key = parse_device_public_key(payload.device_public_key)
        try:
            device_public_key.verify(
                _decode_base64(payload.signature, "signature"),
                transcript,
                ec.ECDSA(hashes.SHA256()),
            )
        except (InvalidSignature, ValueError) as error:
            raise PairingSignatureError("pair claim 设备签名无效") from error

        # 3. 同一 pairing 只接受同一 transcript 的幂等重试
        claim = PendingPairingClaim(
            pairing_id=payload.pairing_id,
            device_public_key=payload.device_public_key,
            device_name=payload.device_name,
            capabilities=tuple(payload.capabilities),
            client_nonce=payload.client_nonce,
            confirmation_code=_confirmation_code(transcript),
        )
        with self._lock:
            existing = self._claims.get(payload.pairing_id)
            if existing is not None and existing != claim:
                raise PairingStateError("同一 pairing_id 已存在不同设备 claim")
            self._claims[payload.pairing_id] = claim
        return claim

    def pending_claim(self, pairing_id: str) -> PendingPairingClaim | None:
        with self._lock:
            return self._claims.get(pairing_id)

    def approve(self, pairing_id: str, confirmation_code: str) -> DeviceRecord:
        """校验电脑确认码，并消费 secret、登记可自动认证的设备。"""

        # 1. 电脑只能批准当前进程已验签的 claim
        with self._lock:
            claim = self._claims.get(pairing_id)
        if claim is None:
            raise PairingConfirmationError("尚无可确认的设备 claim")
        if not hmac.compare_digest(claim.confirmation_code, confirmation_code):
            raise PairingConfirmationError("配对确认码不一致")

        # 2. pending -> confirmed 可在消费失败后幂等恢复
        session = self._storage.read_pairing_session(pairing_id)
        if session is None:
            raise PairingConfirmationError("配对会话不存在")
        now = self._now()
        if session.status == "pending":
            _ = self._storage.confirm_pairing(pairing_id, now=now)
        elif session.status != "confirmed":
            raise PairingStateError(f"配对会话不能批准: status={session.status}")

        # 3. 注册设备和作废 secret 由 storage 在同一事务提交
        device = DeviceRecord(
            device_id=uuid4().hex,
            public_key=claim.device_public_key,
            display_name=claim.device_name,
            created_at=now,
            revoked_at=None,
            capabilities=claim.capabilities,
        )
        device = self._storage.consume_pairing(pairing_id, device, now=now)
        with self._lock:
            del self._claims[pairing_id]
        return device

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("pairing clock 必须返回带时区的 datetime")
        return now


def pair_claim_signing_bytes(
    *,
    server_id: str,
    pairing_id: str,
    one_time_secret: str,
    device_public_key: str,
    device_name: str,
    capabilities: list[str],
    client_nonce: str,
) -> bytes:
    """供 Android 和契约测试生成完全一致的 pair claim transcript。"""

    return _pair_claim_transcript(
        server_id=server_id,
        pairing_id=pairing_id,
        secret_hash=_pairing_secret_hash(one_time_secret),
        device_public_key=device_public_key,
        device_name=device_name,
        capabilities=capabilities,
        client_nonce=client_nonce,
    )


def parse_device_public_key(encoded: str) -> ec.EllipticCurvePublicKey:
    """解析 Android X.509 DER 公钥，并把曲线限制为 P-256。"""

    try:
        public_key = serialization.load_der_public_key(
            _decode_base64(encoded, "device_public_key")
        )
    except (TypeError, ValueError) as error:
        raise PairingSignatureError("设备公钥解析失败") from error
    if not isinstance(public_key, ec.EllipticCurvePublicKey) or not isinstance(
        public_key.curve, ec.SECP256R1
    ):
        raise PairingSignatureError("设备公钥必须是 ECDSA P-256")
    return public_key


def _pair_claim_transcript(
    *,
    server_id: str,
    pairing_id: str,
    secret_hash: str,
    device_public_key: str,
    device_name: str,
    capabilities: list[str],
    client_nonce: str,
) -> bytes:
    return _canonical_json(
        {
            "capabilities": capabilities,
            "client_nonce": client_nonce,
            "device_name": device_name,
            "device_public_key": device_public_key,
            "pairing_id": pairing_id,
            "protocol_version": 1,
            "secret_hash": secret_hash,
            "server_id": server_id,
        }
    )


def _pairing_secret_hash(one_time_secret: str) -> str:
    return hashlib.sha256(
        _PAIRING_SECRET_DOMAIN + one_time_secret.encode("ascii")
    ).hexdigest()


def _confirmation_code(transcript: bytes) -> str:
    value = int.from_bytes(hashlib.sha256(transcript).digest()[:8], "big") % 1_000_000
    return f"{value:06d}"


def _validate_endpoints(endpoints: tuple[str, ...], field: str) -> None:
    if len(endpoints) > _MAX_ENDPOINTS:
        raise ValueError(f"{field} 数量超过上限")
    for endpoint in endpoints:
        if not endpoint.startswith("wss://"):
            raise ValueError(f"{field} 只允许 wss:// endpoint")


def _decode_base64(value: str, field: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise PairingSignatureError(f"{field} 不是合法 Base64") from error


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
    "PairClaimPayload",
    "PairingConfirmationError",
    "PairingError",
    "PairingOffer",
    "PairingSecretError",
    "PairingService",
    "PairingSignatureError",
    "PendingPairingClaim",
    "pair_claim_signing_bytes",
    "parse_device_public_key",
]

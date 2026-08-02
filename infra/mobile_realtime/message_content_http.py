from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from infra.mobile_realtime.key_protection import LoadedKeyset
from infra.mobile_realtime.storage import MobileRealtimeStorage

_AUDIENCE = "mobile-message-content"
_DEFAULT_TTL = timedelta(seconds=60)


class MessageContentTicketError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MessageContentGrant:
    ticket: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class VerifiedMessageContentTicket:
    device_id: str
    connection_epoch: int
    session_id: str
    message_id: str
    byte_length: int
    sha256: str


class MessageContentTicketIssuer:
    """签发并校验绑定不可变消息正文的短期下载授权。"""

    def __init__(
        self,
        keyset: LoadedKeyset,
        storage: MobileRealtimeStorage,
        *,
        ttl: timedelta = _DEFAULT_TTL,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("message content ticket ttl 必须大于零")
        self._keyset = keyset
        self._storage = storage
        self._ttl = ttl
        self._clock = clock

    def issue(
        self,
        *,
        device_id: str,
        connection_epoch: int,
        session_id: str,
        message_id: str,
        byte_length: int,
        sha256: str,
    ) -> MessageContentGrant:
        """把设备身份和正文摘要封装成短期签名授权。"""

        now = self._now()
        expires_at = now + self._ttl
        claims = {
            "aud": _AUDIENCE,
            "byte_length": byte_length,
            "connection_epoch": connection_epoch,
            "device_id": device_id,
            "exp": _unix_seconds(expires_at),
            "iat": _unix_seconds(now),
            "message_id": message_id,
            "server_id": self._keyset.manifest.server_id,
            "session_id": session_id,
            "sha256": sha256,
            "v": 1,
        }
        payload = _canonical_json(claims)
        signature = self._keyset.identity_private_key.sign(
            payload,
            ec.ECDSA(hashes.SHA256()),
        )
        return MessageContentGrant(
            ticket=f"{_b64url(payload)}.{_b64url(signature)}",
            expires_at=expires_at,
        )

    def verify(self, ticket: str) -> VerifiedMessageContentTicket:
        """验签、校验不可变正文身份并重新读取设备撤销状态。"""

        # 1. 在 HTTP 边界一次性校验结构、签名和 claims
        parts = ticket.split(".")
        if len(parts) != 2:
            raise MessageContentTicketError("message content ticket 格式无效")
        payload = _decode_b64url(parts[0])
        signature = _decode_b64url(parts[1])
        try:
            self._keyset.identity_private_key.public_key().verify(
                signature,
                payload,
                ec.ECDSA(hashes.SHA256()),
            )
        except InvalidSignature as error:
            raise MessageContentTicketError("message content ticket 签名无效") from error
        claims = _decode_claims(payload)
        _require_exact_keys(
            claims,
            {
                "aud",
                "byte_length",
                "connection_epoch",
                "device_id",
                "exp",
                "iat",
                "message_id",
                "server_id",
                "session_id",
                "sha256",
                "v",
            },
        )
        if claims["v"] != 1 or claims["aud"] != _AUDIENCE:
            raise MessageContentTicketError("message content ticket 版本或用途无效")
        if claims["server_id"] != self._keyset.manifest.server_id:
            raise MessageContentTicketError("message content ticket 不属于当前服务端")
        device_id = _require_text(claims["device_id"], "device_id", 512)
        connection_epoch = _require_positive_int(
            claims["connection_epoch"],
            "connection_epoch",
        )
        issued_at = _require_nonnegative_int(claims["iat"], "iat")
        expires_at = _require_nonnegative_int(claims["exp"], "exp")
        now_seconds = _unix_seconds(self._now())
        if expires_at <= now_seconds or issued_at > now_seconds:
            raise MessageContentTicketError("message content ticket 已过期或尚未生效")
        sha256 = _require_text(claims["sha256"], "sha256", 64).lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise MessageContentTicketError("message content ticket sha256 无效")

        # 2. 授权执行前重新读取撤销状态，短期票据不成为持久权限
        device = self._storage.read_device(device_id)
        if device is None or device.revoked_at is not None:
            raise MessageContentTicketError("message content ticket 的设备无效")
        return VerifiedMessageContentTicket(
            device_id=device_id,
            connection_epoch=connection_epoch,
            session_id=_require_text(claims["session_id"], "session_id", 512),
            message_id=_require_text(claims["message_id"], "message_id", 512),
            byte_length=_require_nonnegative_int(claims["byte_length"], "byte_length"),
            sha256=sha256,
        )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("message content ticket clock 必须返回带时区时间")
        return now.astimezone(timezone.utc)


def format_message_content_expiry(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MessageContentTicketError("message content ticket 无法规范化") from error


def _decode_claims(payload: bytes) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise MessageContentTicketError(f"message content ticket 字段重复: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise MessageContentTicketError(f"message content ticket 包含非标准常量: {value}")

    try:
        claims = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MessageContentTicketError("message content ticket claims 无效") from error
    if not isinstance(claims, dict):
        raise MessageContentTicketError("message content ticket claims 必须是对象")
    return claims


def _decode_b64url(value: str) -> bytes:
    if not value:
        raise MessageContentTicketError("message content ticket 段不能为空")
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as error:
        raise MessageContentTicketError("message content ticket Base64URL 无效") from error


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unix_seconds(value: datetime) -> int:
    return int(value.timestamp())


def _require_exact_keys(raw: dict[str, object], expected: set[str]) -> None:
    if set(raw) != expected:
        raise MessageContentTicketError("message content ticket claims 字段无效")


def _require_text(value: object, field: str, max_length: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= max_length:
        raise MessageContentTicketError(f"message content ticket {field} 无效")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MessageContentTicketError(f"message content ticket {field} 无效")
    return value


def _require_positive_int(value: object, field: str) -> int:
    parsed = _require_nonnegative_int(value, field)
    if parsed == 0:
        raise MessageContentTicketError(f"message content ticket {field} 无效")
    return parsed


__all__ = [
    "MessageContentGrant",
    "MessageContentTicketError",
    "MessageContentTicketIssuer",
    "VerifiedMessageContentTicket",
    "format_message_content_expiry",
]

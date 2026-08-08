from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from infra.mobile_realtime.key_protection import LoadedKeyset
from infra.mobile_realtime.signed_ticket import (
    b64url,
    decode_b64url,
    require_exact_keys,
    require_nonnegative_int,
    require_positive_int,
    require_text,
    sign,
    unix_seconds,
    verify_signature,
)
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
            "exp": unix_seconds(expires_at),
            "iat": unix_seconds(now),
            "message_id": message_id,
            "server_id": self._keyset.manifest.server_id,
            "session_id": session_id,
            "sha256": sha256,
            "v": 1,
        }
        return MessageContentGrant(
            ticket=sign(
                self._keyset,
                claims,
                label="message content ticket",
                error_factory=MessageContentTicketError,
            ),
            expires_at=expires_at,
        )

    def verify(self, ticket: str) -> VerifiedMessageContentTicket:
        """验签、校验不可变正文身份并重新读取设备撤销状态。"""

        # 1. 在 HTTP 边界一次性校验结构、签名和 claims
        claims = verify_signature(
            ticket,
            self._keyset,
            label="message content ticket",
            error_factory=MessageContentTicketError,
        )
        require_exact_keys(
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
            label="message content ticket",
            error_factory=MessageContentTicketError,
        )
        if claims["v"] != 1 or claims["aud"] != _AUDIENCE:
            raise MessageContentTicketError("message content ticket 版本或用途无效")
        if claims["server_id"] != self._keyset.manifest.server_id:
            raise MessageContentTicketError("message content ticket 不属于当前服务端")
        device_id = require_text(claims["device_id"], "device_id", 512, label="message content ticket", error_factory=MessageContentTicketError)
        connection_epoch = require_positive_int(claims["connection_epoch"], "connection_epoch", label="message content ticket", error_factory=MessageContentTicketError)
        issued_at = require_nonnegative_int(claims["iat"], "iat", label="message content ticket", error_factory=MessageContentTicketError)
        expires_at = require_nonnegative_int(claims["exp"], "exp", label="message content ticket", error_factory=MessageContentTicketError)
        now_seconds = unix_seconds(self._now())
        if expires_at <= now_seconds or issued_at > now_seconds:
            raise MessageContentTicketError("message content ticket 已过期或尚未生效")
        sha256 = require_text(claims["sha256"], "sha256", 64, label="message content ticket", error_factory=MessageContentTicketError).lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise MessageContentTicketError("message content ticket sha256 无效")

        # 2. 授权执行前重新读取撤销状态，短期票据不成为持久权限
        device = self._storage.read_device(device_id)
        if device is None or device.revoked_at is not None:
            raise MessageContentTicketError("message content ticket 的设备无效")
        return VerifiedMessageContentTicket(
            device_id=device_id,
            connection_epoch=connection_epoch,
            session_id=require_text(claims["session_id"], "session_id", 512, label="message content ticket", error_factory=MessageContentTicketError),
            message_id=require_text(claims["message_id"], "message_id", 512, label="message content ticket", error_factory=MessageContentTicketError),
            byte_length=require_nonnegative_int(claims["byte_length"], "byte_length", label="message content ticket", error_factory=MessageContentTicketError),
            sha256=sha256,
        )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("message content ticket clock 必须返回带时区时间")
        return now.astimezone(timezone.utc)


def format_message_content_expiry(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "MessageContentGrant",
    "MessageContentTicketError",
    "MessageContentTicketIssuer",
    "VerifiedMessageContentTicket",
    "format_message_content_expiry",
]

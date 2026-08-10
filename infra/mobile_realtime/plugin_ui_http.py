from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from infra.mobile_realtime.key_protection import LoadedKeyset
from infra.mobile_realtime.signed_ticket import (
    canonical_json,
    require_exact_keys,
    require_nonnegative_int,
    require_positive_int,
    require_text,
    sign,
    unix_seconds,
    verify_signature,
)
from infra.mobile_realtime.storage import MobileRealtimeStorage

_AUDIENCE = "mobile-plugin-ui-query"
_DEFAULT_TTL = timedelta(seconds=30)


class PluginUiHttpTicketError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PluginUiHttpGrant:
    ticket: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class VerifiedPluginUiHttpTicket:
    device_id: str
    connection_epoch: int


class PluginUiHttpTicketIssuer:
    """签发并校验绑定请求摘要的短期插件查询授权。"""

    def __init__(
        self,
        keyset: LoadedKeyset,
        storage: MobileRealtimeStorage,
        *,
        ttl: timedelta = _DEFAULT_TTL,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("plugin UI HTTP ticket ttl 必须大于零")
        self._keyset = keyset
        self._storage = storage
        self._ttl = ttl
        self._clock = clock

    def issue(
        self,
        *,
        device_id: str,
        connection_epoch: int,
        request_body: dict[str, object],
    ) -> PluginUiHttpGrant:
        """把已认证设备和不可变查询摘要封装成短期签名授权。"""

        now = self._now()
        expires_at = now + self._ttl
        claims = {
            "aud": _AUDIENCE,
            "connection_epoch": connection_epoch,
            "device_id": device_id,
            "exp": unix_seconds(expires_at),
            "iat": unix_seconds(now),
            "request_sha256": plugin_ui_http_request_sha256(request_body),
            "server_id": self._keyset.manifest.server_id,
            "v": 1,
        }
        return PluginUiHttpGrant(
            ticket=sign(
                self._keyset,
                claims,
                label="plugin UI HTTP ticket",
                error_factory=PluginUiHttpTicketError,
            ),
            expires_at=expires_at,
        )

    def verify(
        self,
        ticket: str,
        *,
        request_body: dict[str, object],
    ) -> VerifiedPluginUiHttpTicket:
        """验签、核对请求摘要与设备撤销状态，并返回已认证身份。"""

        # 1. token 结构、签名和 claims 都在 HTTP 信任边界一次性校验
        claims = verify_signature(
            ticket,
            self._keyset,
            label="plugin UI HTTP ticket",
            error_factory=PluginUiHttpTicketError,
        )
        require_exact_keys(
            claims,
            {
                "aud",
                "connection_epoch",
                "device_id",
                "exp",
                "iat",
                "request_sha256",
                "server_id",
                "v",
            },
            label="plugin UI HTTP ticket",
            error_factory=PluginUiHttpTicketError,
        )
        if claims["v"] != 1 or claims["aud"] != _AUDIENCE:
            raise PluginUiHttpTicketError("plugin UI HTTP ticket 版本或用途无效")
        if claims["server_id"] != self._keyset.manifest.server_id:
            raise PluginUiHttpTicketError("plugin UI HTTP ticket 不属于当前服务端")
        device_id = require_text(claims["device_id"], "device_id", 512, label="plugin UI HTTP ticket", error_factory=PluginUiHttpTicketError)
        connection_epoch = require_positive_int(claims["connection_epoch"], "connection_epoch", label="plugin UI HTTP ticket", error_factory=PluginUiHttpTicketError)
        issued_at = require_nonnegative_int(claims["iat"], "iat", label="plugin UI HTTP ticket", error_factory=PluginUiHttpTicketError)
        expires_at = require_nonnegative_int(claims["exp"], "exp", label="plugin UI HTTP ticket", error_factory=PluginUiHttpTicketError)
        now_seconds = unix_seconds(self._now())
        if expires_at <= now_seconds or issued_at > now_seconds:
            raise PluginUiHttpTicketError("plugin UI HTTP ticket 已过期或尚未生效")
        expected_digest = plugin_ui_http_request_sha256(request_body)
        if claims["request_sha256"] != expected_digest:
            raise PluginUiHttpTicketError("plugin UI HTTP ticket 与请求不匹配")

        # 2. 撤销状态在执行前实时读取，签名授权不能绕过设备撤销
        device = self._storage.read_device(device_id)
        if device is None or device.revoked_at is not None:
            raise PluginUiHttpTicketError("plugin UI HTTP ticket 的设备无效")
        return VerifiedPluginUiHttpTicket(
            device_id=device_id,
            connection_epoch=connection_epoch,
        )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("plugin UI HTTP ticket clock 必须返回带时区时间")
        return now.astimezone(timezone.utc)


def plugin_ui_http_request_sha256(request_body: dict[str, object]) -> str:
    return hashlib.sha256(
        canonical_json(
            request_body,
            label="plugin UI HTTP 请求",
            error_factory=PluginUiHttpTicketError,
        )
    ).hexdigest()


def format_plugin_ui_http_expiry(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "PluginUiHttpGrant",
    "PluginUiHttpTicketError",
    "PluginUiHttpTicketIssuer",
    "VerifiedPluginUiHttpTicket",
    "format_plugin_ui_http_expiry",
    "plugin_ui_http_request_sha256",
]

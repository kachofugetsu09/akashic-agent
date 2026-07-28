from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from infra.mobile_realtime.key_protection import LoadedKeyset
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
            "exp": _unix_seconds(expires_at),
            "iat": _unix_seconds(now),
            "request_sha256": plugin_ui_http_request_sha256(request_body),
            "server_id": self._keyset.manifest.server_id,
            "v": 1,
        }
        payload = _canonical_json(claims)
        signature = self._keyset.identity_private_key.sign(
            payload,
            ec.ECDSA(hashes.SHA256()),
        )
        return PluginUiHttpGrant(
            ticket=f"{_b64url(payload)}.{_b64url(signature)}",
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
        parts = ticket.split(".")
        if len(parts) != 2:
            raise PluginUiHttpTicketError("plugin UI HTTP ticket 格式无效")
        payload = _decode_b64url(parts[0])
        signature = _decode_b64url(parts[1])
        try:
            self._keyset.identity_private_key.public_key().verify(
                signature,
                payload,
                ec.ECDSA(hashes.SHA256()),
            )
        except InvalidSignature as error:
            raise PluginUiHttpTicketError("plugin UI HTTP ticket 签名无效") from error
        claims = _decode_claims(payload)
        _require_exact_keys(
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
        )
        if claims["v"] != 1 or claims["aud"] != _AUDIENCE:
            raise PluginUiHttpTicketError("plugin UI HTTP ticket 版本或用途无效")
        if claims["server_id"] != self._keyset.manifest.server_id:
            raise PluginUiHttpTicketError("plugin UI HTTP ticket 不属于当前服务端")
        device_id = _require_text(claims["device_id"], "device_id", 512)
        connection_epoch = _require_positive_int(
            claims["connection_epoch"],
            "connection_epoch",
        )
        issued_at = _require_nonnegative_int(claims["iat"], "iat")
        expires_at = _require_nonnegative_int(claims["exp"], "exp")
        now_seconds = _unix_seconds(self._now())
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
    return hashlib.sha256(_canonical_json(request_body)).hexdigest()


def format_plugin_ui_http_expiry(value: datetime) -> str:
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
        raise PluginUiHttpTicketError("plugin UI HTTP 请求无法规范化") from error


def _decode_claims(payload: bytes) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise PluginUiHttpTicketError(
                    f"plugin UI HTTP ticket claims 字段重复: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PluginUiHttpTicketError(
            f"plugin UI HTTP ticket claims 包含非标准常量: {value}"
        )

    try:
        claims = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PluginUiHttpTicketError("plugin UI HTTP ticket claims 无效") from error
    if not isinstance(claims, dict):
        raise PluginUiHttpTicketError("plugin UI HTTP ticket claims 必须是对象")
    return claims


def _decode_b64url(value: str) -> bytes:
    if not value:
        raise PluginUiHttpTicketError("plugin UI HTTP ticket 段不能为空")
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as error:
        raise PluginUiHttpTicketError("plugin UI HTTP ticket Base64URL 无效") from error


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unix_seconds(value: datetime) -> int:
    return int(value.timestamp())


def _require_exact_keys(raw: dict[str, object], expected: set[str]) -> None:
    if set(raw) != expected:
        raise PluginUiHttpTicketError("plugin UI HTTP ticket claims 字段无效")


def _require_text(value: object, field: str, max_length: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= max_length:
        raise PluginUiHttpTicketError(f"plugin UI HTTP ticket {field} 无效")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PluginUiHttpTicketError(f"plugin UI HTTP ticket {field} 无效")
    return value


def _require_positive_int(value: object, field: str) -> int:
    parsed = _require_nonnegative_int(value, field)
    if parsed == 0:
        raise PluginUiHttpTicketError(f"plugin UI HTTP ticket {field} 无效")
    return parsed


__all__ = [
    "PluginUiHttpGrant",
    "PluginUiHttpTicketError",
    "PluginUiHttpTicketIssuer",
    "VerifiedPluginUiHttpTicket",
    "format_plugin_ui_http_expiry",
    "plugin_ui_http_request_sha256",
]

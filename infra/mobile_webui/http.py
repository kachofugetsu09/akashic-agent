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
from infra.mobile_webui.store import (
    MobileWebUiStore,
    ReleaseSelectionChangedError,
    ReleaseView,
    TargetResourceNotFoundError,
)

AUDIENCE = "mobile-webui-v1"
DEFAULT_TTL = timedelta(seconds=300)
MAX_RANGE_BYTES = 8 * 1024 * 1024


class WebUiTicketError(ValueError):
    """表示 WebUI HTTP ticket 不可接受。"""

    def __init__(self, message: str, *, code: str = "invalid_ticket") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class WebUiHttpGrant:
    ticket: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class VerifiedWebUiTicket:
    device_id: str
    connection_epoch: int
    target_key: str
    generation_id: str
    release_epoch: str
    manifest_digest: str
    selection_digest: str


class WebUiTicketIssuer:
    """签发并在每个 HTTP 请求重新核对设备、代际和发布选择。"""

    def __init__(
        self,
        keyset: LoadedKeyset,
        storage: MobileRealtimeStorage,
        publication: MobileWebUiStore,
        *,
        ttl: timedelta = DEFAULT_TTL,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        connection_checker: Callable[[str, int], bool] | None = None,
    ) -> None:
        if ttl <= timedelta(0) or ttl > DEFAULT_TTL:
            raise ValueError("WebUI ticket ttl 必须在 0..300 秒内")
        self._keyset = keyset
        self._storage = storage
        self._publication = publication
        self._ttl = ttl
        self._clock = clock
        self._connection_checker = connection_checker

    def issue(
        self,
        *,
        device_id: str,
        connection_epoch: int,
        release: ReleaseView,
        target_key: str,
    ) -> WebUiHttpGrant:
        target = release.target(target_key)
        if target is None:
            raise WebUiTicketError("target 不属于当前 ReleaseView", code="target_changed")
        now = self._now()
        expires_at = now + self._ttl
        claims = {
            "aud": AUDIENCE,
            "connection_epoch": connection_epoch,
            "device_id": device_id,
            "exp": _unix_seconds(expires_at),
            "iat": _unix_seconds(now),
            "manifest_digest": target.manifest_digest,
            "generation_id": target.generation_id,
            "release_epoch": release.release_epoch,
            "selection_digest": release.selection_digest,
            "server_id": self._keyset.manifest.server_id,
            "target_key": target.target_key,
            "v": 1,
        }
        payload = _canonical_json(claims)
        signature = self._keyset.identity_private_key.sign(payload, ec.ECDSA(hashes.SHA256()))
        return WebUiHttpGrant(
            ticket=f"{_b64url(payload)}.{_b64url(signature)}",
            expires_at=expires_at,
        )

    def verify(
        self,
        ticket: str,
        *,
        resource_kind: str,
        resource_digest: str,
    ) -> VerifiedWebUiTicket:
        """验签后校验当前连接、发布 epoch 和 target 成员关系。"""

        # 1. 验证签名、claims 结构和时钟窗口
        parts = ticket.split(".")
        if len(parts) != 2:
            raise WebUiTicketError("WebUI ticket 格式无效")
        payload = _decode_b64url(parts[0])
        signature = _decode_b64url(parts[1])
        try:
            self._keyset.identity_private_key.public_key().verify(signature, payload, ec.ECDSA(hashes.SHA256()))
        except InvalidSignature as error:
            raise WebUiTicketError("WebUI ticket 签名无效") from error
        claims = _decode_claims(payload)
        expected = {
            "aud", "connection_epoch", "device_id", "exp", "iat",
            "generation_id", "manifest_digest", "release_epoch", "selection_digest", "server_id", "target_key", "v",
        }
        if set(claims) != expected or claims["v"] != 1 or claims["aud"] != AUDIENCE:
            raise WebUiTicketError("WebUI ticket claims 无效")
        if claims["server_id"] != self._keyset.manifest.server_id:
            raise WebUiTicketError("WebUI ticket 不属于当前服务端")
        device_id = _require_text(claims["device_id"], "device_id")
        connection_epoch = _require_positive_int(claims["connection_epoch"], "connection_epoch")
        target_key = _require_text(claims["target_key"], "target_key")
        release_epoch = _require_text(claims["release_epoch"], "release_epoch")
        manifest_digest = _require_digest(claims["manifest_digest"], "manifest_digest")
        generation_id = _require_digest(claims["generation_id"], "generation_id")
        selection_digest = _require_digest(claims["selection_digest"], "selection_digest")
        issued_at = _require_nonnegative_int(claims["iat"], "iat")
        expires_at = _require_nonnegative_int(claims["exp"], "exp")
        now = _unix_seconds(self._now())
        if expires_at <= now or issued_at > now or expires_at - issued_at > int(self._ttl.total_seconds()) + 1:
            raise WebUiTicketError("WebUI ticket 已过期或 TTL 无效")

        # 2. 每个请求重读设备状态与当前连接代际
        device = self._storage.read_device(device_id)
        if device is None or device.revoked_at is not None:
            raise WebUiTicketError("设备不存在或已撤销")
        if self._connection_checker is not None and not self._connection_checker(device_id, connection_epoch):
            raise WebUiTicketError("WebUI ticket connection_epoch 已失效")

        # 3. 发布指针和请求 digest 都必须属于 ticket 绑定的 target
        release = self._publication.get_release_light()
        target = release.target(target_key)
        if release.release_epoch != release_epoch or release.selection_digest != selection_digest or target is None or target.manifest_digest != manifest_digest or target.generation_id != generation_id:
            raise WebUiTicketError("WebUI ticket 对应的 release 已变化", code="target_changed")
        if resource_kind == "manifest":
            if resource_digest != manifest_digest:
                raise WebUiTicketError("manifest digest 与 target 不匹配", code="resource_not_found")
        elif resource_kind == "blob":
            try:
                _ = self._publication.verify_target_resource(
                    target_key=target_key,
                    selection_digest=selection_digest,
                    resource_digest=resource_digest,
                )
            except TargetResourceNotFoundError as error:
                raise WebUiTicketError("blob 不属于当前 target", code="resource_not_found") from error
            except ReleaseSelectionChangedError as error:
                raise WebUiTicketError("release selection 已变化", code="target_changed") from error
        else:
            raise WebUiTicketError("未知 WebUI resource kind")
        return VerifiedWebUiTicket(
            device_id=device_id,
            connection_epoch=connection_epoch,
            target_key=target_key,
            generation_id=generation_id,
            release_epoch=release_epoch,
            manifest_digest=manifest_digest,
            selection_digest=selection_digest,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("WebUI ticket clock 必须带 timezone")
        return value.astimezone(timezone.utc)


def parse_single_range(header: str | None, total: int) -> tuple[int, int] | None:
    """解析一个有界 bytes Range；多段和不满足范围的请求明确失败。"""

    if header is None:
        return None
    if not header.startswith("bytes=") or "," in header:
        raise WebUiTicketError("Range 只支持单个 bytes range")
    value = header[6:].strip()
    if "-" not in value or total <= 0:
        raise WebUiTicketError("Range 无法满足")
    start_text, end_text = value.split("-", 1)
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError
            start = max(0, total - suffix)
            end = total - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else total - 1
            if start < 0 or end < start or start >= total:
                raise ValueError
            end = min(end, total - 1)
    except (TypeError, ValueError) as error:
        raise WebUiTicketError("Range 无法满足") from error
    if end - start + 1 > MAX_RANGE_BYTES:
        raise WebUiTicketError("Range 超过 8 MiB 上限")
    return start, end


def _canonical_json(value: dict[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decode_claims(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WebUiTicketError("WebUI ticket claims 不是 JSON") from error
    if not isinstance(value, dict):
        raise WebUiTicketError("WebUI ticket claims 必须是 object")
    return value


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_b64url(value: str) -> bytes:
    try:
        if not value or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for char in value):
            raise ValueError
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, binascii.Error) as error:
        raise WebUiTicketError("WebUI ticket base64 无效") from error


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or any(char.isspace() for char in value):
        raise WebUiTicketError(f"{label} 无效")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WebUiTicketError(f"{label} 无效")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WebUiTicketError(f"{label} 无效")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise WebUiTicketError(f"{label} 无效")
    return value


def _unix_seconds(value: datetime) -> int:
    return int(value.astimezone(timezone.utc).timestamp())


__all__ = [
    "AUDIENCE",
    "MAX_RANGE_BYTES",
    "VerifiedWebUiTicket",
    "WebUiHttpGrant",
    "WebUiTicketError",
    "WebUiTicketIssuer",
    "parse_single_range",
]

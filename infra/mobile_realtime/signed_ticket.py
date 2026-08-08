from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable
from datetime import datetime

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from infra.mobile_realtime.key_protection import LoadedKeyset

ErrorFactory = Callable[[str], ValueError]


class SignedTicketError(ValueError):
    """表示签名授权票据不可接受。"""


def canonical_json(
    value: object,
    *,
    label: str,
    error_factory: ErrorFactory = SignedTicketError,
    ensure_ascii: bool = True,
) -> bytes:
    """把 claims 序列化为定序字节，供签名与摘要使用。"""

    try:
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise error_factory(f"{label} 无法规范化") from error


def b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def decode_b64url(value: str, *, label: str, error_factory: ErrorFactory = SignedTicketError) -> bytes:
    if not value:
        raise error_factory(f"{label} 段不能为空")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as error:
        raise error_factory(f"{label} Base64URL 无效") from error
    if b64url(decoded) != value:
        raise error_factory(f"{label} Base64URL 无效")
    return decoded


def decode_claims(payload: bytes, *, label: str, error_factory: ErrorFactory = SignedTicketError) -> dict[str, object]:
    """严格解码 claims：拒绝重复字段与非标准常量。"""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise error_factory(f"{label} 字段重复: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise error_factory(f"{label} 包含非标准常量: {value}")

    try:
        claims = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise error_factory(f"{label} claims 无效") from error
    if not isinstance(claims, dict):
        raise error_factory(f"{label} claims 必须是对象")
    return claims


def unix_seconds(value: datetime) -> int:
    return int(value.timestamp())


def sign(
    keyset: LoadedKeyset,
    claims: dict[str, object],
    *,
    label: str,
    error_factory: ErrorFactory = SignedTicketError,
    ensure_ascii: bool = True,
) -> str:
    """规范化 claims、签名并拼装为 `payload.signature` 票据。"""

    payload = canonical_json(claims, label=label, error_factory=error_factory, ensure_ascii=ensure_ascii)
    signature = keyset.identity_private_key.sign(payload, ec.ECDSA(hashes.SHA256()))
    return f"{b64url(payload)}.{b64url(signature)}"


def verify_signature(
    ticket: str,
    keyset: LoadedKeyset,
    *,
    label: str,
    error_factory: ErrorFactory = SignedTicketError,
) -> dict[str, object]:
    """拆分、验签并严格解码票据 claims。"""

    parts = ticket.split(".")
    if len(parts) != 2:
        raise error_factory(f"{label} 格式无效")
    payload = decode_b64url(parts[0], label=label, error_factory=error_factory)
    signature = decode_b64url(parts[1], label=label, error_factory=error_factory)
    try:
        keyset.identity_private_key.public_key().verify(signature, payload, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as error:
        raise error_factory(f"{label} 签名无效") from error
    return decode_claims(payload, label=label, error_factory=error_factory)


def require_exact_keys(
    raw: dict[str, object],
    expected: set[str],
    *,
    label: str,
    error_factory: ErrorFactory = SignedTicketError,
) -> None:
    if set(raw) != expected:
        raise error_factory(f"{label} claims 字段无效")


def require_text(
    value: object,
    field: str,
    max_length: int,
    *,
    label: str,
    error_factory: ErrorFactory = SignedTicketError,
    reject_whitespace: bool = False,
) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= max_length:
        raise error_factory(f"{label} {field} 无效")
    if reject_whitespace and any(char.isspace() for char in value):
        raise error_factory(f"{label} {field} 无效")
    return value


def require_nonnegative_int(value: object, field: str, *, label: str, error_factory: ErrorFactory = SignedTicketError) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise error_factory(f"{label} {field} 无效")
    return value


def require_positive_int(value: object, field: str, *, label: str, error_factory: ErrorFactory = SignedTicketError) -> int:
    parsed = require_nonnegative_int(value, field, label=label, error_factory=error_factory)
    if parsed == 0:
        raise error_factory(f"{label} {field} 无效")
    return parsed


def require_digest(value: object, field: str, *, label: str, error_factory: ErrorFactory = SignedTicketError) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise error_factory(f"{label} {field} 无效")
    return value

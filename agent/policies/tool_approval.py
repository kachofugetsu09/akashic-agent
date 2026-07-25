from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

_SENSITIVE_KEY_PARTS = frozenset(
    {"token", "password", "secret", "api_key", "authorization", "cookie"}
)
_TEXT_HASH_KEYS = frozenset({"content", "command", "code", "body"})
_SAFE_INLINE_TEXT_KEYS = frozenset(
    {
        "path",
        "url",
        "source_path",
        "chunk_id",
        "selector",
        "description",
        "query",
        "tool_name",
    }
)
_MAX_INLINE_TEXT = 160
_MAX_SEQUENCE_ITEMS = 20


def canonical_args_hash(arguments: Mapping[str, Any]) -> str:
    encoded = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def summarize_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): summarize_argument_value(str(key), value)
        for key, value in sorted(arguments.items(), key=lambda item: str(item[0]))
    }


def summarize_argument_value(key: str, value: Any) -> object:
    lower_key = key.lower()
    if _is_sensitive_key(lower_key):
        return {"redacted": True}
    if isinstance(value, Mapping):
        return {
            str(child_key): summarize_argument_value(str(child_key), child_value)
            for child_key, child_value in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [
            summarize_argument_value(key, item) for item in value[:_MAX_SEQUENCE_ITEMS]
        ]
    if isinstance(value, str):
        if lower_key in _TEXT_HASH_KEYS or lower_key not in _SAFE_INLINE_TEXT_KEYS:
            return {
                "kind": "text",
                "length": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
        return value[:_MAX_INLINE_TEXT]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return {
        "kind": type(value).__name__,
        "sha256": hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def build_approval_payload(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    action: str,
    reason: str,
    risk: str,
    approval_scope: str,
    approval_request_id: str = "",
    expires_at: str = "",
) -> dict[str, Any]:
    args_hash = canonical_args_hash(arguments)
    args_summary = summarize_arguments(arguments)
    approval_request: dict[str, Any] = {
        "tool_name": tool_name,
        "risk": risk,
        "reason": reason,
        "approval_scope": approval_scope or "tool_call",
        "required_user_action": "approve_or_deny",
        "args_hash": args_hash,
        "args_summary": args_summary,
    }
    if approval_request_id:
        approval_request["approval_request_id"] = approval_request_id
    if expires_at:
        approval_request["expires_at"] = expires_at
    return {
        "ok": False,
        "blocked": True,
        "deferred": action == "defer",
        "error_code": reason,
        "message": "工具调用需要用户授权后才能执行。",
        "invoker_reached": False,
        "approval_request": approval_request,
    }


def _is_sensitive_key(lower_key: str) -> bool:
    return any(part in lower_key for part in _SENSITIVE_KEY_PARTS)

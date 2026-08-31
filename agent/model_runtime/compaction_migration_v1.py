"""Frozen persisted compaction identity used by historical migrations."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime


def normalize_session_created_at(created_at: datetime | str) -> str:
    """Normalize one persisted Session incarnation timestamp."""

    if isinstance(created_at, datetime):
        value = created_at
    elif isinstance(created_at, str):
        try:
            value = datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise ValueError("session created_at 必须是 ISO-8601 时间") from exc
    else:
        raise TypeError("session created_at 必须是 datetime 或 ISO-8601 字符串")
    if value.tzinfo is None:
        raise ValueError("session created_at 必须包含时区")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def compaction_scope_id(session_key: str, created_at: datetime | str) -> str:
    """Return the v1 Session-incarnation scope identity."""

    if not isinstance(session_key, str) or not session_key:
        raise ValueError("session key 不能为空")
    normalized = normalize_session_created_at(created_at)
    digest = hashlib.sha256(f"{session_key}\0{normalized}".encode()).hexdigest()
    return f"{session_key}@{digest[:16]}"


def compaction_source_ref(scope_id: str, generation: int) -> str:
    """Return the v1 persisted source reference."""

    digest = hashlib.sha256(f"{scope_id}\0{generation}".encode()).hexdigest()
    return f"context-compaction:{scope_id}:{generation}:{digest[:16]}"


__all__ = [
    "compaction_scope_id",
    "compaction_source_ref",
    "normalize_session_created_at",
]

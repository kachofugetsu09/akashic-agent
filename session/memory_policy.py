"""Decode legacy session-level memory exclusions during Akasha rebuilds only."""

from __future__ import annotations

from typing import Any

_SKIP_MEMORY_KEY = "skip_post_memory"
_SCHEDULER_PREFIX = "scheduler:"


def validate_session_memory_metadata(metadata: dict[str, Any]) -> None:
    """Validate a persisted legacy skip_post_memory value.

    非 boolean（如字符串 "false"）一律 fail-loud，防止宽松判断把假值当作排除。
    """
    if _SKIP_MEMORY_KEY not in metadata:
        return
    value = metadata[_SKIP_MEMORY_KEY]
    if not isinstance(value, bool):
        raise ValueError(
            f"session metadata {_SKIP_MEMORY_KEY} 必须是 boolean，收到 {value!r}"
        )


def legacy_excludes_memory(session_key: str, metadata: dict[str, Any]) -> bool:
    """Read the historical scheduler/session exclusion contract for replay only.

    命中条件：scheduler 前缀（内置排除，定时任务 session）或
    sessions.metadata["skip_post_memory"] is True（显式标记，严格 boolean）。
    读取已有 metadata 时同样执行严格校验，损坏数据 fail-loud。
    """
    validate_session_memory_metadata(metadata)
    if session_key.startswith(_SCHEDULER_PREFIX):
        return True
    return metadata.get(_SKIP_MEMORY_KEY) is True

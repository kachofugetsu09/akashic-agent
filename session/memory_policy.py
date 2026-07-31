"""session 级记忆排除策略：唯一权威谓词与写入校验。"""

from __future__ import annotations

from typing import Any

_SKIP_MEMORY_KEY = "skip_post_memory"
_SCHEDULER_PREFIX = "scheduler:"


def validate_session_memory_metadata(metadata: dict[str, Any]) -> None:
    """写入口校验：sessions.metadata 中 skip_post_memory 必须是 JSON boolean。

    非 boolean（如字符串 "false"）一律 fail-loud，防止宽松判断把假值当作排除。
    """
    if _SKIP_MEMORY_KEY not in metadata:
        return
    value = metadata[_SKIP_MEMORY_KEY]
    if not isinstance(value, bool):
        raise ValueError(
            f"session metadata {_SKIP_MEMORY_KEY} 必须是 boolean，收到 {value!r}"
        )


def excludes_memory(session_key: str, metadata: dict[str, Any]) -> bool:
    """session 级记忆排除统一谓词。

    命中条件：scheduler 前缀（内置排除，定时任务 session）或
    sessions.metadata["skip_post_memory"] is True（显式标记，严格 boolean）。
    读取已有 metadata 时同样执行严格校验，损坏数据 fail-loud。
    """
    validate_session_memory_metadata(metadata)
    if session_key.startswith(_SCHEDULER_PREFIX):
        return True
    return metadata.get(_SKIP_MEMORY_KEY) is True

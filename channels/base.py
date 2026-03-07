from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from session.manager import SessionManager

_DEFAULT_UPLOAD_DIR = Path.home() / ".akasic" / "workspace" / "uploads"


class AttachmentStore:
    """为 channel 媒体文件提供统一的持久化落盘目录。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or _DEFAULT_UPLOAD_DIR

    def create_path(self, prefix: str, suffix: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / f"{prefix}{uuid4().hex}{suffix}"

    def write_bytes(self, data: bytes, *, prefix: str, suffix: str) -> Path:
        path = self.create_path(prefix, suffix)
        path.write_bytes(data)
        return path


class SessionIdentityIndex:
    """维护 identity -> chat_id 的索引，并同步写入 session metadata。"""

    def __init__(
        self,
        session_manager: SessionManager,
        *,
        channel: str,
        metadata_key: str,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self._session_manager = session_manager
        self._channel = channel
        self._metadata_key = metadata_key
        self._normalizer = normalizer or (lambda value: value)
        self.mapping: dict[str, str] = {}

    def rebuild(self) -> dict[str, str]:
        self.mapping.clear()
        for entry in self._session_manager.get_channel_metadata(self._channel):
            raw_value = entry["metadata"].get(self._metadata_key)
            if not isinstance(raw_value, str):
                continue
            normalized = self._normalize(raw_value)
            if normalized:
                self.mapping[normalized] = entry["chat_id"]
        return dict(self.mapping)

    def resolve(self, identity: str) -> str | None:
        normalized = self._normalize(identity)
        if not normalized:
            return None
        return self.mapping.get(normalized)

    async def remember(self, identity: str, chat_id: str) -> None:
        normalized = self._normalize(identity)
        if not normalized:
            return
        self.mapping[normalized] = chat_id
        session = self._session_manager.get_or_create(f"{self._channel}:{chat_id}")
        if session.metadata.get(self._metadata_key) == normalized:
            return
        session.metadata[self._metadata_key] = normalized
        await self._session_manager.save_async(session)

    def _normalize(self, value: str) -> str:
        return self._normalizer((value or "").strip())


class MessageDeduper:
    """滑动窗口去重，避免 channel 重投或重复事件被处理多次。"""

    def __init__(self, max_size: int) -> None:
        self._max_size = max(1, max_size)
        self._seen: set[str] = set()
        self._order: deque[str] = deque()

    def seen(self, key: str) -> bool:
        if key in self._seen:
            return True
        self._seen.add(key)
        self._order.append(key)
        while len(self._order) > self._max_size:
            self._seen.discard(self._order.popleft())
        return False

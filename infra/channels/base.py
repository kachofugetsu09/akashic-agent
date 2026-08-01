from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Callable
from uuid import uuid4

from session.manager import SessionManager

_MISSING_METADATA = object()


class AttachmentStore:
    """为 channel 媒体文件提供统一的持久化落盘目录。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _resolve_root(self) -> Path:
        if self.root.is_symlink():
            raise ValueError(f"附件目录不能是符号链接: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError(f"附件目录不能是符号链接: {self.root}")
        if not os.access(self.root, os.W_OK):
            raise PermissionError(f"附件目录不可写: {self.root}")
        return self.root

    def create_path(self, prefix: str, suffix: str) -> Path:
        root = self._resolve_root()
        return root / f"{prefix}{uuid4().hex}{suffix}"

    def create_persistent_path(self, prefix: str, suffix: str) -> Path:
        """只在配置的持久目录中分配路径，目录不可用时直接失败。"""

        root = self._resolve_root()
        return root / f"{prefix}{uuid4().hex}{suffix}"

    def write_bytes(self, data: bytes, *, prefix: str, suffix: str) -> Path:
        staging = self.create_staging_path(prefix=f".{prefix}", suffix=".part")
        try:
            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            return self.publish_staging(staging, prefix=prefix, suffix=suffix)
        except BaseException as exc:
            try:
                staging.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise BaseExceptionGroup(
                    "附件 staging 清理失败",
                    [exc, cleanup_error],
                ) from exc
            raise

    def create_staging_path(self, *, prefix: str = ".upload-", suffix: str = ".part") -> Path:
        """在附件目录创建一个仅供当前写入的 staging 路径。"""

        root = self._resolve_root()
        return root / f"{prefix}{uuid4().hex}{suffix}"

    def publish_staging(self, staging: Path, *, prefix: str, suffix: str) -> Path:
        """fsync 后原子发布 staging 文件，目录不一致时直接失败。"""

        root = self._resolve_root().resolve()
        staging_path = staging.resolve()
        if staging_path.parent != root:
            raise ValueError(f"staging 文件不在附件目录: {staging}")
        path = root / f"{prefix}{uuid4().hex}{suffix}"
        os.replace(staging_path, path)
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
        session = self._session_manager.get_or_create(f"{self._channel}:{chat_id}")
        if session.metadata.get(self._metadata_key) == normalized:
            self.mapping[normalized] = chat_id
            return
        previous = session.metadata.get(self._metadata_key, _MISSING_METADATA)
        session.metadata[self._metadata_key] = normalized
        try:
            await self._session_manager.save_async(session)
        except BaseException:
            if previous is _MISSING_METADATA:
                session.metadata.pop(self._metadata_key, None)
            else:
                session.metadata[self._metadata_key] = previous
            raise
        self.mapping[normalized] = chat_id

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

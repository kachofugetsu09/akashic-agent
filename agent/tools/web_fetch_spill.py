"""web_fetch 的 execution-owned 临时响应文件。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4


INLINE_MAX_BYTES = 5 * 1024 * 1024
SPILL_MAX_FILE_BYTES = 20 * 1024 * 1024
SPILL_MAX_TOTAL_BYTES = 64 * 1024 * 1024


class SpillLimitExceeded(RuntimeError):
    """响应超出单文件或 execution spill 总量上限。"""


@dataclass
class SpillHandle:
    execution_id: str
    path: Path
    _file: object
    size: int = 0
    finalized: bool = False


@dataclass
class SpillCleanup:
    execution_id: str
    released: bool
    status: str
    path: str | None = None
    error: str | None = None


@dataclass
class WebFetchSpillStore:
    """把单次执行的大响应放在私有临时目录并保留可重试清理诊断。"""

    root: Path | None = None
    max_file_bytes: int = SPILL_MAX_FILE_BYTES
    max_total_bytes: int = SPILL_MAX_TOTAL_BYTES
    _handles: dict[str, SpillHandle] = field(default_factory=dict, init=False)
    _total_bytes: int = field(default=0, init=False)
    _diagnostics: dict[str, SpillCleanup] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.max_file_bytes <= 0 or self.max_total_bytes <= 0:
            raise ValueError("spill limit must be positive")
        if self.root is None:
            self.root = Path(tempfile.mkdtemp(prefix="akashic-web-fetch-"))
            os.chmod(self.root, 0o700)
        else:
            self.root = self.root.expanduser()
            if self.root.exists() and self.root.is_symlink():
                raise ValueError(f"spill root cannot be a symlink: {self.root}")
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)

    def open(self, execution_id: str) -> SpillHandle:
        """为 execution 分配唯一私有文件；重复 owner 直接拒绝。"""

        owner = str(execution_id or "").strip()
        if not owner:
            raise ValueError("execution_id is required")
        if owner in self._handles:
            raise ValueError(f"spill already exists for execution: {owner}")
        assert self.root is not None
        path = self.root / f"{uuid4().hex}.spill"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        handle = SpillHandle(owner, path, os.fdopen(fd, "wb"))
        self._handles[owner] = handle
        return handle

    def write(self, handle: SpillHandle, chunk: bytes) -> None:
        """在写入前检查单文件与 execution 总量上限。"""

        if handle.finalized:
            raise ValueError("spill handle is finalized")
        if not isinstance(chunk, bytes):
            raise TypeError("spill chunk must be bytes")
        next_size = handle.size + len(chunk)
        next_total = self._total_bytes + len(chunk)
        if next_size > self.max_file_bytes:
            raise SpillLimitExceeded("spill file limit exceeded")
        if next_total > self.max_total_bytes:
            raise SpillLimitExceeded("spill total limit exceeded")
        handle._file.write(chunk)
        handle.size = next_size
        self._total_bytes = next_total

    def finalize(self, handle: SpillHandle) -> None:
        """fsync 并关闭 spill 文件，返回前确保文件内容已经提交。"""

        if handle.finalized:
            return
        handle._file.flush()
        os.fsync(handle._file.fileno())
        handle._file.close()
        handle.finalized = True

    def release(self, execution_id: str) -> SpillCleanup:
        """显式释放 execution 文件；失败时保留 owner/path 诊断而不抛错。"""

        owner = str(execution_id or "").strip()
        handle = self._handles.get(owner)
        if handle is None:
            return self._diagnostics.get(
                owner,
                SpillCleanup(owner, released=True, status="already_released"),
            )
        try:
            if not handle.finalized:
                handle._file.close()
                handle.finalized = True
            handle.path.unlink(missing_ok=True)
        except OSError as exc:
            diagnostic = SpillCleanup(
                owner,
                released=False,
                status="cleanup_degraded",
                path=str(handle.path),
                error=str(exc),
            )
            self._diagnostics[owner] = diagnostic
            return diagnostic
        self._total_bytes -= handle.size
        self._handles.pop(owner, None)
        diagnostic = SpillCleanup(
            owner,
            released=True,
            status="released",
            path=str(handle.path),
        )
        self._diagnostics[owner] = diagnostic
        return diagnostic

    def diagnostics(self, execution_id: str) -> SpillCleanup | None:
        return self._diagnostics.get(str(execution_id or "").strip())

from __future__ import annotations

import os
from pathlib import Path
from typing import IO


class WorkspaceInstanceLock:
    """保证一个 workspace 同时只有一个 runtime owner。"""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace / ".instance.lock"
        self._stream: IO[str] | None = None

    def acquire(self) -> None:
        """非阻塞获取进程锁；冲突时保留 owner 信息并明确失败。"""

        # 1. 锁文件本身可持久存在，内核 flock 才是 owner 真相。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.seek(0)
            owner = stream.read().strip() or "unknown"
            stream.close()
            raise RuntimeError(
                f"workspace 已由其他 runtime 占用: {self.path} owner={owner}"
            ) from exc

        # 2. 获取后刷新诊断 owner，不把文件存在误当成锁。
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        self._stream = stream

    def release(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

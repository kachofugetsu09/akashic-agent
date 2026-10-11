from __future__ import annotations

import os
from pathlib import Path
from typing import IO


class _FileLock:
    """独占一个诊断锁文件，内核锁和文件句柄由同一对象持有。"""

    def __init__(self, path: Path, conflict: str) -> None:
        self.path = path
        self._conflict = conflict
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
                f"{self._conflict}: {self.path} owner={owner}"
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


class WorkspaceInstanceLock(_FileLock):
    """保证一个 workspace 同时只有一个 runtime owner。"""

    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace / ".instance.lock", "workspace 已由其他 runtime 占用")


class PluginPublicationLock(_FileLock):
    """保证一个 plugin-home 同时只有一个发布或消费 owner。"""

    def __init__(self, plugins_home: Path) -> None:
        super().__init__(plugins_home / ".publication.lock", "plugin-home 已有发布或消费 owner")


class WorkspaceMaintenanceLock:
    """阻止 Supervisor 与 Runtime 在离线维护期间取得 workspace。"""

    def __init__(self, workspace: Path) -> None:
        self.paths = (
            workspace / ".supervisor.lock",
            workspace / ".instance.lock",
        )
        self._streams: list[IO[str]] = []

    def acquire(self) -> None:
        """按 Supervisor→Runtime 顺序非阻塞取得两个生命周期锁。"""

        if os.name == "nt":
            raise RuntimeError("离线插件批量安装只支持 Linux 和 macOS")
        import fcntl

        # 1. 与正式启动使用同一锁顺序，任一 owner 存活都立即拒绝。
        try:
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(
                        stream.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except OSError as exc:
                    stream.seek(0)
                    owner = stream.read().strip() or "unknown"
                    stream.close()
                    raise RuntimeError(
                        f"workspace 仍有生命周期 owner: {path} owner={owner}"
                    ) from exc
                stream.seek(0)
                stream.truncate()
                stream.write(f"maintenance:{os.getpid()}")
                stream.flush()
                self._streams.append(stream)
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        import fcntl

        # 2. 逆序释放，不删除可复用的诊断锁文件。
        while self._streams:
            stream = self._streams.pop()
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()

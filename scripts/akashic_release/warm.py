from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from utils.timing import measure

_SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")
_RECENT_SECONDS = 7 * 24 * 3600
_CHUNK = 8 << 20
WARM_TIMEOUT_SECONDS = 120.0


# 活跃库只认 Core 与插件数据目录的顶层文件；备份与重建副本在子目录里，不进入预热。
def live_databases(workspace: Path, *, now: float | None = None) -> list[Path]:
    cutoff = (time.time() if now is None else now) - _RECENT_SECONDS
    plugin_data = workspace / "plugin-data"
    roots = [workspace, workspace / "memory", workspace / "data"]
    if plugin_data.is_dir():
        roots.extend(sorted(path for path in plugin_data.iterdir() if path.is_dir()))
    found: list[tuple[float, Path]] = []
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for path in root.iterdir():
            if not path.name.endswith(_SQLITE_SUFFIXES) or path.is_symlink() or not path.is_file():
                continue
            # 只看主文件：任何进程只读打开旧库都会刷新 -wal 的 mtime。
            modified = path.stat().st_mtime
            if modified >= cutoff:
                wal = path.with_name(path.name + "-wal")
                found.extend((modified, item) for item in (path, wal) if item.is_file())
    # 最近写入的先读：预算不够时优先保住会话日志这类开机必读的库。
    return [path for _, path in sorted(found, key=lambda row: row[0], reverse=True)]


def _available_memory() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


# 停机前把新 Core 开机要读的活跃 SQLite 顺序读进页缓存；机械盘上冷启动的随机读要几十秒。
class StateCacheWarmer:
    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="release-cache-warm", daemon=True)
            self._thread.start()

    # 预热只是加速：到时限仍未读完就照常停机，读取线程随进程结束。
    def wait(self, timeout: float = WARM_TIMEOUT_SECONDS) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        with measure("release.warm") as result:
            budget = _available_memory() // 2
            warmed = 0
            for path in live_databases(self._workspace):
                if warmed >= budget:
                    break
                warmed += _read_through(path, budget - warmed)
            result.update(bytes=warmed, budget=budget)


def _read_through(path: Path, limit: int) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    total = 0
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        buffer = bytearray(_CHUNK)
        while total < limit:
            count = os.readv(descriptor, [buffer])
            if count == 0:
                break
            total += count
    except OSError:
        pass
    finally:
        os.close(descriptor)
    return total

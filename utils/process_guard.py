"""跨 Linux 和 macOS 的稳定进程等待与信号边界。"""

from __future__ import annotations

import os
import signal
import sys
from dataclasses import dataclass

from utils.pidfd import open_pidfd, send_pidfd_signal

_FALLBACK_POLL_INTERVAL_SECONDS = 0.1


@dataclass(frozen=True)
class ProcessRef:
    """Linux 持有 pidfd；macOS 保留 PID 并要求调用方校验身份。"""

    pid: int
    wait_fd: int | None

    @property
    def stable(self) -> bool:
        return self.wait_fd is not None

    def close(self) -> None:
        if self.wait_fd is not None:
            os.close(self.wait_fd)


def open_process_ref(pid: int) -> ProcessRef:
    if sys.platform.startswith("linux"):
        return ProcessRef(pid, open_pidfd(pid))
    if sys.platform == "darwin":
        return ProcessRef(pid, None)
    raise RuntimeError("进程引用仅支持 Linux 和 macOS")


def process_wait_timeout(ref: ProcessRef, timeout: float | None) -> float | None:
    """无可等待进程 fd 时限制 selector 阻塞时间，以便轮询 Popen。"""

    if ref.wait_fd is not None:
        return timeout
    if timeout is None:
        return _FALLBACK_POLL_INTERVAL_SECONDS
    return min(max(0.0, timeout), _FALLBACK_POLL_INTERVAL_SECONDS)


def signal_process_ref(ref: ProcessRef, sig: signal.Signals) -> None:
    if ref.wait_fd is not None:
        send_pidfd_signal(ref.wait_fd, sig)
        return
    os.kill(ref.pid, sig)

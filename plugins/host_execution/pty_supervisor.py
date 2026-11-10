"""交互终端的监管进程：作为 shell 的父进程，负责整个终端会话的结束。

以脚本运行（``python -I pty_supervisor.py SHELL [ARGS...]``），标准输入是 PTY 从端，只依赖标准库。
shell 退出后先不回收：未回收的 shell 占着自己的 PID，会话号就不会被复用，
这段时间里按会话号找到的进程一定属于这个终端。清理完会话里剩下的进程后才回收 shell，
并以 shell 的退出码退出。服务端只向本进程发 SIGTERM 请求关闭。
"""
from __future__ import annotations

from collections.abc import Callable
import ctypes
import os
from pathlib import Path
import signal
import sys
import time

_GRACE_S = 2.0
_POLL_S = 0.05
_SYS_PIDFD_SEND_SIGNAL = 424
_SYS_PIDFD_OPEN = 434
_LIBC = ctypes.CDLL(None, use_errno=True)


_WATCHED = {signal.SIGTERM, signal.SIGHUP, signal.SIGCHLD}


def _session_members(sid: int) -> list[int]:
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    members: list[int] = []
    for entry in proc.iterdir():
        if entry.name.isdigit() and int(entry.name) != sid and _session_of(int(entry.name)) == sid:
            members.append(int(entry.name))
    return members


# 已退出未回收的进程不再算成员；读不到时视为已不存在。
def _session_of(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
    except OSError:
        return None
    if len(fields) < 4 or fields[0] in {"Z", "X"}:
        return None
    return int(fields[3])


# 先用 pidfd 固定进程身份，再核对会话号，扫描与发信号之间 PID 被复用也不会误伤。
def _signal_member(pid: int, sid: int, sig: signal.Signals) -> None:
    fd = int(_LIBC.syscall(_SYS_PIDFD_OPEN, pid, 0))
    if fd < 0:
        if ctypes.get_errno() == 38 and _session_of(pid) == sid:  # ENOSYS：旧内核只能直接发
            _kill(pid, sig)
        return
    try:
        if _session_of(pid) == sid:
            _ = _LIBC.syscall(_SYS_PIDFD_SEND_SIGNAL, fd, int(sig), None, 0)
    finally:
        os.close(fd)


def _kill(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _signal_session(shell: int, sig: signal.Signals) -> None:
    # shell 未回收，它的 PID 与进程组号都还归这个会话；没有 /proc 的平台只能挂断 shell 的进程组。
    _kill(shell, sig)
    if not Path("/proc").is_dir():
        try:
            os.killpg(shell, sig)
        except ProcessLookupError:
            pass
    for pid in _session_members(shell):
        _signal_member(pid, shell, sig)


def _exited(shell: int) -> bool:
    return os.waitid(os.P_PID, shell, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _wait(condition: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_S)
    return True


# 关闭等同关掉终端窗口：挂断整个会话，宽限后强制结束；shell 自己退出时会话里剩下的进程同样处理。
def _finish(shell: int, closing: bool) -> int:
    # 1. 关闭请求：先让 shell 一族收到挂断，不响应的再结束。
    if closing and not _wait(lambda: _exited(shell), 0):
        _signal_session(shell, signal.SIGHUP)
        if not _wait(lambda: _exited(shell), _GRACE_S):
            _signal_session(shell, signal.SIGKILL)
    _ = os.waitid(os.P_PID, shell, os.WEXITED | os.WNOWAIT)
    # 2. shell 已退出但还占着会话号：后台作业与 shell 同生同死。
    if _session_members(shell):
        _signal_session(shell, signal.SIGHUP)
        if not _wait(lambda: not _session_members(shell), _GRACE_S):
            _signal_session(shell, signal.SIGKILL)
            _ = _wait(lambda: not _session_members(shell), _GRACE_S)
    # 3. 最后才回收 shell，退出码原样交给服务端。
    code = os.waitstatus_to_exitcode(os.waitpid(shell, 0)[1])
    return code if code >= 0 else 128 - code


# 等 shell 退出或关闭请求；信号在 fork 前就已屏蔽，这里同步取出，不会打断清理。
def _wait_for_end(shell: int) -> bool:
    while True:
        if _exited(shell):
            return False
        info = signal.sigwaitinfo(_WATCHED)
        if info.si_signo != signal.SIGCHLD:
            return True


# Python 启动时忽略的 SIGPIPE、SIGXFSZ 会跨 exec 保留；还原后 `yes | head` 这类管道照常静默结束。
def _exec_shell(argv: list[str]) -> None:
    for signum in (signal.SIGPIPE, signal.SIGXFSZ):
        _ = signal.signal(signum, signal.SIG_DFL)
    os.login_tty(0)
    os.execv(argv[0], argv)


def main(argv: list[str]) -> int:
    # 1. 没有 waitid 或 sigwaitinfo 的平台无法在回收前固定会话号，退化为直接成为 shell。
    if not hasattr(os, "waitid") or not hasattr(signal, "sigwaitinfo"):
        _exec_shell(argv)
    _ = signal.pthread_sigmask(signal.SIG_BLOCK, _WATCHED)
    shell = os.fork()
    if shell == 0:
        _ = signal.pthread_sigmask(signal.SIG_UNBLOCK, _WATCHED)
        _exec_shell(argv)
    # 2. 监管进程放下 PTY，否则会话结束后主端读不到 EOF。
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        _ = os.dup2(null, fd)
    os.close(null)
    _ = signal.signal(signal.SIGINT, signal.SIG_IGN)
    # 3. 收尾期间再来的关闭请求留在屏蔽集里，不影响清理。
    return _finish(shell, _wait_for_end(shell))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

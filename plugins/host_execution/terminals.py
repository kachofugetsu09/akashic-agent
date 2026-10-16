"""用户交互终端：执行主机上的长期 PTY 会话，输出留在环形缓冲里按偏移重复读取。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import errno
import fcntl
import os
from pathlib import Path
import pwd
import signal
import struct
import sys
import termios

from plugins.host_execution.contract import (
    TERMINAL_MAX_COLS, TERMINAL_MAX_ROWS, TERMINAL_MAX_WAIT_MS, TERMINAL_MAX_WRITE_BYTES,
    TerminalInfo, TerminalOutput, UnknownTerminalError,
)
from .process_runtime import _read_fd, _write_fd
from .shell_runtime import resolve_shell

MAX_TERMINALS_PER_OWNER = 8
MAX_TERMINALS = 32
BUFFER_BYTES = 256 * 1024
READ_CHUNK_BYTES = 64 * 1024
_CLOSE_GRACE_S = 2.0
# 交互 shell 需要控制终端才有作业控制与 Ctrl+C；子进程先成为会话首进程再接管 PTY。
_TRAMPOLINE = "import os,sys;os.login_tty(0);os.execv(sys.argv[1],sys.argv[1:])"
_LOGIN_SHELLS = {"bash", "zsh", "fish", "sh", "dash", "ksh"}


@dataclass
class _Terminal:
    terminal_id: int
    owner: str
    shell: str
    cwd: str
    process: asyncio.subprocess.Process
    master_fd: int
    buffer: bytearray = field(default_factory=bytearray)
    base: int = 0
    exit_code: int | None = None
    exited: bool = False
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: list[asyncio.Task[None]] = field(default_factory=lambda: list[asyncio.Task[None]]())

    def info(self) -> TerminalInfo:
        return TerminalInfo(self.terminal_id, self.shell, self.cwd, self.exit_code, self.exited)

    def append(self, chunk: bytes) -> None:
        # 超出上限时从下一个换行处截断，回放时少出现半截转义序列。
        self.buffer += chunk
        excess = len(self.buffer) - BUFFER_BYTES
        if excess > 0:
            newline = self.buffer.find(b"\n", excess, excess + 4096)
            cut = excess if newline < 0 else newline + 1
            del self.buffer[:cut]
            self.base += cut
        self.notify()

    def notify(self) -> None:
        self.changed.set()
        self.changed = asyncio.Event()

    def slice(self, offset: int) -> TerminalOutput:
        start = min(max(offset, self.base), self.base + len(self.buffer))
        index = start - self.base
        data = bytes(self.buffer[index:index + READ_CHUNK_BYTES])
        done = self.exited and index + len(data) >= len(self.buffer)
        return TerminalOutput(data, start, self.exit_code if done else None, done)


class PtyTerminals:
    """本机 PTY 终端集合；owner 只能看到和操作自己的终端。"""

    def __init__(self) -> None:
        self._terminals: dict[int, _Terminal] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._closed = False

    async def open_terminal(
        self, *, owner_session_key: str, cwd: Path | None, cols: int, rows: int,
    ) -> TerminalInfo:
        _check_size(cols, rows)
        async with self._lock:
            # 1. 关闭后与容量满时拒绝，计数包含已退出但未关闭的终端。
            if self._closed:
                raise RuntimeError("终端服务已关闭")
            owned = sum(1 for item in self._terminals.values() if item.owner == owner_session_key)
            if owned >= MAX_TERMINALS_PER_OWNER or len(self._terminals) >= MAX_TERMINALS:
                raise RuntimeError("终端数量已达上限，请先关闭不用的终端")
            # 2. 登记后再启动读写任务，关闭路径总能找到它。
            terminal = await _spawn(self._next_id, owner_session_key, cwd, cols, rows)
            self._next_id += 1
            self._terminals[terminal.terminal_id] = terminal
        terminal.tasks.append(asyncio.create_task(_pump(terminal), name=f"terminal-{terminal.terminal_id}"))
        return terminal.info()

    async def write_terminal(self, *, owner_session_key: str, terminal_id: int, data: bytes) -> None:
        if len(data) > TERMINAL_MAX_WRITE_BYTES:
            raise ValueError("单次终端输入过长")
        terminal = self._owned(owner_session_key, terminal_id)
        if not terminal.exited and data:
            await _write_fd(terminal.master_fd, data)

    async def resize_terminal(
        self, *, owner_session_key: str, terminal_id: int, cols: int, rows: int,
    ) -> None:
        _check_size(cols, rows)
        terminal = self._owned(owner_session_key, terminal_id)
        if not terminal.exited:
            _set_size(terminal.master_fd, cols, rows)

    async def read_terminal(
        self, *, owner_session_key: str, terminal_id: int, offset: int, wait_ms: int,
    ) -> TerminalOutput:
        if offset < 0 or not 0 <= wait_ms <= TERMINAL_MAX_WAIT_MS:
            raise ValueError("终端读取参数越界")
        terminal = self._owned(owner_session_key, terminal_id)
        output = terminal.slice(offset)
        if output.data or output.exited or wait_ms == 0:
            return output
        # 没有新输出时等到输出、退出或超时，调用方据此做长轮询。
        try:
            _ = await asyncio.wait_for(terminal.changed.wait(), wait_ms / 1000)
        except TimeoutError:
            pass
        return terminal.slice(offset)

    async def close_terminal(self, *, owner_session_key: str, terminal_id: int) -> bool:
        async with self._lock:
            terminal = self._terminals.get(terminal_id)
            if terminal is None or terminal.owner != owner_session_key:
                return False
            del self._terminals[terminal_id]
        await _close(terminal)
        return True

    async def list_terminals(self, owner_session_key: str) -> list[TerminalInfo]:
        return [
            item.info() for item in sorted(self._terminals.values(), key=lambda item: item.terminal_id)
            if item.owner == owner_session_key
        ]

    async def close_owner(self, owner_session_key: str) -> None:
        async with self._lock:
            owned = [item for item in self._terminals.values() if item.owner == owner_session_key]
            for item in owned:
                del self._terminals[item.terminal_id]
        _ = await asyncio.gather(*(_close(item) for item in owned))

    async def shutdown(self) -> None:
        async with self._lock:
            self._closed = True
            owned = list(self._terminals.values())
            self._terminals.clear()
        _ = await asyncio.gather(*(_close(item) for item in owned))

    def _owned(self, owner: str, terminal_id: int) -> _Terminal:
        terminal = self._terminals.get(terminal_id)
        if terminal is None or terminal.owner != owner:
            raise UnknownTerminalError(f"终端不存在: {terminal_id}")
        return terminal


def _check_size(cols: int, rows: int) -> None:
    if not (1 <= cols <= TERMINAL_MAX_COLS and 1 <= rows <= TERMINAL_MAX_ROWS):
        raise ValueError("终端尺寸越界")


def _set_size(fd: int, cols: int, rows: int) -> None:
    _ = fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


# 用户在终端里用自己的登录 shell；不受 Agent shell 支持范围限制。
def _user_shell() -> str:
    for candidate in (os.environ.get("SHELL"), _passwd_shell()):
        if candidate and os.path.isabs(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return str(resolve_shell().path)


def _passwd_shell() -> str | None:
    try:
        return pwd.getpwuid(os.getuid()).pw_shell
    except KeyError:
        return None


def _terminal_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(TERM="xterm-256color", COLORTERM="truecolor", TERM_PROGRAM="akashic")
    for name in ("NO_COLOR", "PAGER", "GIT_PAGER", "GH_PAGER", "TERMINFO", "TERMINFO_DIRS"):
        _ = env.pop(name, None)
    return env


async def _spawn(terminal_id: int, owner: str, cwd: Path | None, cols: int, rows: int) -> _Terminal:
    # 1. 目录不存在时退回主目录，终端照常可用。
    directory = cwd if cwd is not None and cwd.is_dir() else Path.home()
    shell = _user_shell()
    argv = [shell, "-l"] if Path(shell).name in _LOGIN_SHELLS else [shell]
    # 2. 先设好尺寸再启动，shell 首屏就按真实列宽排版。
    master_fd, slave_fd = os.openpty()
    try:
        _set_size(slave_fd, cols, rows)
        os.set_blocking(master_fd, False)
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", _TRAMPOLINE, *argv,
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            cwd=str(directory), env=_terminal_env(),
        )
    except BaseException:
        os.close(master_fd)
        raise
    finally:
        os.close(slave_fd)
    return _Terminal(terminal_id, owner, shell, str(directory), process, master_fd)


# 读到 EOF 后回收退出码；shell 退出而后台子进程仍占着 PTY 时，宽限后结束整个会话。
async def _pump(terminal: _Terminal) -> None:
    async def read() -> None:
        while True:
            chunk = await _read_fd(terminal.master_fd, READ_CHUNK_BYTES)
            if not chunk:
                return
            terminal.append(chunk)

    reader = asyncio.create_task(read())
    waiter = asyncio.create_task(terminal.process.wait())
    try:
        _ = await asyncio.wait({reader, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if not reader.done():
            _ = await asyncio.wait({reader}, timeout=_CLOSE_GRACE_S)
        _signal_group(terminal.process.pid, signal.SIGHUP)
        terminal.exit_code = await waiter
    finally:
        _ = reader.cancel()
        _ = waiter.cancel()
        terminal.exited = True
        terminal.notify()


def _signal_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise


# Linux 上按会话结束全部进程，包含 shell 放到别的进程组里的后台作业；
# 会话里还有进程时内核不会复用这个 id，因此不会误杀无关进程。
def _kill_session(sid: int) -> None:
    _signal_group(sid, signal.SIGKILL)
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
        except OSError:
            continue
        if len(fields) > 3 and fields[3] == str(sid):
            try:
                os.kill(int(entry.name), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


# 关闭等同关掉终端窗口：先释放 PTY 主端让内核挂断会话，宽限后结束会话里剩下的进程。
async def _close(terminal: _Terminal) -> None:
    for task in terminal.tasks:
        _ = task.cancel()
    _ = await asyncio.gather(*terminal.tasks, return_exceptions=True)
    os.close(terminal.master_fd)
    if terminal.process.returncode is None:
        try:
            _ = await asyncio.wait_for(terminal.process.wait(), _CLOSE_GRACE_S)
        except TimeoutError:
            pass
    _kill_session(terminal.process.pid)
    _ = await terminal.process.wait()

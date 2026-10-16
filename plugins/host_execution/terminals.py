"""用户交互终端：执行主机上的长期 PTY 会话，输出留在环形缓冲里按偏移重复读取。"""
from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import errno
import fcntl
import os
from pathlib import Path
import pwd
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
# 监管进程挂断会话最多用两段宽限（shell、残留进程），再留一点余量。
_SUPERVISOR_WAIT_S = 2 * _CLOSE_GRACE_S + 1.0
# shell 由监管进程启动并接管 PTY；会话的结束只经监管进程，服务端从不按 PID 数字发信号。
_SUPERVISOR = Path(__file__).with_name("pty_supervisor.py")
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
    # 同一终端可能有多个浏览器同时输入，写入必须串行，否则 fd 上的等待回调会互相覆盖。
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # 会话结束或开始关闭时置位；在途写入与它竞争，不在已挂断的主端上空转。
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    closing: asyncio.Task[None] | None = None
    fd_open: bool = True

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
        async with terminal.write_lock:
            if terminal.ended.is_set() or not data:
                return
            await _write_until_ended(terminal, data)

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
            task = self._closing(terminal)
        await _settle([task])
        return True

    async def list_terminals(self, owner_session_key: str) -> list[TerminalInfo]:
        return [
            item.info() for item in sorted(self._terminals.values(), key=lambda item: item.terminal_id)
            if item.owner == owner_session_key and item.closing is None
        ]

    async def close_owner(self, owner_session_key: str) -> None:
        async with self._lock:
            tasks = [self._closing(item) for item in self._terminals.values() if item.owner == owner_session_key]
        await _settle(tasks)

    async def shutdown(self) -> None:
        async with self._lock:
            self._closed = True
            tasks = [self._closing(item) for item in self._terminals.values()]
        await _settle(tasks)

    # 关闭是终端自己的任务：调用方被取消也会做完；完成前终端留在登记表里，shutdown 能看到并等待，
    # 失败时保留记录并允许重试。
    def _closing(self, terminal: _Terminal) -> asyncio.Task[None]:
        task = terminal.closing
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            task = asyncio.create_task(self._close(terminal), name=f"terminal-close-{terminal.terminal_id}")
            terminal.closing = task
        return task

    async def _close(self, terminal: _Terminal) -> None:
        try:
            await _close(terminal)
        except Exception:
            # 失败的关闭恢复成可列出、可再关的终端，不留下看不见却仍在运行的会话。
            terminal.closing = None
            raise
        async with self._lock:
            _ = self._terminals.pop(terminal.terminal_id, None)

    def _owned(self, owner: str, terminal_id: int) -> _Terminal:
        terminal = self._terminals.get(terminal_id)
        if terminal is None or terminal.owner != owner or terminal.closing is not None:
            raise UnknownTerminalError(f"终端不存在: {terminal_id}")
        return terminal


# 输入缓冲写满后会话挂断时，主端写入只报 EAGAIN 而不是 EIO：写入与“会话结束”竞争，
# 结束先到就取消写入。取消在持锁期间完成，写入注册的 fd 回调随之撤下，关闭再释放主端。
async def _write_until_ended(terminal: _Terminal, data: bytes) -> None:
    write = asyncio.create_task(_write_fd(terminal.master_fd, data))
    ended = asyncio.create_task(terminal.ended.wait())
    try:
        _ = await asyncio.wait({write, ended}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (write, ended):
            _ = task.cancel()
        _ = await asyncio.gather(write, ended, return_exceptions=True)
    error = None if write.cancelled() else write.exception()
    # 从端全部关闭后主端写入返回 EIO，与结束同义。
    if error is not None and not (isinstance(error, OSError) and error.errno == errno.EIO):
        raise error


# 等待一组关闭任务；调用方被取消时任务继续，失败如实抛出第一个错误。
async def _settle(tasks: list[asyncio.Task[None]]) -> None:
    results = await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result


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
            sys.executable, "-I", str(_SUPERVISOR), *argv,
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            cwd=str(directory), env=_terminal_env(),
        )
    except BaseException:
        os.close(master_fd)
        raise
    finally:
        os.close(slave_fd)
    return _Terminal(terminal_id, owner, shell, str(directory), process, master_fd)


# 监管进程退出即整个会话已结束；再给读取一点时间收完残余输出，然后发布终态。
async def _pump(terminal: _Terminal) -> None:
    async def read() -> None:
        while True:
            chunk = await _read_fd(terminal.master_fd, READ_CHUNK_BYTES)
            if not chunk:
                return
            terminal.append(chunk)

    reader = asyncio.create_task(read())
    try:
        code = await terminal.process.wait()
        if not reader.done():
            _ = await asyncio.wait({reader}, timeout=_CLOSE_GRACE_S)
        terminal.exit_code = code
    finally:
        _ = reader.cancel()
        terminal.exited = True
        terminal.ended.set()
        terminal.notify()


# 关闭等同关掉终端窗口：请监管进程挂断整个会话；它不响应时直接结束它，
# 主端关闭后内核也会挂断会话。各步可重复执行，失败后重试不会二次释放。
async def _close(terminal: _Terminal) -> None:
    # 1. 先结束在途写入，再等会话结束。
    terminal.ended.set()
    process = terminal.process
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            _ = await asyncio.wait_for(process.wait(), _SUPERVISOR_WAIT_S)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            _ = await process.wait()
    # 2. 停读取，等在途写入退出后释放主端。
    for task in terminal.tasks:
        _ = task.cancel()
    _ = await asyncio.gather(*terminal.tasks, return_exceptions=True)
    # 持锁时没有写入在 fd 上注册回调（写入已因 ended 撤下），此时关闭不会让同号新 fd 收到旧回调；
    # 排队中的写入拿到锁后看到 ended 即返回。
    if terminal.fd_open:
        async with terminal.write_lock:
            terminal.fd_open = False
            os.close(terminal.master_fd)

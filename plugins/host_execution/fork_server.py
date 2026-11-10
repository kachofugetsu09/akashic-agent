from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import shlex
import shutil
import signal
import sys
import tempfile
from contextlib import ExitStack
from collections.abc import Mapping
from pathlib import Path

from core.common.diagnostic_log import log_event

logger = logging.getLogger(__name__)

# 只有显式设为 1 才启用；zsh 子 shell 的 $$ / PPID 与独立进程不同。
FORK_SERVER_ENV = "AKASHIC_SHELL_FORK_SERVER"
_READY_TIMEOUT_S = 10.0
_ACK_TIMEOUT_S = 5.0
_MAX_SERVERS = 4

# 常驻 zsh：source 快照一次，之后每条命令 fork 成独立进程组（pid == pgid）。
# monitor 只需要 SHTTY：stdin 是一个不作控制终端的 pty，后台任务因此各自建组，
# 又因为没有控制终端，命令打开 /dev/tty 仍和今天一样直接失败。
# server 退出（EOF、宿主被杀）不能向在飞 job 发 SIGHUP：与每命令 start_new_session 的存活语义一致；
# 不用 disown，server 仍需 wait 回传状态。用户自己的 hup/checkjobs/bgnice 在 job 内还原。
_SERVER_SCRIPT = r"""
_ak_restore=:
for _ak_o in hup checkjobs bgnice; do [[ -o $_ak_o ]] && _ak_restore+="; builtin setopt $_ak_o"; done
builtin unset _ak_o
builtin setopt monitor 2>/dev/null || builtin exit 97
builtin setopt no_hup no_check_jobs no_bg_nice
_ak_reply() { emulate -L zsh; builtin print -ru$_ak_out -- "$*" }
_ak_next() {
  emulate -L zsh
  local -a req=(${=1})
  local rc=0
  if [[ $req[1] == W ]]; then
    builtin wait $req[3] || rc=$?
    _ak_reply exit $req[2] $rc
    return 1
  fi
  typeset -g _ak_id=$req[2]
  typeset -ga _ak
  builtin eval "_ak=( $(<$_ak_dir/$_ak_id.req) )"
}
_ak_job() {
  emulate -L zsh
  builtin exec {_ak_in}<&- {_ak_out}>&-
  local e
  for e in "${(@)_ak[3,-1]}"; do
    if [[ $e == -* ]]; then builtin unset -- ${e#-}; else builtin export -- $e; fi
  done
  if [[ -n $_ak[1] ]]; then builtin cd -q -- $_ak[1] || builtin exit 1; fi
  typeset -g _ak_cmd=$_ak[2]
}
_ak_reply ready
while IFS= builtin read -ru$_ak_in _ak_line; do
  _ak_next "$_ak_line" || continue
  _ak_prev=$!
  { _ak_job; builtin eval "$_ak_restore; builtin unset _ak _ak_cmd _ak_id _ak_line _ak_prev _ak_in _ak_out _ak_dir _ak_restore; builtin unfunction _ak_reply _ak_next _ak_job; $_ak_cmd" } </dev/null >|$_ak_dir/$_ak_id.out 2>&1 &
  if [[ $! == $_ak_prev ]]; then _ak_reply fail $_ak_id; else _ak_reply ack $_ak_id $!; fi
done
"""




class ForkServerLost(RuntimeError):
    """fork server 已退出或失去响应；已受理的命令不会重跑。"""


def fork_server_enabled() -> bool:
    return sys.platform.startswith("linux") and os.environ.get(FORK_SERVER_ENV) == "1"


def _zsh_quote(value: str) -> str:
    escaped = "".join(
        "\\\\" if char == "\\" else "\\'" if char == "'"
        else f"\\x{ord(char):02x}" if ord(char) < 0x20 or ord(char) == 0x7F else char
        for char in value
    )
    return f"$'{escaped}'"


# 插件不能导入 utils.pidfd；部分 Python 构建缺 os.pidfd_open，退回统一编号的系统调用。
def _open_pidfd(pid: int) -> int:
    opener = getattr(os, "pidfd_open", None)
    if opener is not None:
        return opener(pid)
    fd = ctypes.CDLL(None, use_errno=True).syscall(434, pid, 0)
    if fd < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return fd


def _pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None


class ForkedProcess:
    """fork server 子进程的 asyncio.subprocess.Process 等价面；pid 同时是进程组 id。"""

    def __init__(self, pid: int, stdout: asyncio.StreamReader) -> None:
        self.pid = pid
        self.stdout: asyncio.StreamReader | None = stdout
        self.reaping = False
        self._exit: asyncio.Future[int] = asyncio.get_running_loop().create_future()

    @property
    def returncode(self) -> int | None:
        if self._exit.done() and self._exit.exception() is None:
            return self._exit.result()
        return None

    async def wait(self) -> int:
        return await asyncio.shield(self._exit)

    def send_signal(self, sig: int, /) -> None:
        os.killpg(self.pid, sig)

    def kill(self) -> None:
        os.killpg(self.pid, signal.SIGKILL)

    def settle(self, code: int | None) -> None:
        if self._exit.done():
            return
        if code is None:
            self._exit.set_exception(ForkServerLost("fork server 已退出，执行退出状态未确认"))
            _ = self._exit.exception()
            return
        self._exit.set_result(code)


class ShellForkServer:
    """一个快照对应一个常驻 zsh；Python 用 pidfd 观察退出，再由 server wait 回传状态。"""

    def __init__(self, shell_path: Path, snapshot: Path, base_env: dict[str, str]) -> None:
        self.shell_path = shell_path
        self.snapshot = snapshot
        self.base_env = base_env
        self.alive = False
        self._process: asyncio.subprocess.Process | None = None
        self._request_fd: int | None = None
        self._pty_master: int | None = None
        self._reply_transport: asyncio.ReadTransport | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._dir = Path(tempfile.mkdtemp(prefix="akashic-fork-"))
        self._next_id = 0
        self._acks: dict[int, asyncio.Future[int]] = {}
        self._children: dict[int, ForkedProcess] = {}
        self._ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    # 启动 zsh 并等待 ready；失败由调用方 close 回收并回退到每命令 spawn。
    async def start(self) -> None:
        import pty

        # 1. 本地描述符逐个登记，部分启动失败也能关闭。
        loop = asyncio.get_running_loop()
        with ExitStack() as local_fds:
            request_read, self._request_fd = os.pipe()
            local_fds.callback(os.close, request_read)
            reply_read, reply_write = os.pipe()
            local_fds.enter_context(os.fdopen(reply_read, "rb", 0))
            local_fds.callback(os.close, reply_write)
            self._pty_master, slave = pty.openpty()
            local_fds.callback(os.close, slave)
            prelude = (f"_ak_in={request_read} _ak_out={reply_write} _ak_dir={shlex.quote(str(self._dir))}; "
                       f". {shlex.quote(str(self.snapshot))} || builtin exit $?\n")
            self._process = await asyncio.create_subprocess_exec(
                str(self.shell_path), "-c", prelude + _SERVER_SCRIPT, stdin=slave,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                env=self.base_env, pass_fds=(request_read, reply_write), start_new_session=True,
            )
            # 2. transport 接管独立副本，ExitStack 仍负责原始描述符。
            os.set_blocking(self._request_fd, False)
            reader = asyncio.StreamReader()
            self._reply_transport, _ = await loop.connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(os.dup(reply_read), "rb", 0),
            )
        self._reader_task = asyncio.create_task(self._read_replies(reader), name="shell-fork-server")
        loop.add_reader(self._pty_master, self._discard_tty)
        await asyncio.wait_for(asyncio.shield(self._ready), _READY_TIMEOUT_S)
        self.alive = True

    def _discard_tty(self) -> None:
        if self._pty_master is None:
            return
        try:
            _ = os.read(self._pty_master, 4096)
        except OSError:
            _ = asyncio.get_running_loop().remove_reader(self._pty_master)

    # 受理一条命令；请求写出后任何失败都只上报，绝不回退重跑。
    async def spawn(self, command: str, cwd: Path | None, env: Mapping[str, str]) -> ForkedProcess:
        # 1. 先建 FIFO 读端和临时写端：子进程打开写端不阻塞，leader 退出前也不会提前 EOF。
        self._next_id += 1
        request_id = self._next_id
        fifo = self._dir / f"{request_id}.out"
        request = self._dir / f"{request_id}.req"
        os.mkfifo(fifo, 0o600)
        read_fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        hold_fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        changed = [f"{name}={value}" for name, value in env.items() if self.base_env.get(name) != value]
        removed = [f"-{name}" for name in self.base_env if name not in env]
        fields = ["" if cwd is None else str(cwd), command, *removed, *changed]
        _ = request.write_text(" ".join(map(_zsh_quote, fields)), encoding="utf-8")
        ack: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self._acks[request_id] = ack

        # 2. 只有 ack 带回 pid 才进入跟踪；超时视为 server 卡死，整体回收。
        try:
            self._send(f"R {request_id}")
            pid = await asyncio.wait_for(ack, _ACK_TIMEOUT_S)
        except BaseException as exc:
            os.close(read_fd)
            os.close(hold_fd)
            fifo.unlink(missing_ok=True)
            if isinstance(exc, asyncio.TimeoutError):
                await self.close()
                raise ForkServerLost("fork server 未在时限内确认，命令是否已启动未确认") from exc
            raise
        finally:
            _ = self._acks.pop(request_id, None)
            request.unlink(missing_ok=True)
        return await self._track(request_id, pid, read_fd, hold_fd)

    # 用 pidfd 观察已受理子进程的退出，再向 server 取回 wait 状态。
    async def _track(self, request_id: int, pid: int, read_fd: int, hold_fd: int) -> ForkedProcess:
        # 1. 输出读端交给 asyncio；pid 已被回收或不再是组长时直接按已退出处理。
        loop = asyncio.get_running_loop()
        stdout = asyncio.StreamReader()
        _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(stdout), os.fdopen(read_fd, "rb", 0))
        child = ForkedProcess(pid, stdout)
        self._children[request_id] = child
        try:
            pidfd: int | None = _open_pidfd(pid)
        except ProcessLookupError:
            pidfd = None
        if pidfd is not None and _pgid(pid) != pid:
            os.close(pidfd)
            pidfd = None

        # 2. leader 退出后关闭临时写端，后代全部关闭输出时读端得到 EOF。
        def _exited() -> None:
            if pidfd is not None:
                _ = loop.remove_reader(pidfd)
                os.close(pidfd)
            os.close(hold_fd)
            (self._dir / f"{request_id}.out").unlink(missing_ok=True)
            child.reaping = True
            try:
                self._send(f"W {request_id} {pid}")
            except (ForkServerLost, OSError):
                _ = self._children.pop(request_id, None)
                child.settle(None)

        if pidfd is None:
            _exited()
        else:
            loop.add_reader(pidfd, _exited)
        return child

    def _send(self, line: str) -> None:
        if self._request_fd is None or not self.alive:
            raise ForkServerLost("fork server 不可用")
        _ = os.write(self._request_fd, line.encode() + b"\n")

    async def _read_replies(self, reader: asyncio.StreamReader) -> None:
        try:
            while line := await reader.readline():
                kind, _, rest = line.decode().strip().partition(" ")
                request_id, _, value = rest.partition(" ")
                if kind == "ready":
                    self._ready.set_result(None)
                elif kind == "exit" and (child := self._children.pop(int(request_id), None)) is not None:
                    child.settle(int(value))
                elif (ack := self._acks.get(int(request_id or 0))) is not None and not ack.done():
                    if kind == "ack":
                        ack.set_result(int(value))
                    else:
                        ack.set_exception(OSError("fork server 无法创建子进程"))
        finally:
            self._mark_lost()

    # server 退出：未 ack 的请求失败；已在 wait 的子进程退出状态标为未确认。
    def _mark_lost(self) -> None:
        self.alive = False
        if not self._ready.done():
            self._ready.set_exception(ForkServerLost("fork server 启动失败"))
            _ = self._ready.exception()
        for ack in self._acks.values():
            if not ack.done():
                ack.set_exception(ForkServerLost("fork server 退出，命令是否已启动未确认"))
        for request_id, child in list(self._children.items()):
            if child.reaping:
                del self._children[request_id]
                child.settle(None)

    async def close(self) -> None:
        """回收 server；确认失败时也回收尚未登记的直接子进程组。"""
        # 1. 先停止 server，固定未确认 job 的集合，避免遍历期间继续 fork。
        process = self._process
        if self._acks and process is not None and process.returncode is None:
            try:
                os.kill(process.pid, signal.SIGSTOP)
                # SIGSTOP 生效后才读取 children，保证没有 fork 窗口。
                while Path(f"/proc/{process.pid}/stat").read_text().split(") ", 1)[1][0] not in "TtZ":
                    await asyncio.sleep(0)
                children = Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()
                tracked = {child.pid for child in self._children.values()}
                for raw_pid in children:
                    pid = int(raw_pid)
                    if pid in tracked:
                        continue
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                os.kill(process.pid, signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError):
                pass
        # 2. 关闭请求管道让 zsh 读到 EOF 自行退出；卡死时按进程组强杀。
        self.alive = False
        if self._request_fd is not None:
            os.close(self._request_fd)
            self._request_fd = None
        process = self._process
        if process is not None and process.returncode is None:
            try:
                _ = await asyncio.wait_for(process.wait(), 2.0)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                _ = await process.wait()
        # 3. 显式关闭控制管道，后代意外继承描述符也不阻塞退出。
        if self._reply_transport is not None:
            self._reply_transport.close()
            self._reply_transport = None
        # 回收 reader、pty 和请求目录。
        if self._reader_task is not None:
            _ = await asyncio.gather(self._reader_task, return_exceptions=True)
        if self._pty_master is not None:
            _ = asyncio.get_running_loop().remove_reader(self._pty_master)
            os.close(self._pty_master)
            self._pty_master = None
        shutil.rmtree(self._dir, ignore_errors=True)


class ShellForkPool:
    """manager 持有的 fork server 集合，随 manager.shutdown 一起回收。"""

    def __init__(self) -> None:
        self._enabled = fork_server_enabled()
        self._servers: dict[Path, ShellForkServer] = {}

    # 返回 None 表示命令尚未交给任何 server，调用方可以安全地走每命令 spawn。
    async def spawn(
        self, shell_path: Path, snapshot: Path, command: str, cwd: Path | None, env: Mapping[str, str],
    ) -> ForkedProcess | None:
        # 1. 关闭开关、cwd 不存在或环境名无法 export 时保持原有 spawn 语义。
        if not self._enabled or (cwd is not None and not cwd.is_dir()):
            return None
        if not all(name.isidentifier() for name in env):
            return None
        server = self._servers.get(snapshot)
        if server is not None and not server.alive:
            await server.close()
            del self._servers[snapshot]
            server = None
        # 2. 首次使用该快照时启动 server；启动失败关闭整个池，不再反复付启动成本。
        if server is None:
            if len(self._servers) >= _MAX_SERVERS:
                return None
            server = ShellForkServer(shell_path, snapshot, dict(env))
            self._servers[snapshot] = server
            try:
                await server.start()
            except (OSError, asyncio.TimeoutError, ForkServerLost) as exc:
                await server.close()
                del self._servers[snapshot]
                self._enabled = False
                log_event(logger, logging.WARNING, "shell.fork_server_disabled", message=str(exc), error_type=type(exc).__name__)
                return None
        return await server.spawn(command, cwd, env)

    async def close(self) -> None:
        for snapshot, server in list(self._servers.items()):
            await server.close()
            del self._servers[snapshot]

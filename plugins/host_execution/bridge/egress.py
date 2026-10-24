from __future__ import annotations

import asyncio
import logging
import os
import socket
import stat
from pathlib import Path

from core.common.diagnostic_log import log_event

logger = logging.getLogger(__name__)

# 宿主出口：声明了 host_egress 的 Workload 经这个 SOCKS5 入口访问网络（只有 CONNECT，无认证）。
# 监听 unix socket，不占宿主端口、不经宿主防火墙；连接由宿主上的本进程发起，
# 所以 localhost 是宿主本机，域名按宿主的 DNS 解析，内网和外网也和宿主看到的一样。
EGRESS_SOCKET_NAME = "socks.sock"
_CONNECT_TIMEOUT_S = 30.0
_CHUNK = 65536
# 看守间隔即入口丢失后的最长恢复时间。
_WATCH_INTERVAL_S = 2.0
_PROBE_TIMEOUT_S = 1.0


class EgressPathError(RuntimeError):
    """监听路径被不属于本实例的东西占着；不删除、不覆盖，等它消失后再绑定。"""


# 宿主出口的唯一生命周期 owner：绑定 socket，按 inode 看守路径，路径丢失时就地重绑。
# 已建立的连接属于旧 listener，重绑不打断它们；Bridge、Core 与 Workload 都不需要重启。
class HostEgress:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._path = directory / EGRESS_SOCKET_NAME
        self._server: asyncio.AbstractServer | None = None
        self._identity: tuple[int, int] | None = None
        self._problem: str | None = None
        self._watch: asyncio.Task[None] | None = None

    @property
    def ready(self) -> bool:
        return self._server is not None and self._problem is None

    # 首次绑定失败只记原因：看守任务照常启动，路径空出来后自己接上。
    async def start(self) -> None:
        try:
            await self._bind()
        except (OSError, EgressPathError) as error:
            self._report(str(error))
        else:
            log_event(logger, logging.INFO, "host_egress.started", outcome="ready")
        self._watch = asyncio.create_task(self._watch_path(), name="host-egress-watch")

    async def close(self) -> None:
        if self._watch is not None:
            self._watch.cancel()
            await asyncio.gather(self._watch, return_exceptions=True)
        self._release()

    async def _watch_path(self) -> None:
        while True:
            await asyncio.sleep(_WATCH_INTERVAL_S)
            if self._owns_path():
                self._report(None)
                continue
            try:
                await self._bind()
            except (OSError, EgressPathError) as error:
                self._report(str(error))
                continue
            if self._problem is None:
                log_event(logger, logging.WARNING, "host_egress.rebound",
                          message=f"监听路径丢失，已就地重新绑定: {self._path}", outcome="ready")
            self._report(None)

    # 1. 路径仍是本实例绑定的 socket（同一 dev/inode）才算健康。
    def _owns_path(self) -> bool:
        try:
            current = self._path.lstat()
        except FileNotFoundError:
            return False
        return self._identity == (current.st_dev, current.st_ino)

    # 2. 清出路径后自己 bind：bind 不会替换已存在的路径，并发的另一个 owner 只会让本次失败。
    async def _bind(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        await self._clear_stale()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(self._path))
            # Workload 与 Bridge 用同一个宿主用户身份运行；组可写留给按组授权的部署。
            os.chmod(self._path, 0o660)
            server = await asyncio.start_unix_server(_handle, sock=sock)
        except BaseException:
            sock.close()
            raise
        bound = self._path.lstat()
        self._release()
        self._server, self._identity = server, (bound.st_dev, bound.st_ino)

    # 3. 只清理本用户留下、已无人监听的 socket；普通文件、他人的或仍在监听的都明确拒绝。
    async def _clear_stale(self) -> None:
        try:
            found = self._path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(found.st_mode):
            raise EgressPathError(f"出口路径被非 socket 占用: {self._path}")
        if found.st_uid != os.getuid():
            raise EgressPathError(f"出口 socket 属于其他用户: {self._path}")
        if await _listening(self._path):
            raise EgressPathError(f"出口 socket 已有其他进程监听: {self._path}")
        current = self._path.lstat()
        if (current.st_dev, current.st_ino) == (found.st_dev, found.st_ino):
            self._path.unlink()

    # 停止旧 listener；路径只在仍是自己的 inode 时删除，不碰别人新建的入口。
    def _release(self) -> None:
        if self._server is not None:
            self._server.close()
        if self._identity is not None and self._owns_path():
            self._path.unlink(missing_ok=True)
        self._server, self._identity = None, None

    # 只在状态变化时记一次：丢失、恢复、无法恢复的原因都进日志。
    def _report(self, problem: str | None) -> None:
        if problem == self._problem:
            return
        if problem is None:
            log_event(logger, logging.INFO, "host_egress.recovered", outcome="ready")
        else:
            log_event(logger, logging.WARNING, "host_egress.unavailable", message=problem, outcome="failed")
        self._problem = problem


async def _listening(path: Path) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_unix_connection(str(path)), _PROBE_TIMEOUT_S)
    except (OSError, TimeoutError):
        return False
    writer.close()
    return True


# 一条连接：协商无认证，读 CONNECT 目标，在宿主上连过去，然后双向转发到任一侧关闭。
async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    upstream: asyncio.StreamWriter | None = None
    try:
        # 1. 协商：只接受"无认证"。
        version, count = await reader.readexactly(2)
        methods = await reader.readexactly(count)
        if version != 5 or 0 not in methods:
            writer.write(b"\x05\xff")
            return
        writer.write(b"\x05\x00")
        # 2. 请求：只支持 CONNECT；目标可以是 IPv4、域名或 IPv6。
        version, command, _, kind = await reader.readexactly(4)
        host = await _read_host(reader, kind)
        port = int.from_bytes(await reader.readexactly(2), "big")
        if version != 5 or command != 1 or host is None:
            writer.write(_reply(7 if host is not None else 8))
            return
        # 3. 在宿主上连接目标；失败按"拒绝连接"回复，浏览器显示连接失败。
        try:
            upstream_reader, upstream = await asyncio.wait_for(
                asyncio.open_connection(host, port), _CONNECT_TIMEOUT_S
            )
        except (OSError, TimeoutError):
            writer.write(_reply(5))
            return
        writer.write(_reply(0))
        await asyncio.gather(_pipe(reader, upstream), _pipe(upstream_reader, writer))
    except (asyncio.IncompleteReadError, ConnectionError):
        pass
    finally:
        for stream in (upstream, writer):
            if stream is not None:
                stream.close()


async def _read_host(reader: asyncio.StreamReader, kind: int) -> str | None:
    if kind == 1:
        return socket.inet_ntop(socket.AF_INET, await reader.readexactly(4))
    if kind == 3:
        length = (await reader.readexactly(1))[0]
        return (await reader.readexactly(length)).decode("ascii", errors="replace")
    if kind == 4:
        return socket.inet_ntop(socket.AF_INET6, await reader.readexactly(16))
    return None


def _reply(code: int) -> bytes:
    return bytes((5, code, 0, 1, 0, 0, 0, 0, 0, 0))


# 一个方向读到 EOF 时只关写端，另一个方向照常收完。
async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(_CHUNK):
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError):
        writer.close()

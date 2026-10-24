from __future__ import annotations

import asyncio
import logging
import os
import socket
from pathlib import Path

from core.common.diagnostic_log import log_event

logger = logging.getLogger(__name__)

# 宿主出口：声明了 host_egress 的 Workload 经这个 SOCKS5 入口访问网络（只有 CONNECT，无认证）。
# 监听 unix socket，不占宿主端口、不经宿主防火墙；连接由宿主上的本进程发起，
# 所以 localhost 是宿主本机，域名按宿主的 DNS 解析，内网和外网也和宿主看到的一样。
EGRESS_SOCKET_NAME = "socks.sock"
_CONNECT_TIMEOUT_S = 30.0
_CHUNK = 65536


async def serve_egress(directory: Path) -> asyncio.AbstractServer:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / EGRESS_SOCKET_NAME
    if path.exists():
        if not path.is_socket():
            raise RuntimeError(f"拒绝覆盖非 socket: {path}")
        path.unlink()
    server = await asyncio.start_unix_server(_handle, path=str(path))
    # Workload 与 Bridge 用同一个宿主用户身份运行；组可写留给按组授权的部署。
    os.chmod(path, 0o660)
    log_event(logger, logging.INFO, "host_egress.started", outcome="ready")
    return server


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

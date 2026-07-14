from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from pathlib import Path

from agent.control.service import ControlService
from infra.control.connection import NdjsonConnection

logger = logging.getLogger(__name__)


class SocketAppServer:
    """通过 workspace 私有 Unix socket 暴露 app-server。"""

    def __init__(
        self,
        endpoint: str | Path,
        service: ControlService,
        *,
        max_connections: int = 32,
        max_pending_requests: int = 128,
        max_message_bytes: int = 2 * 1024 * 1024,
        outbound_queue_size: int = 512,
    ) -> None:
        raw_endpoint = str(endpoint)
        self._tcp_address = _parse_loopback_tcp(raw_endpoint)
        self.endpoint: str | Path = raw_endpoint if self._tcp_address else Path(raw_endpoint)
        self._service = service
        self._slots = asyncio.Semaphore(max_connections)
        self._max_pending_requests = max_pending_requests
        self._max_message_bytes = max_message_bytes
        self._outbound_queue_size = outbound_queue_size
        self._server: asyncio.AbstractServer | None = None
        self._connections: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        """清理 stale socket 后监听，并把文件权限收紧到当前用户。"""

        # 1. 只删除无法连接的旧 socket；活跃 owner 必须 fail-loud。
        if self._tcp_address is not None:
            host, port = self._tcp_address
            self._server = await asyncio.start_server(self._accept, host=host, port=port)
            socket = self._server.sockets[0]
            bound_host, bound_port = socket.getsockname()[:2]
            self.endpoint = f"{bound_host}:{bound_port}"
            logger.info("app-server listening on %s", self.endpoint)
            return

        endpoint = self.endpoint
        assert isinstance(endpoint, Path)
        endpoint.parent.mkdir(parents=True, exist_ok=True)
        if endpoint.exists():
            try:
                reader, writer = await asyncio.open_unix_connection(str(endpoint))
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                endpoint.unlink()
            else:
                writer.close()
                await writer.wait_closed()
                raise RuntimeError(f"app-server endpoint 已由其他进程占用: {endpoint}")

        # 2. 绑定完成后立即建立权限不变量。
        self._server = await asyncio.start_unix_server(self._accept, path=str(endpoint))
        try:
            os.chmod(endpoint, 0o600)
        except OSError:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            endpoint.unlink(missing_ok=True)
            raise
        logger.info("app-server listening on %s", self.endpoint)

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        async with self._slots:
            connection = NdjsonConnection(
                reader,
                writer,
                self._service,
                max_message_bytes=self._max_message_bytes,
                max_pending_requests=self._max_pending_requests,
                outbound_queue_size=self._outbound_queue_size,
            )
            task = asyncio.current_task()
            assert task is not None
            self._connections.add(task)
            try:
                await connection.run()
            except (ConnectionError, BrokenPipeError):
                logger.info("app-server client disconnected")
            finally:
                self._connections.discard(task)

    async def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        for task in tuple(self._connections):
            task.cancel()
        if self._connections:
            await asyncio.gather(*self._connections, return_exceptions=True)
        self._connections.clear()
        if isinstance(self.endpoint, Path):
            self.endpoint.unlink(missing_ok=True)


def _parse_loopback_tcp(endpoint: str) -> tuple[str, int] | None:
    if endpoint.startswith("/") or endpoint.count(":") != 1:
        return None
    host, raw_port = endpoint.rsplit(":", 1)
    try:
        address = ipaddress.ip_address(host)
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"无效 app-server TCP endpoint: {endpoint}") from exc
    if not address.is_loopback:
        raise ValueError(f"app-server TCP 只允许 loopback: {endpoint}")
    if not 0 <= port <= 65535:
        raise ValueError(f"app-server TCP 端口无效: {port}")
    return host, port


def is_tcp_endpoint(endpoint: str) -> bool:
    return _parse_loopback_tcp(endpoint) is not None

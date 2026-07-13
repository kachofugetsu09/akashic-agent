"""IPC server 渠道：POSIX 使用 Unix domain socket，Windows 使用 loopback TCP，供本地 CLI 与运行中的 agent 进程通信。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from agent.config import _normalize_cli_socket_endpoint
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus, OutboundSubscription

if TYPE_CHECKING:
    from proactive_v2.loop import ProactiveLoop

logger = logging.getLogger(__name__)

CHANNEL = "cli"
CommandHandler = Callable[[dict[str, object]], Awaitable[str]]


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int] | None:
    if endpoint.count(":") != 1:
        return None
    host, port = endpoint.rsplit(":", 1)
    if not host:
        return None
    try:
        return host, int(port)
    except ValueError:
        return None


def _normalize_endpoint(endpoint: str) -> str:
    return _normalize_cli_socket_endpoint(endpoint)


class IPCServerChannel:
    def __init__(
        self,
        bus: MessageBus,
        socket_path: str,
        proactive_loop: "ProactiveLoop | None" = None,
        default_session_key: str = "",
    ) -> None:
        self._bus = bus
        self._socket_path = _normalize_endpoint(socket_path)
        self._proactive_loop = proactive_loop
        self._default_session_key = default_session_key.strip()
        self._writers: dict[str, asyncio.StreamWriter] = {}
        self._server: asyncio.AbstractServer | None = None
        self._command_handlers: dict[str, CommandHandler] = {}
        self._outbound_subscription: OutboundSubscription | None = None

    def register_command(self, name: str, handler: CommandHandler) -> None:
        self._command_handlers[name] = handler

    async def start(self) -> None:
        tcp_endpoint = _parse_tcp_endpoint(self._socket_path)
        if tcp_endpoint is not None:
            host, port = tcp_endpoint
            self._server = await asyncio.start_server(
                self._handle_connection,
                host=host,
                port=port,
            )
            self._outbound_subscription = self._bus.subscribe_outbound(
                CHANNEL,
                self._on_response,
            )
            logger.info("IPC server listening on tcp://%s:%s", host, port)
            return

        if not hasattr(asyncio, "start_unix_server"):
            raise RuntimeError("Unix sockets are unavailable on this platform; use a host:port endpoint instead.")
        Path(self._socket_path).unlink(missing_ok=True)
        server = await asyncio.start_unix_server(
            self._handle_connection,
            path=self._socket_path,
        )
        try:
            os.chmod(self._socket_path, 0o600)
            subscription = self._bus.subscribe_outbound(
                CHANNEL,
                self._on_response,
            )
        except OSError as start_error:
            server.close()
            cleanup_error: OSError | None = None
            try:
                await server.wait_closed()
            except OSError as error:
                cleanup_error = error
            finally:
                try:
                    Path(self._socket_path).unlink(missing_ok=True)
                except OSError as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None:
                raise start_error from cleanup_error
            raise
        self._server = server
        self._outbound_subscription = subscription
        logger.info("IPC server listening on %s", self._socket_path)

    async def stop(self) -> None:
        """关闭 IPC server、客户端连接和 outbound subscription。"""

        # 1. 在任何 await 前转移并关闭全部资源所有权。
        subscription = self._outbound_subscription
        self._outbound_subscription = None
        if subscription is not None:
            subscription.close()

        server = self._server
        self._server = None
        writers = tuple(self._writers.values())
        self._writers.clear()
        first_error: OSError | None = None
        if server is not None:
            try:
                server.close()
            except OSError as error:
                first_error = error
        for writer in writers:
            try:
                writer.close()
            except OSError as error:
                if first_error is None:
                    first_error = error

        # 2. 等待所有关闭动作，记录首个 OSError 但不跳过后续资源。
        if server is not None:
            try:
                await server.wait_closed()
            except OSError as error:
                if first_error is None:
                    first_error = error
            finally:
                if _parse_tcp_endpoint(self._socket_path) is None:
                    try:
                        Path(self._socket_path).unlink(missing_ok=True)
                    except OSError as error:
                        if first_error is None:
                            first_error = error
        for writer in writers:
            try:
                await writer.wait_closed()
            except OSError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def set_proactive_loop(self, proactive_loop: "ProactiveLoop") -> None:
        self._proactive_loop = proactive_loop
        logger.info("[cli] ProactiveLoop attached")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername") or "local"
        chat_id = f"cli-{id(writer)}"
        self._writers[chat_id] = writer
        logger.info("[cli] client connected session=%s peer=%s", chat_id, peer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    raw_data: object = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("[cli] received non-JSON payload")
                    continue
                if not isinstance(raw_data, dict):
                    logger.warning(
                        "[cli] received non-object JSON payload: type=%s",
                        type(raw_data).__name__,
                    )
                    continue
                data = cast(dict[str, object], raw_data)

                if data.get("type") == "command":
                    await self._handle_command(data, chat_id, writer)
                    continue

                content = str(data.get("content", "")).strip()
                if not content:
                    continue
                preview = content[:60] + "..." if len(content) > 60 else content
                metadata = _session_override_metadata(
                    data,
                    default_session_key=self._default_session_key,
                )
                session_key = metadata.get("session_key_override", f"{CHANNEL}:{chat_id}")
                logger.info(
                    "[cli] received route=%s session=%s content=%r",
                    chat_id,
                    session_key,
                    preview,
                )
                await self._bus.publish_inbound(
                    InboundMessage(
                        channel=CHANNEL,
                        sender="cli-user",
                        chat_id=chat_id,
                        content=content,
                        metadata=metadata,
                    )
                )
        finally:
            self._writers.pop(chat_id, None)
            writer.close()
            await writer.wait_closed()
            logger.info("[cli] client disconnected session=%s", chat_id)

    async def _handle_command(
        self,
        data: dict[str, object],
        chat_id: str,
        writer: asyncio.StreamWriter,
    ) -> None:
        cmd = data.get("command", "")
        logger.info("[cli] received command cmd=%r session=%s", cmd, chat_id)
        handler = self._command_handlers.get(str(cmd))
        if handler is not None:
            try:
                message = await handler(data)
            except (ValueError, RuntimeError) as error:
                await self._write_command_result(
                    writer,
                    ok=False,
                    message=str(error),
                )
                return
            await self._write_command_result(writer, ok=True, message=message)
            return
        await self._write_command_result(
            writer,
            ok=False,
            message=f"unknown command: {cmd!r}",
        )

    @staticmethod
    async def _write_command_result(
        writer: asyncio.StreamWriter,
        *,
        ok: bool,
        message: str,
    ) -> None:
        payload = (
            json.dumps(
                {"type": "command_result", "ok": ok, "message": message},
                ensure_ascii=False,
            )
            + "\n"
        )
        writer.write(payload.encode("utf-8"))
        await writer.drain()

    async def _on_response(self, msg: OutboundMessage) -> None:
        writer = self._writers.get(msg.chat_id)
        if writer and not writer.is_closing():
            payload = (
                json.dumps(
                    {
                        "type": "assistant",
                        "content": msg.content,
                        "metadata": msg.metadata or {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            writer.write(payload.encode("utf-8"))
            await writer.drain()


def _session_override_metadata(
    data: dict[str, object],
    *,
    default_session_key: str,
) -> dict[str, object]:
    session_key = str(
        data.get("as_session_key")
        or data.get("session_key")
        or default_session_key
        or ""
    ).strip()
    if not session_key:
        channel = str(data.get("as_channel") or "").strip()
        chat_id = str(data.get("as_chat_id") or "").strip()
        session_key = f"{channel}:{chat_id}" if channel and chat_id else ""
    if not session_key:
        return {}
    channel, chat_id = _split_session_key(session_key)
    return {
        "session_key_override": session_key,
        "context_channel": channel,
        "context_chat_id": chat_id,
    }


def _split_session_key(session_key: str) -> tuple[str, str]:
    channel, sep, chat_id = session_key.partition(":")
    if not sep:
        return "", session_key
    return channel, chat_id

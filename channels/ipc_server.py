"""
IPC Server Channel（服务端）

在 Unix socket 上监听，CLI 客户端连接后可双向通信。
每条连接独立维护 session，消息流向：
  CLI client → socket → MessageBus → AgentLoop → socket → CLI client
"""
import asyncio
import json
import logging
import os
from pathlib import Path

from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus

logger = logging.getLogger(__name__)

CHANNEL = "cli"


class IPCServerChannel:

    def __init__(self, bus: MessageBus, socket_path: str) -> None:
        self._bus = bus
        self._socket_path = socket_path
        self._writers: dict[str, asyncio.StreamWriter] = {}  # chat_id → writer
        bus.subscribe_outbound(CHANNEL, self._on_response)

    async def start(self) -> None:
        # 清理上次遗留的 socket 文件
        Path(self._socket_path).unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self._socket_path
        )
        os.chmod(self._socket_path, 0o600)  # 仅当前用户可连接
        logger.info(f"IPC server 监听: {self._socket_path}")

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()
        Path(self._socket_path).unlink(missing_ok=True)

    # ── 私有方法 ──────────────────────────────────────────────────

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername") or "unix"
        chat_id = f"cli-{id(writer)}"
        self._writers[chat_id] = writer
        logger.info(f"[cli] 客户端已连接  session={chat_id}  peer={peer}")
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    data = json.loads(line)
                    content = data.get("content", "").strip()
                    if not content:
                        continue
                    preview = content[:60] + "..." if len(content) > 60 else content
                    logger.info(f"[cli] 收到消息  session={chat_id}  内容: {preview!r}")
                    await self._bus.publish_inbound(InboundMessage(
                        channel=CHANNEL,
                        sender="cli-user",
                        chat_id=chat_id,
                        content=content,
                    ))
                except json.JSONDecodeError:
                    logger.warning(f"[cli] 收到非 JSON 数据，已忽略")
        finally:
            self._writers.pop(chat_id, None)
            writer.close()
            logger.info(f"[cli] 客户端已断开  session={chat_id}")

    async def _on_response(self, msg: OutboundMessage) -> None:
        writer = self._writers.get(msg.chat_id)
        if writer and not writer.is_closing():
            payload = json.dumps(
                {
                    "type": "assistant",
                    "content": msg.content,
                    "metadata": msg.metadata or {},
                },
                ensure_ascii=False,
            ) + "\n"
            writer.write(payload.encode())
            await writer.drain()

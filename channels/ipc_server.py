"""
IPC Server Channel（服务端）

在 Unix socket 上监听，CLI 客户端连接后可双向通信。
每条连接独立维护 session，消息流向：
  CLI client → socket → MessageBus → AgentLoop → socket → CLI client

特殊命令（type="command"）：
  当前无内置命令，统一返回 unknown。
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus

if TYPE_CHECKING:
    from proactive.loop import ProactiveLoop

logger = logging.getLogger(__name__)

CHANNEL = "cli"


class IPCServerChannel:
    def __init__(
        self,
        bus: MessageBus,
        socket_path: str,
        proactive_loop: "ProactiveLoop | None" = None,
    ) -> None:
        self._bus = bus
        self._socket_path = socket_path
        self._proactive_loop = proactive_loop
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

    def set_proactive_loop(self, proactive_loop: "ProactiveLoop") -> None:
        """在 IPC server 启动后注入 ProactiveLoop。"""
        self._proactive_loop = proactive_loop
        logger.info("[cli] ProactiveLoop 已注入")

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
                    # 特殊命令分支
                    if data.get("type") == "command":
                        await self._handle_command(data, chat_id, writer)
                        continue
                    content = data.get("content", "").strip()
                    if not content:
                        continue
                    preview = content[:60] + "..." if len(content) > 60 else content
                    logger.info(f"[cli] 收到消息  session={chat_id}  内容: {preview!r}")
                    await self._bus.publish_inbound(
                        InboundMessage(
                            channel=CHANNEL,
                            sender="cli-user",
                            chat_id=chat_id,
                            content=content,
                        )
                    )
                except json.JSONDecodeError:
                    logger.warning(f"[cli] 收到非 JSON 数据，已忽略")
        finally:
            self._writers.pop(chat_id, None)
            writer.close()
            logger.info(f"[cli] 客户端已断开  session={chat_id}")

    async def _handle_command(
        self,
        data: dict,
        chat_id: str,
        writer: asyncio.StreamWriter,
    ) -> None:
        """处理 type=command 的特殊指令，结果通过同一 writer 回写。"""
        cmd = data.get("command", "")
        logger.info("[cli] 收到命令 cmd=%r session=%s", cmd, chat_id)

        await self._write_command_result(
            writer,
            ok=False,
            message=f"未知命令: {cmd!r}",
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
        writer.write(payload.encode())
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
            writer.write(payload.encode())
            await writer.drain()

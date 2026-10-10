from __future__ import annotations

import asyncio
import json
import sys
import os
from collections.abc import Mapping
from uuid import uuid4

from .protocol.router import ConnectionRouter
from .service import ControlService
from .contract import RequestTransport
from agent.plugin_composition.control_frames import FrameBook


class StdioAppServer(RequestTransport):
    """在 stdin/stdout 上运行单连接 NDJSON app-server。"""

    def __init__(self, service: ControlService, *, max_message_bytes: int = 2 * 1024 * 1024,
                 control_frames: FrameBook | None = None, output_fd: int = 1) -> None:
        self._service = service
        self._output_fd = output_fd
        self._writer: asyncio.StreamWriter | None = None
        self._max_message_bytes = max_message_bytes
        self._write_lock = asyncio.Lock()
        self.connection_id = f"stdio:{uuid4().hex}"
        self._frames = control_frames or service.control_frames

    async def _send(self, message: dict[str, object]) -> None:
        await self._send_frame(message, page=None)

    async def _send_message_page(
        self, message: dict[str, object], page: Mapping[str, object]
    ) -> None:
        """写入 Router 已确认的 message/read 或 messages.appended 页面。"""
        await self._send_frame(message, page=page)

    async def _send_frame(
        self, message: dict[str, object], *, page: Mapping[str, object] | None
    ) -> None:
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        tracked = () if page is None else self._frames.resolve_page(self.connection_id, page)
        written = asyncio.get_running_loop().create_future() if tracked else None
        if written is not None:
            self._frames.attach_page(tracked, written)
        async with self._write_lock:
            try:
                writer = self._writer
                assert writer is not None
                writer.write(payload.encode("utf-8"))
                async with asyncio.timeout(10):
                    await writer.drain()
            except BaseException as error:
                if written is not None:
                    written.set_exception(error)
                raise
            else:
                if written is not None:
                    written.set_result(None)

    async def run(self) -> None:
        """只持有标准流的副本，取消时关闭真实 pipe transport。"""
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=self._max_message_bytes + 1)
        input_file = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
        output_file = os.fdopen(os.dup(self._output_fd), "wb", buffering=0)
        input_transport = None
        output_transport = None
        router = ConnectionRouter(
            self._service, self._send, transport=self,
            send_message_page=self._send_message_page,
        )
        try:
            input_transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), input_file)
            output_transport, protocol = await loop.connect_write_pipe(lambda: asyncio.streams.FlowControlMixin(loop=loop), output_file)
            self._writer = asyncio.StreamWriter(output_transport, protocol, reader, loop)
            while True:
                line = await reader.readline()
                if not line:
                    return
                if len(line) > self._max_message_bytes:
                    await self._send({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32600, "message": "Message too large"},
                    })
                    return
                await router.handle_line(line)
        finally:
            if input_transport is not None:
                input_transport.close()
            else:
                input_file.close()
            if output_transport is not None:
                output_transport.close()
            else:
                output_file.close()
            await router.close()
            error = ConnectionError("stdio connection closed")
            self._frames.fail_connection(self.connection_id, error)

from __future__ import annotations

import asyncio
import json
import sys

from agent.control.protocol.router import ConnectionRouter
from agent.control.service import ControlService


class StdioAppServer:
    """在 stdin/stdout 上运行单连接 NDJSON app-server。"""

    def __init__(self, service: ControlService, *, max_message_bytes: int = 2 * 1024 * 1024) -> None:
        self._service = service
        self._max_message_bytes = max_message_bytes
        self._write_lock = asyncio.Lock()

    async def _send(self, message: dict[str, object]) -> None:
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._write_lock:
            await asyncio.to_thread(self._write, payload)

    @staticmethod
    def _write(payload: str) -> None:
        sys.stdout.write(payload)
        sys.stdout.flush()

    async def run(self) -> None:
        router = ConnectionRouter(self._service, self._send)
        try:
            while True:
                line = await asyncio.to_thread(sys.stdin.buffer.readline)
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
            await router.close()

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from uuid import uuid4

from agent.control.protocol.router import ConnectionRouter
from agent.control.service import ControlService
from agent.control.protocol.method import OutputReservation, RequestTransport
from .connection import _FrameReservation, _needs_delivery_receipt


class StdioAppServer(RequestTransport):
    """在 stdin/stdout 上运行单连接 NDJSON app-server。"""

    def __init__(self, service: ControlService, *, max_message_bytes: int = 2 * 1024 * 1024) -> None:
        self._service = service
        self._max_message_bytes = max_message_bytes
        self._write_lock = asyncio.Lock()
        self.connection_id = f"stdio:{uuid4().hex}"
        self._reservations: dict[tuple[str, str], _FrameReservation] = {}

    def reserve_input(self, session_id: str, input_id: str) -> OutputReservation:
        key = (session_id, input_id)
        if key not in self._reservations:
            self._reservations[key] = _FrameReservation(session_id, input_id)
        return self._reservations[key]

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
        written = (
            asyncio.get_running_loop().create_future()
            if page is not None and _needs_delivery_receipt(page, self._reservations)
            else None
        )
        if page is not None and written is not None:
            for reservation in self._reservations.values():
                reservation.observe(page, written)
        async with self._write_lock:
            try:
                await asyncio.to_thread(self._write, payload)
            except BaseException as error:
                if written is not None:
                    written.set_exception(error)
                raise
            else:
                if written is not None:
                    written.set_result(None)

    @staticmethod
    def _write(payload: str) -> None:
        sys.stdout.write(payload)
        sys.stdout.flush()

    async def run(self) -> None:
        router = ConnectionRouter(
            self._service, self._send, transport=self,
            send_message_page=self._send_message_page,
        )
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
            error = ConnectionError("stdio connection closed")
            for reservation in self._reservations.values():
                reservation.fail(error)
            self._reservations.clear()

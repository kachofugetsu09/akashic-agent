from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from agent.control.protocol.router import ConnectionRouter
from agent.control.service import ControlService


@dataclass(frozen=True)
class _PendingFrame:
    payload: bytes
    written: asyncio.Future[None] | None


class NdjsonConnection:
    """在有界 writer queue 上运行一条 JSON-RPC NDJSON 连接。"""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        service: ControlService,
        *,
        max_message_bytes: int,
        max_pending_requests: int,
        outbound_queue_size: int,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._queue: asyncio.Queue[_PendingFrame | None] = asyncio.Queue(
            outbound_queue_size
        )
        self._max_message_bytes = max_message_bytes
        self._router = ConnectionRouter(
            service,
            self.send,
            max_pending_requests=max_pending_requests,
        )
        self._request_tasks: set[asyncio.Task[None]] = set()

    async def send(self, message: dict[str, object]) -> None:
        encoded = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        written = (
            asyncio.get_running_loop().create_future()
            if message.get("method") == "turn/completed"
            else None
        )
        try:
            self._queue.put_nowait(_PendingFrame(encoded, written))
        except asyncio.QueueFull as exc:
            self._writer.close()
            raise ConnectionError("client outbound queue is full") from exc
        if written is not None:
            await asyncio.shield(written)

    async def run(self) -> None:
        writer_task = asyncio.create_task(self._write_loop(), name="control-writer")
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                if len(line) > self._max_message_bytes:
                    await self.send({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32600, "message": "Message too large"},
                    })
                    break
                task = asyncio.create_task(
                    self._router.handle_line(line),
                    name="control-request",
                )
                self._request_tasks.add(task)
                task.add_done_callback(self._on_request_done)
        finally:
            for task in self._request_tasks:
                task.cancel()
            if self._request_tasks:
                await asyncio.gather(*self._request_tasks, return_exceptions=True)
            self._request_tasks.clear()
            await self._router.close()
            if not writer_task.done():
                await self._queue.put(None)
            await writer_task
            self._writer.close()
            await self._writer.wait_closed()

    def _on_request_done(self, task: asyncio.Task[None]) -> None:
        self._request_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._writer.close()

    async def _write_loop(self) -> None:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            try:
                self._writer.write(frame.payload)
                await self._writer.drain()
            except BaseException as exc:
                if frame.written is not None and not frame.written.done():
                    frame.written.set_exception(exc)
                self._fail_pending_frames(exc)
                raise
            if frame.written is not None and not frame.written.done():
                frame.written.set_result(None)

    def _fail_pending_frames(self, error: BaseException) -> None:
        while not self._queue.empty():
            frame = self._queue.get_nowait()
            if (
                frame is not None
                and frame.written is not None
                and not frame.written.done()
            ):
                frame.written.set_exception(error)

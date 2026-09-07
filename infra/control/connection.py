from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import uuid4

from agent.control.protocol.router import ConnectionRouter
from agent.control.protocol.method import OutputReservation, RequestTransport
from agent.control.service import ControlService


@dataclass(frozen=True)
class _PendingFrame:
    payload: bytes
    written: asyncio.Future[None] | None


class _FrameReservation:
    """只接受同一连接完整 Output frame 的 writer-drain 回执。"""

    def __init__(self, session_id: str, input_id: str) -> None:
        self.session_id = session_id
        self.input_id = input_id
        self._expected: str | None = None
        self._delivered: set[str] = set()
        self._observed: dict[str, set[asyncio.Future[None]]] = {}
        self._waiters: dict[str, asyncio.Future[None]] = {}
        self._closed: BaseException | None = None

    def expect(self, message_id: str) -> None:
        if not message_id:
            raise ValueError("最终 Output message id 不能为空")
        if self._expected is not None and self._expected != message_id:
            raise ValueError("一次 Input 不能绑定多个最终 Output")
        self._expected = message_id

    async def wait_output(self, message_id: str) -> None:
        self.expect(message_id)
        if message_id in self._delivered:
            return
        if self._closed is not None:
            raise self._closed
        loop = asyncio.get_running_loop()
        waiter = self._waiters.setdefault(message_id, loop.create_future())
        await asyncio.shield(waiter)

    def observe(self, page: Mapping[str, object], written: asyncio.Future[None]) -> bool:
        """观察 Router 已确认的 message page，不从普通 RPC 猜测消息。"""
        rows = page.get("items")
        if not isinstance(rows, list):
            raise TypeError("message page items 必须是列表")
        tracked = False
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("message page item 必须是对象")
            if row.get("session_id") != self.session_id or not _complete_output(row):
                continue
            tracked = True
            message_id = row["id"]
            assert isinstance(message_id, str)
            observed = self._observed.setdefault(message_id, set())
            if written in observed:
                continue
            observed.add(written)
            written.add_done_callback(
                lambda future, identity=message_id: self._frame_done(identity, future)
            )
        return tracked

    def _frame_done(self, message_id: str, future: asyncio.Future[None]) -> None:
        try:
            future.result()
        except BaseException as error:
            self.fail(error)
        else:
            self._delivered.add(message_id)
            waiter = self._waiters.pop(message_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(None)

    def fail(self, error: BaseException) -> None:
        self._closed = error
        for waiter in self._waiters.values():
            if not waiter.done():
                waiter.set_exception(error)
        self._waiters.clear()


def _complete_output(row: Mapping[str, object]) -> bool:
    body = row.get("body")
    return (
        isinstance(row.get("id"), str)
        and isinstance(body, Mapping)
        and body.get("kind") == "output"
        and body.get("finish") == "complete"
        and isinstance(body.get("parts"), list)
    )


def _needs_delivery_receipt(
    page: Mapping[str, object], reservations: Mapping[tuple[str, str], _FrameReservation],
) -> bool:
    """只为 Router 标记的 message page 创建 writer future。"""
    rows = page.get("items")
    if not isinstance(rows, list):
        raise TypeError("message page items 必须是列表")
    sessions = {reservation.session_id for reservation in reservations.values()}
    return any(
        isinstance(row, Mapping)
        and _complete_output(row)
        and row.get("session_id") in sessions
        for row in rows
    )


class NdjsonConnection(RequestTransport):
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
            transport=self,
            send_message_page=self.send_message_page,
        )
        self._request_tasks: set[asyncio.Task[None]] = set()
        self.connection_id = f"ndjson:{uuid4().hex}"
        self._reservations: dict[tuple[str, str], _FrameReservation] = {}

    def reserve_input(self, session_id: str, input_id: str) -> OutputReservation:
        key = (session_id, input_id)
        existing = self._reservations.get(key)
        if existing is not None:
            return existing
        reservation = _FrameReservation(session_id, input_id)
        self._reservations[key] = reservation
        return reservation

    async def send(self, message: dict[str, object]) -> None:
        await self._send(message, page=None)

    async def send_message_page(
        self, message: dict[str, object], page: Mapping[str, object]
    ) -> None:
        """写入 Router 已确认的 message/read 或 messages.appended 页面。"""
        await self._send(message, page=page)

    async def _send(
        self, message: dict[str, object], *, page: Mapping[str, object] | None
    ) -> None:
        encoded = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        written = (
            asyncio.get_running_loop().create_future()
            if page is not None and _needs_delivery_receipt(page, self._reservations)
            else None
        )
        if page is not None and written is not None:
            for reservation in self._reservations.values():
                reservation.observe(page, written)
        try:
            self._queue.put_nowait(_PendingFrame(encoded, written))
        except asyncio.QueueFull as exc:
            if written is not None:
                written.set_exception(exc)
            self._writer.close()
            raise ConnectionError("client outbound queue is full") from exc

    async def run(self) -> None:
        writer_task = asyncio.create_task(self._write_loop(), name="control-writer")
        try:
            while True:
                try:
                    line = await self._reader.readline()
                except ValueError as error:
                    raise ConnectionError("control frame exceeds reader limit") from error
                if not line:
                    break
                if len(line) > self._max_message_bytes:
                    raise ConnectionError("control frame exceeds message limit")
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
            # 对端已关闭或连接已失败，不能等满队列和卡住的 drain 完成。
            _ = writer_task.cancel()
            self._writer.transport.abort()
            results = await asyncio.gather(writer_task, return_exceptions=True)
            self._fail_pending_frames(ConnectionError("control connection closed"))
            error = ConnectionError("control connection closed")
            for reservation in self._reservations.values():
                reservation.fail(error)
            self._reservations.clear()
            await self._writer.wait_closed()
            for result in results:
                if isinstance(result, Exception):
                    raise result

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

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import Future
from typing import Any, cast

DEFAULT_MAX_MESSAGE_BYTES = 2 * 1024 * 1024


class RemoteError(RuntimeError):
    def __init__(self, code: int, message: str, data: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data
        retryable = data.get("retryable") if isinstance(data, dict) else None
        self.retryable = retryable is True


class ConnectionClosedError(ConnectionError):
    pass


class ProtocolError(RuntimeError):
    pass


class SlowConsumerError(ConnectionError):
    pass


class _WireClient:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.pending: dict[int, asyncio.Future[object]] = {}
        self.turn_queues: dict[str, asyncio.Queue[object]] = {}
        self.notifications: asyncio.Queue[object] = asyncio.Queue(512)
        self.next_id = 1
        self.closed = False
        self.reader_task = asyncio.create_task(self._read(), name="akashic-sdk-reader")

    @classmethod
    async def connect(
        cls,
        endpoint: str,
        *,
        workspace_token: str | None = None,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> _WireClient:
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        reader_limit = max_message_bytes + 1
        if endpoint.count(":") == 1 and not endpoint.startswith("/"):
            host, raw_port = endpoint.rsplit(":", 1)
            reader, writer = await asyncio.open_connection(
                host,
                int(raw_port),
                limit=reader_limit,
            )
        else:
            reader, writer = await asyncio.open_unix_connection(
                endpoint,
                limit=reader_limit,
            )
        wire = cls(reader, writer)
        try:
            _ = await wire.request(
                "initialize",
                {
                    "protocolVersion": "1.0",
                    "clientInfo": {"name": "akashic-agent-sdk", "version": "0.1.0"},
                    "capabilities": {"reasoningEvents": False},
                    "workspaceToken": workspace_token,
                },
            )
            await wire.notify("initialized", {})
        except BaseException:
            await wire.close()
            raise
        return wire

    async def request(self, method: str, params: dict[str, object]) -> object:
        if self.closed:
            raise ConnectionClosedError("connection is closed")
        request_id = self.next_id
        self.next_id += 1
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        return await future

    async def notify(self, method: str, params: dict[str, object]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _write(self, payload: dict[str, object]) -> None:
        self.writer.write((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        await self.writer.drain()

    async def _read(self) -> None:
        failure: BaseException | None = None
        try:
            while line := await self.reader.readline():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSON-RPC frame must be object")
                message = cast(dict[str, Any], value)
                if "id" in message:
                    request_id = message["id"]
                    if not isinstance(request_id, int):
                        raise ValueError("response id must be int")
                    future = self.pending.pop(request_id, None)
                    if future is None:
                        raise ProtocolError(f"unknown response id: {request_id}")
                    if future.done():
                        continue
                    error = message.get("error")
                    if isinstance(error, dict):
                        future.set_exception(RemoteError(int(error["code"]), str(error["message"]), error.get("data")))
                    else:
                        future.set_result(message.get("result"))
                    continue
                params = message.get("params")
                if not isinstance(params, dict):
                    continue
                turn_id = params.get("turnId")
                if isinstance(turn_id, str):
                    queue = self.turn_queues.setdefault(turn_id, asyncio.Queue(512))
                    try:
                        queue.put_nowait(message)
                    except asyncio.QueueFull as exc:
                        raise SlowConsumerError(f"turn notification queue overflow: {turn_id}") from exc
                else:
                    try:
                        self.notifications.put_nowait(message)
                    except asyncio.QueueFull as exc:
                        raise SlowConsumerError("global notification queue overflow") from exc
                # 已缓冲的 readline 可连续同步返回，让活跃消费者有机会排空有界队列。
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = exc
        finally:
            self.closed = True
            reason = failure or ConnectionClosedError("server closed connection")
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(reason)
            self.pending.clear()
            for queue in [*self.turn_queues.values(), self.notifications]:
                if queue.full():
                    _ = queue.get_nowait()
                queue.put_nowait(reason)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.writer.close()
        await self.writer.wait_closed()
        self.reader_task.cancel()
        _ = await asyncio.gather(self.reader_task, return_exceptions=True)


class TurnHandle:
    def __init__(self, wire: _WireClient, thread_id: str, turn_id: str) -> None:
        self._wire = wire
        self.thread_id = thread_id
        self.id = turn_id
        self._terminal: dict[str, Any] | None = None

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        queue = self._wire.turn_queues.setdefault(self.id, asyncio.Queue(512))
        while self._terminal is None:
            event = await queue.get()
            if isinstance(event, BaseException):
                raise event
            if not isinstance(event, dict):
                raise ProtocolError("turn notification must be object")
            if event.get("method") == "turn/completed":
                params = event.get("params")
                assert isinstance(params, dict)
                turn = params.get("turn")
                assert isinstance(turn, dict)
                self._terminal = cast(dict[str, Any], turn)
            yield event

    def events(self) -> AsyncIterator[dict[str, Any]]:
        return self.stream()

    async def result(self) -> dict[str, Any]:
        if self._terminal is None:
            async for _ in self.stream():
                pass
        assert self._terminal is not None
        return self._terminal

    async def interrupt(self) -> dict[str, Any]:
        return cast(dict[str, Any], await self._wire.request(
            "turn/interrupt", {"threadId": self.thread_id, "turnId": self.id}
        ))


class Thread:
    def __init__(self, wire: _WireClient, record: dict[str, Any]) -> None:
        self._wire = wire
        self.record = record
        self.id = str(record["id"])

    async def turn(self, input_text: str) -> TurnHandle:
        record = cast(dict[str, Any], await self._wire.request(
            "turn/start", {"threadId": self.id, "input": input_text, "metadata": {}}
        ))
        turn_id = str(record["id"])
        _ = self._wire.turn_queues.setdefault(turn_id, asyncio.Queue(512))
        return TurnHandle(self._wire, self.id, turn_id)

    async def run(self, input_text: str) -> dict[str, Any]:
        return await (await self.turn(input_text)).result()

class AsyncAkashic:
    def __init__(self, wire: _WireClient) -> None:
        self._wire = wire

    @classmethod
    async def connect(
        cls,
        endpoint: str,
        *,
        workspace_token: str | None = None,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> AsyncAkashic:
        return cls(
            await _WireClient.connect(
                endpoint,
                workspace_token=workspace_token,
                max_message_bytes=max_message_bytes,
            )
        )

    async def thread_start(self, metadata: dict[str, object] | None = None) -> Thread:
        record = cast(dict[str, Any], await self._wire.request("thread/start", {"metadata": metadata or {}}))
        return Thread(self._wire, record)

    async def thread_resume(self, thread_id: str) -> Thread:
        record = cast(dict[str, Any], await self._wire.request("thread/resume", {"threadId": thread_id}))
        return Thread(self._wire, record)

    async def thread_list(self, *, cursor: str | None = None, limit: int = 50) -> dict[str, Any]:
        return cast(dict[str, Any], await self._wire.request("thread/list", {"cursor": cursor, "limit": limit}))

    async def thread_read(self, thread_id: str, *, include_turns: bool = True) -> dict[str, Any]:
        return cast(dict[str, Any], await self._wire.request(
            "thread/read", {"threadId": thread_id, "includeTurns": include_turns}
        ))

    async def thread_delete(self, thread_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], await self._wire.request("thread/delete", {"threadId": thread_id}))

    async def turn_read(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], await self._wire.request(
            "turn/read", {"threadId": thread_id, "turnId": turn_id}
        ))

    async def notifications(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._wire.notifications.get()
            if isinstance(event, BaseException):
                raise event
            if not isinstance(event, dict):
                raise ProtocolError("notification must be object")
            yield event

    async def close(self) -> None:
        await self._wire.close()

    async def __aenter__(self) -> AsyncAkashic:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="akashic-sdk-loop", daemon=True)
        self.thread.start()

    def run(self, coroutine: Any) -> Any:
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


class _SyncThread:
    def __init__(self, owner: Akashic, thread: Thread) -> None:
        self._owner = owner
        self._thread = thread
        self.id = thread.id

    def run(self, input_text: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._owner._runner.run(self._thread.run(input_text)))

    def turn(self, input_text: str) -> _SyncTurnHandle:
        handle = cast(TurnHandle, self._owner._runner.run(self._thread.turn(input_text)))
        return _SyncTurnHandle(self._owner, handle)

class _SyncTurnHandle:
    def __init__(self, owner: Akashic, handle: TurnHandle) -> None:
        self._owner = owner
        self._handle = handle
        self.thread_id = handle.thread_id
        self.id = handle.id

    def events(self) -> Iterator[dict[str, Any]]:
        stream = self._handle.events().__aiter__()
        while True:
            try:
                yield cast(dict[str, Any], self._owner._runner.run(stream.__anext__()))
            except StopAsyncIteration:
                return

    def result(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._owner._runner.run(self._handle.result()))

    def interrupt(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._owner._runner.run(self._handle.interrupt()))


class Akashic:
    def __init__(self, runner: _LoopThread, async_client: AsyncAkashic) -> None:
        self._runner = runner
        self._async = async_client

    @classmethod
    def connect(
        cls,
        endpoint: str,
        *,
        workspace_token: str | None = None,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> Akashic:
        runner = _LoopThread()
        return cls(
            runner,
            runner.run(
                AsyncAkashic.connect(
                    endpoint,
                    workspace_token=workspace_token,
                    max_message_bytes=max_message_bytes,
                )
            ),
        )

    def thread_start(self, metadata: dict[str, object] | None = None) -> _SyncThread:
        return _SyncThread(self, self._runner.run(self._async.thread_start(metadata)))

    def thread_resume(self, thread_id: str) -> _SyncThread:
        return _SyncThread(self, self._runner.run(self._async.thread_resume(thread_id)))

    def thread_list(self, *, cursor: str | None = None, limit: int = 50) -> dict[str, Any]:
        return cast(dict[str, Any], self._runner.run(self._async.thread_list(cursor=cursor, limit=limit)))

    def thread_read(self, thread_id: str, *, include_turns: bool = True) -> dict[str, Any]:
        return cast(dict[str, Any], self._runner.run(
            self._async.thread_read(thread_id, include_turns=include_turns)
        ))

    def thread_delete(self, thread_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._runner.run(self._async.thread_delete(thread_id)))

    def turn_read(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._runner.run(self._async.turn_read(thread_id, turn_id)))

    def close(self) -> None:
        self._runner.run(self._async.close())
        self._runner.close()

    def __enter__(self) -> Akashic:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

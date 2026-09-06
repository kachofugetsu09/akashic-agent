from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncGenerator, Iterator
from concurrent.futures import Future
from typing import Any, cast
from uuid import uuid4

DEFAULT_MAX_MESSAGE_BYTES = 2 * 1024 * 1024
logger = logging.getLogger(__name__)


class RemoteError(RuntimeError):
    def __init__(self, code: int, message: str, data: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data
        self.retryable = isinstance(data, dict) and data.get("retryable") is True


class ConnectionClosedError(ConnectionError):
    pass


class ProtocolError(RuntimeError):
    pass


class SlowConsumerError(ConnectionError):
    pass


class SessionSubscription:
    """读取一个 Session；游标由调用者在处理消息后保存。"""

    def __init__(self, wire: _WireClient, session_id: str, queue_size: int) -> None:
        self._wire = wire
        self.session_id = session_id
        self.id = uuid4().hex
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(queue_size)
        self._closed = False
        self._error: Exception | None = None
        self._iterator_lock = threading.Lock()
        self._iterator_claimed = False

    def _finish(self, error: Exception | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._error = error
        # 队列失败后从调用者已处理的游标补读，不能将丢弃的事件算作已读。
        while not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(None)

    def events(self) -> AsyncGenerator[dict[str, Any], None]:
        with self._iterator_lock:
            if self._iterator_claimed:
                raise RuntimeError("订阅只能有一个消费者；请从已保存的 seq 重新 session_follow")
            self._iterator_claimed = True
        return self._events()

    async def _events(self) -> AsyncGenerator[dict[str, Any], None]:
        while not self._closed:
            event = await self._queue.get()
            if event is None:
                break
            yield event
        if self._error is not None:
            raise self._error

    async def close(self) -> None:
        self._finish()
        await self._wire.unfollow(self)

    async def __aenter__(self) -> SessionSubscription:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class _WireClient:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.pending: dict[int, asyncio.Future[object]] = {}
        self.subscriptions: dict[str, SessionSubscription] = {}
        self.tasks: set[asyncio.Task[None]] = set()
        self._follow_lock = asyncio.Lock()
        self.next_id = 1
        self.closed = False
        self.reader_task = asyncio.create_task(self._read(), name="akashic-sdk-reader")

    @classmethod
    async def connect(
        cls, endpoint: str, *, workspace_token: str | None = None,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> _WireClient:
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        if endpoint.count(":") == 1 and not endpoint.startswith("/"):
            host, raw_port = endpoint.rsplit(":", 1)
            reader, writer = await asyncio.open_connection(host, int(raw_port), limit=max_message_bytes + 1)
        else:
            reader, writer = await asyncio.open_unix_connection(endpoint, limit=max_message_bytes + 1)
        wire = cls(reader, writer)
        try:
            await wire.request("initialize", {
                "protocolVersion": "2.0",
                "clientInfo": {"name": "akashic-agent-sdk", "version": "0.2.0"},
                "workspaceToken": workspace_token,
            })
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
        try:
            await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            return await future
        except BaseException:
            # 已发出的请求仍可收到响应；reader 会取走取消的 future。
            future.cancel()
            raise

    async def notify(self, method: str, params: dict[str, object]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _write(self, payload: dict[str, object]) -> None:
        self.writer.write((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        await self.writer.drain()

    async def follow(self, session_id: str, after_seq: int, queue_size: int) -> SessionSubscription:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        async with self._follow_lock:
            subscription = SessionSubscription(self, session_id, queue_size)
            self.subscriptions[subscription.id] = subscription
            try:
                await self.request("session/follow", {
                    "session_id": session_id, "after_seq": after_seq, "subscription_id": subscription.id,
                })
            except BaseException:
                subscription._finish()
                self._stop_later(subscription)
                raise
            # ACK 之后替换旧身份。旧通知不能污染新队列，旧 unfollow 也不能关闭新订阅。
            for previous in tuple(self.subscriptions.values()):
                if previous.session_id == session_id and previous is not subscription:
                    previous._finish()
                    self.subscriptions.pop(previous.id)
            return subscription

    async def unfollow(self, subscription: SessionSubscription) -> None:
        if self.subscriptions.pop(subscription.id, None) is not None and not self.closed:
            await self.request("session/unfollow", {
                "session_id": subscription.session_id, "subscription_id": subscription.id,
            })

    def _stop_later(self, subscription: SessionSubscription) -> None:
        task = asyncio.create_task(self.unfollow(subscription), name="akashic-sdk-unfollow")
        self.tasks.add(task)
        def finished(task: asyncio.Task[None]) -> None:
            self.tasks.remove(task)
            if not task.cancelled() and (error := task.exception()) is not None:
                logger.error("session unfollow failed", exc_info=error)
        task.add_done_callback(finished)

    async def _read(self) -> None:
        """单 reader 分发响应；慢订阅仅关闭自身，连接故障才终止所有等待者。"""
        failure: Exception = ConnectionClosedError("server closed connection")
        try:
            while line := await self.reader.readline():
                value = json.loads(line)
                if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
                    raise ProtocolError("JSON-RPC frame must be a version 2.0 object")
                message = cast(dict[str, Any], value)
                if "id" in message:
                    request_id = message["id"]
                    if not isinstance(request_id, int) or isinstance(request_id, bool):
                        raise ProtocolError("response id must be int")
                    future = self.pending.get(request_id)
                    if future is None:
                        raise ProtocolError(f"unknown response id: {request_id}")
                    if future.done():
                        del self.pending[request_id]
                        continue
                    error = message.get("error")
                    if isinstance(error, dict):
                        future.set_exception(RemoteError(int(error["code"]), str(error["message"]), error.get("data")))
                    else:
                        future.set_result(message["result"])
                    del self.pending[request_id]
                    continue
                params = message.get("params")
                if message.get("method") not in ("session/event", "session/error") or not isinstance(params, dict):
                    raise ProtocolError("unknown server notification")
                identity = params.get("subscription_id")
                if not isinstance(identity, str):
                    raise ProtocolError("subscription_id must be str")
                subscription = self.subscriptions.get(identity)
                if subscription is None or subscription._closed:
                    continue
                if message["method"] == "session/error":
                    error = params["error"]
                    subscription._finish(RemoteError(-32603, str(error["message"]), error))
                    self._stop_later(subscription)
                else:
                    event = params.get("event")
                    if not isinstance(event, dict) or event.get("session_id") != subscription.session_id:
                        raise ProtocolError("session event does not match subscription")
                    try:
                        subscription._queue.put_nowait(event)
                    except asyncio.QueueFull:
                        subscription._finish(SlowConsumerError(
                            f"session queue overflow; read again from your saved seq: {subscription.session_id}"
                        ))
                        self._stop_later(subscription)
                # 连续缓冲帧之间给活跃消费者一次调度机会。
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            failure = ProtocolError(f"invalid server frame: {exc}")
        except Exception as exc:
            failure = exc
        finally:
            self.closed = True
            self.writer.close()
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(failure)
            self.pending.clear()
            for subscription in self.subscriptions.values():
                subscription._finish(failure)

    async def close(self) -> None:
        self.closed = True
        self.writer.close()
        self.reader_task.cancel()
        await asyncio.gather(self.reader_task, return_exceptions=True)
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.writer.wait_closed()


class AsyncAkashic:
    """Message v2 客户端：发送得到持久化 ACK，订阅读取消息和瞬时回复状态。"""

    def __init__(self, wire: _WireClient) -> None:
        self._wire = wire

    @classmethod
    async def connect(
        cls, endpoint: str, *, workspace_token: str | None = None,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> AsyncAkashic:
        return cls(await _WireClient.connect(endpoint, workspace_token=workspace_token,
                                             max_message_bytes=max_message_bytes))

    async def request(self, method: str, params: dict[str, object]) -> object:
        return await self._wire.request(method, params)

    async def session_create(self) -> dict[str, Any]:
        return cast(dict[str, Any], await self.request("session/create", {}))

    async def session_list(self, *, cursor: list[str] | None = None, limit: int = 50) -> dict[str, Any]:
        return cast(dict[str, Any], await self.request("session/list", {"cursor": cursor, "limit": limit}))

    async def message_read(self, session_id: str, *, after_seq: int = -1,
                           through_seq: int | None = None, limit: int = 50) -> dict[str, Any]:
        return cast(dict[str, Any], await self.request("message/read", {
            "session_id": session_id, "after_seq": after_seq, "through_seq": through_seq, "limit": limit,
        }))

    async def message_send(self, session_id: str, text: str = "", *, message_id: str,
                           attachment_ids: list[str] | None = None,
                           reply_to_message_id: str | None = None, model_id: str | None = None,
                           reasoning_effort: str | None = None, retry_of: str | None = None) -> dict[str, Any]:
        return cast(dict[str, Any], await self.request("message/send", {
            "session_id": session_id, "message_id": message_id, "text": text,
            "attachment_ids": attachment_ids or [], "reply_to_message_id": reply_to_message_id,
            "model_id": model_id, "reasoning_effort": reasoning_effort, "retry_of": retry_of,
        }))

    async def session_follow(self, session_id: str, *, after_seq: int = -1,
                             queue_size: int = 512) -> SessionSubscription:
        return await self._wire.follow(session_id, after_seq, queue_size)

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


class _SyncSubscription:
    def __init__(self, runner: _LoopThread, subscription: SessionSubscription) -> None:
        self._runner = runner
        self._subscription = subscription
        self.session_id = subscription.session_id
        self.id = subscription.id

    def events(self) -> Iterator[dict[str, Any]]:
        stream = self._subscription.events()
        def iterate() -> Iterator[dict[str, Any]]:
            try:
                while True:
                    try:
                        yield self._runner.run(anext(stream))
                    except StopAsyncIteration:
                        return
            finally:
                self._runner.run(stream.aclose())
        return iterate()

    def close(self) -> None:
        self._runner.run(self._subscription.close())

    def __enter__(self) -> _SyncSubscription:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class Akashic:
    def __init__(self, runner: _LoopThread, async_client: AsyncAkashic) -> None:
        self._runner = runner
        self._async = async_client

    @classmethod
    def connect(cls, endpoint: str, *, workspace_token: str | None = None,
                max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES) -> Akashic:
        runner = _LoopThread()
        try:
            client = runner.run(AsyncAkashic.connect(endpoint, workspace_token=workspace_token,
                                                    max_message_bytes=max_message_bytes))
        except BaseException:
            runner.close()
            raise
        return cls(runner, client)

    def request(self, method: str, params: dict[str, object]) -> object:
        return self._runner.run(self._async.request(method, params))

    def session_create(self) -> dict[str, Any]:
        return self._runner.run(self._async.session_create())

    def session_list(self, *, cursor: list[str] | None = None, limit: int = 50) -> dict[str, Any]:
        return self._runner.run(self._async.session_list(cursor=cursor, limit=limit))

    def message_read(self, session_id: str, *, after_seq: int = -1,
                     through_seq: int | None = None, limit: int = 50) -> dict[str, Any]:
        return self._runner.run(self._async.message_read(session_id, after_seq=after_seq,
                                                        through_seq=through_seq, limit=limit))

    def message_send(self, session_id: str, text: str = "", *, message_id: str,
                     attachment_ids: list[str] | None = None,
                     reply_to_message_id: str | None = None, model_id: str | None = None,
                     reasoning_effort: str | None = None, retry_of: str | None = None) -> dict[str, Any]:
        return self._runner.run(self._async.message_send(
            session_id, text, message_id=message_id, attachment_ids=attachment_ids,
            reply_to_message_id=reply_to_message_id, model_id=model_id,
            reasoning_effort=reasoning_effort, retry_of=retry_of,
        ))

    def session_follow(self, session_id: str, *, after_seq: int = -1,
                       queue_size: int = 512) -> _SyncSubscription:
        return _SyncSubscription(self._runner, self._runner.run(
            self._async.session_follow(session_id, after_seq=after_seq, queue_size=queue_size),
        ))

    def close(self) -> None:
        try:
            self._runner.run(self._async.close())
        finally:
            self._runner.close()

    def __enter__(self) -> Akashic:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

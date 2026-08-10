from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, cast

from agent.control.protocol.limits import DEFAULT_CONTROL_FRAME_BYTES

TurnStreamValue = dict[str, Any] | BaseException


class RemoteControlError(RuntimeError):
    def __init__(self, code: int, message: str, data: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class ConnectionClosedError(ConnectionError):
    pass


class ClientTurnHandle:
    def __init__(
        self,
        client: ControlClient,
        thread_id: str,
        turn_id: str,
        record: dict[str, Any],
    ) -> None:
        self._client = client
        self.thread_id = thread_id
        self.id = turn_id
        self.record = record

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        queue = self._client._turn_queue(self.id)
        while True:
            event = await queue.get()
            if isinstance(event, BaseException):
                raise event
            yield event
            if event.get("method") == "turn/completed":
                return

    async def result(self) -> dict[str, Any]:
        async for event in self.events():
            if event.get("method") == "turn/completed":
                params = cast(dict[str, Any], event["params"])
                return cast(dict[str, Any], params["turn"])
        raise ConnectionClosedError("turn event stream closed without terminal event")

    async def interrupt(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            await self._client.request(
                "turn/interrupt", {"threadId": self.thread_id, "turnId": self.id}
            ),
        )


class ControlClient:
    """为 exec 和仓库测试提供单 reader 的异步 JSON-RPC 客户端。"""

    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[object]] = {}
        self._turn_queues: dict[str, asyncio.Queue[TurnStreamValue]] = {}
        self._terminal_turns: set[str] = set()
        self._notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue(512)
        self._next_id = 1
        self._closed = False
        self._close_error: BaseException | None = None

    @classmethod
    async def connect(
        cls,
        endpoint: str,
        *,
        workspace_token: str | None = None,
    ) -> ControlClient:
        if endpoint.count(":") == 1 and not endpoint.startswith("/"):
            host, port = endpoint.rsplit(":", 1)
            reader, writer = await asyncio.open_connection(
                host,
                int(port),
                limit=DEFAULT_CONTROL_FRAME_BYTES + 1,
            )
        else:
            reader, writer = await asyncio.open_unix_connection(
                endpoint,
                limit=DEFAULT_CONTROL_FRAME_BYTES + 1,
            )
        client = cls(reader, writer)
        client._reader_task = asyncio.create_task(
            client._read_loop(), name="control-client-reader"
        )
        try:
            await client.request(
                "initialize",
                {
                    "protocolVersion": "1.0",
                    "clientInfo": {
                        "name": "akashic-control-client",
                        "version": "0.1.0",
                    },
                    "capabilities": {"reasoningEvents": False},
                    "workspaceToken": workspace_token,
                },
            )
            await client.notify("initialized", {})
        except BaseException:
            await client.close()
            raise
        return client

    async def request(self, method: str, params: dict[str, object]) -> object:
        if self._closed:
            raise ConnectionClosedError("control connection is closed")
        request_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._write(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        return await future

    async def notify(self, method: str, params: dict[str, object]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def start_thread(
        self,
        metadata: dict[str, object] | None = None,
        *,
        runtime: str = "stable",
        plugin_rollout_capability: str = "",
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            await self.request(
                "thread/start",
                {
                    "metadata": metadata or {},
                    "runtime": runtime,
                    "pluginRolloutCapability": plugin_rollout_capability,
                },
            ),
        )

    async def start_turn(
        self,
        thread_id: str,
        input_text: str,
        *,
        runtime: str | None = None,
        detached: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> ClientTurnHandle:
        params: dict[str, object] = {
            "threadId": thread_id,
            "input": input_text,
            "metadata": metadata or {},
            "detached": detached,
        }
        if runtime is not None:
            params["runtime"] = runtime
        record = cast(
            dict[str, Any],
            await self.request("turn/start", params),
        )
        turn_id = str(record["id"])
        _ = self._turn_queue(turn_id)
        return ClientTurnHandle(self, thread_id, turn_id, record)

    def _turn_queue(self, turn_id: str) -> asyncio.Queue[TurnStreamValue]:
        queue = self._turn_queues.setdefault(turn_id, asyncio.Queue(512))
        if self._close_error is not None and queue.empty():
            queue.put_nowait(self._close_error)
        return queue

    async def notifications(self) -> AsyncIterator[dict[str, Any]]:
        while not self._closed:
            yield await self._notifications.get()

    async def _write(self, payload: dict[str, object]) -> None:
        self._writer.write(
            (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode()
        )
        await self._writer.drain()

    async def _read_loop(self) -> None:
        error: BaseException | None = None
        try:
            while line := await self._reader.readline():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("JSON-RPC frame must be object")
                message = cast(dict[str, Any], payload)
                if "id" in message:
                    request_id = message["id"]
                    future = self._pending.pop(request_id, None)
                    if future is None:
                        raise ValueError(f"unknown response id: {request_id}")
                    if "error" in message:
                        remote = cast(dict[str, Any], message["error"])
                        future.set_exception(
                            RemoteControlError(
                                int(remote["code"]),
                                str(remote["message"]),
                                remote.get("data"),
                            )
                        )
                    else:
                        future.set_result(message.get("result"))
                    continue
                params = message.get("params")
                if isinstance(params, dict) and isinstance(params.get("turnId"), str):
                    turn_id = params["turnId"]
                    queue = self._turn_queues.setdefault(turn_id, asyncio.Queue(512))
                    try:
                        queue.put_nowait(message)
                    except asyncio.QueueFull as exc:
                        raise ConnectionError(
                            f"turn notification queue overflow: {turn_id}"
                        ) from exc
                    if message.get("method") == "turn/completed":
                        self._terminal_turns.add(turn_id)
                else:
                    try:
                        self._notifications.put_nowait(message)
                    except asyncio.QueueFull as exc:
                        raise ConnectionError(
                            "global notification queue overflow"
                        ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
        finally:
            self._closed = True
            reason = error or ConnectionClosedError("control server closed connection")
            self._close_error = reason
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(reason)
            self._pending.clear()
            for turn_id, queue in self._turn_queues.items():
                if turn_id in self._terminal_turns:
                    continue
                try:
                    queue.put_nowait(reason)
                except asyncio.QueueFull:
                    _ = queue.get_nowait()
                    queue.put_nowait(reason)

    async def close(self) -> None:
        self._closed = True
        if not self._writer.is_closing():
            self._writer.close()
            await self._writer.wait_closed()
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)

    async def __aenter__(self) -> ControlClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

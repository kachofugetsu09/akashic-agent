from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import ValidationError

from agent.control.errors import (
    RuntimeClosedError,
    ThreadBusyError,
    ThreadNotFoundError,
    TurnNotFoundError,
)
from agent.control.protocol.errors import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    INCOMPATIBLE_VERSION,
    METHOD_NOT_FOUND,
    NOT_INITIALIZED,
    SERVER_OVERLOADED,
    THREAD_BUSY,
    THREAD_NOT_FOUND,
    TURN_NOT_FOUND,
    JsonRpcError,
)
from agent.control.protocol.models import METHOD_PARAMS, InitializeParams, StrictModel
from agent.control.service import ControlService

logger = logging.getLogger(__name__)
JsonObject = dict[str, Any]
SendMessage = Callable[[dict[str, object]], Awaitable[None]]


class ConnectionRouter:
    """校验并分发一条连接上的 JSON-RPC 请求和通知。"""

    def __init__(
        self,
        service: ControlService,
        send: SendMessage,
        *,
        max_pending_requests: int = 64,
    ) -> None:
        self._service = service
        self._send = send
        self._pending = asyncio.Semaphore(max_pending_requests)
        self._state = "new"
        self._initialized_seen = False
        self._event_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    async def handle_line(self, line: bytes) -> None:
        """解析单条 NDJSON frame，并在边界返回标准错误。"""

        # 1. 严格解析 UTF-8 JSON object。
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            await self._send(JsonRpcError(-32700, "Parse error").envelope(None))
            return
        if not isinstance(payload, dict):
            await self._send(JsonRpcError(INVALID_REQUEST, "Invalid Request").envelope(None))
            return

        # 2. notification 同步处理，请求由 transport 允许并发调度。
        request = cast(JsonObject, payload)
        request_id = request.get("id")
        if request_id is None:
            await self._handle_notification(request)
            return
        if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
            await self._send(JsonRpcError(INVALID_REQUEST, "Invalid request id").envelope(None))
            return
        if self._pending.locked():
            await self._send(JsonRpcError(SERVER_OVERLOADED, "Server overloaded", {"retryable": True}).envelope(request_id))
            return
        async with self._pending:
            await self._handle_request(request, request_id)

    async def _handle_notification(self, request: JsonObject) -> None:
        if request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
            return
        method = cast(str, request["method"])
        if method == "initialized" and self._state == "initialized":
            self._initialized_seen = True
            return
        logger.warning("忽略无效 JSON-RPC notification method=%s state=%s", method, self._state)

    async def _handle_request(self, request: JsonObject, request_id: str | int) -> None:
        try:
            result = await self._dispatch(request)
        except JsonRpcError as exc:
            await self._send(exc.envelope(request_id))
        except ThreadNotFoundError as exc:
            await self._send(JsonRpcError(THREAD_NOT_FOUND, str(exc)).envelope(request_id))
        except ThreadBusyError as exc:
            await self._send(JsonRpcError(THREAD_BUSY, str(exc), {"retryable": True}).envelope(request_id))
        except TurnNotFoundError as exc:
            await self._send(JsonRpcError(TURN_NOT_FOUND, str(exc)).envelope(request_id))
        except RuntimeClosedError as exc:
            await self._send(JsonRpcError(SERVER_OVERLOADED, str(exc), {"retryable": True}).envelope(request_id))
        except Exception:
            logger.exception("JSON-RPC handler failed request_id=%r", request_id)
            await self._send(JsonRpcError(INTERNAL_ERROR, "Internal error").envelope(request_id))
        else:
            await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})
            await self._post_response_notifications(request, result)

    async def _dispatch(self, request: JsonObject) -> object:
        """验证 request envelope 和 method params 后调用 application service。"""

        # 1. 校验 JSON-RPC envelope。
        if request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
            raise JsonRpcError(INVALID_REQUEST, "Invalid Request")
        unknown = set(request) - {"jsonrpc", "id", "method", "params"}
        if unknown:
            raise JsonRpcError(INVALID_REQUEST, f"Unknown request fields: {', '.join(sorted(unknown))}")
        method = cast(str, request["method"])
        model_type = METHOD_PARAMS.get(method)
        if model_type is None:
            raise JsonRpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

        # 2. 在协议边界一次性建立 typed params。
        raw_params = request.get("params", {})
        if not isinstance(raw_params, dict):
            raise JsonRpcError(INVALID_PARAMS, "params must be an object")
        if method == "initialize" and raw_params.get("protocolVersion") != "1.0":
            raise JsonRpcError(
                INCOMPATIBLE_VERSION,
                "Unsupported protocol version",
                {"supported": ["1.0"]},
            )
        try:
            params = model_type.model_validate(raw_params)
        except ValidationError as exc:
            raise JsonRpcError(INVALID_PARAMS, "Invalid params", {"issues": exc.errors(include_url=False)}) from exc

        # 3. initialize 是唯一允许进入 new 状态的请求。
        if method == "initialize":
            if self._state != "new":
                raise JsonRpcError(INVALID_REQUEST, "initialize may only be sent once")
            init = cast(InitializeParams, params)
            result = self._service.initialize(init)
            self._state = "initialized"
            return result
        if self._state != "initialized" or not self._initialized_seen:
            raise JsonRpcError(NOT_INITIALIZED, "Client must complete initialize/initialized")

        return await self._call_method(method, params)

    async def _call_method(self, method: str, params: StrictModel) -> object:
        values = params.model_dump()
        if method == "server/status":
            return self._service.status()
        if method == "thread/start":
            return self._service.start_thread(values["metadata"])
        if method == "thread/resume":
            return self._service.resume_thread(values["threadId"])
        if method == "thread/list":
            return self._service.list_threads(values["cursor"], values["limit"])
        if method == "thread/read":
            return self._service.read_thread(values["threadId"], values["includeTurns"])
        if method == "thread/delete":
            return self._service.delete_thread(values["threadId"])
        if method == "thread/consolidate/start":
            operation = self._service.start_consolidation(values["threadId"])
            task = asyncio.create_task(
                self._forward_operation(operation),
                name=f"control-operation:{operation.id}",
            )
            self._event_tasks.add(task)
            task.add_done_callback(self._event_tasks.discard)
            return operation.record()
        if method == "turn/read":
            return self._service.read_turn(values["threadId"], values["turnId"])
        if method == "turn/interrupt":
            return await self._service.interrupt_turn(values["threadId"], values["turnId"])
        if method == "turn/start":
            handle = await self._service.start_turn(
                values["threadId"], values["input"], values["metadata"]
            )
            task = asyncio.create_task(self._forward_events(handle), name=f"control-events:{handle.id}")
            self._event_tasks.add(task)
            task.add_done_callback(self._event_tasks.discard)
            return handle.record()
        if method == "plugin/disable-and-drain":
            return await self._service.disable_and_drain_plugin(values["pluginId"])
        raise AssertionError(f"unhandled protocol method: {method}")

    async def _post_response_notifications(
        self,
        request: JsonObject,
        result: object,
    ) -> None:
        method = request.get("method")
        if not isinstance(result, dict):
            return
        if method == "thread/start":
            await self._send({"jsonrpc": "2.0", "method": "thread/started", "params": {"thread": result}})
        elif method == "thread/delete":
            await self._send({"jsonrpc": "2.0", "method": "thread/deleted", "params": result})

    async def _forward_operation(self, operation: Any) -> None:
        result = await asyncio.shield(operation.task)
        await self._send(
            {
                "jsonrpc": "2.0",
                "method": "operation/completed",
                "params": {"operation": result},
            }
        )

    async def _forward_events(self, handle: Any) -> None:
        try:
            async for event in handle.events():
                await self._send(event.to_notification())
                if event.method == "turn/completed":
                    self._service.notify_turn_delivered(handle.id)
        except asyncio.CancelledError:
            self._service.notify_turn_delivery_failed(
                handle.id,
                "connection closed before terminal delivery",
            )
            raise
        except Exception as exc:
            self._service.notify_turn_delivery_failed(handle.id, str(exc))
            logger.exception("turn event forwarding failed turn=%s", handle.id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in self._event_tasks:
            task.cancel()
        if self._event_tasks:
            await asyncio.gather(*self._event_tasks, return_exceptions=True)
        self._event_tasks.clear()

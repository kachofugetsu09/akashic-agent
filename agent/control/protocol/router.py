from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from typing import Any, cast

from pydantic import ValidationError

from agent.control.errors import (
    ControlAdmissionError,
    RuntimeClosedError,
    PluginManagementError,
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
    PLUGIN_OPERATION_FAILED,
    JsonRpcError,
)
from agent.control.protocol.models import METHOD_PARAMS, InitializeParams, StrictModel, MessageSendParams
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
        self._subscriptions: dict[str, tuple[str, asyncio.Task[None] | None]] = {}
        self._subscription_lock = asyncio.Lock()
        self._closed = False
        conflicts = METHOD_PARAMS.keys() & service.methods.keys()
        if conflicts:
            raise ValueError(f"控制方法已经存在: {sorted(conflicts)}")
        self._method_params = {**METHOD_PARAMS, **{
            name: method.params for name, method in service.methods.items()
        }}

    async def handle_line(self, line: bytes) -> None:
        """解析单条 NDJSON frame，并在边界返回标准错误。"""

        # 1. 严格解析 UTF-8 JSON object。
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            await self._send(JsonRpcError(-32700, "Parse error").envelope(None))
            return
        if not isinstance(payload, dict):
            await self._send(
                JsonRpcError(INVALID_REQUEST, "Invalid Request").envelope(None)
            )
            return

        # 2. notification 同步处理，请求由 transport 允许并发调度。
        request = cast(JsonObject, payload)
        request_id = request.get("id")
        if request_id is None:
            await self._handle_notification(request)
            return
        if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
            await self._send(
                JsonRpcError(INVALID_REQUEST, "Invalid request id").envelope(None)
            )
            return
        if self._pending.locked():
            await self._send(
                JsonRpcError(
                    SERVER_OVERLOADED, "Server overloaded", {"retryable": True}
                ).envelope(request_id)
            )
            return
        async with self._pending:
            await self._handle_request(request, request_id)

    async def _handle_notification(self, request: JsonObject) -> None:
        if request.get("jsonrpc") != "2.0" or not isinstance(
            request.get("method"), str
        ):
            return
        method = cast(str, request["method"])
        if method == "initialized" and self._state == "awaiting_initialized":
            self._state = "ready"
            return
        logger.warning(
            "忽略无效 JSON-RPC notification method=%s state=%s", method, self._state
        )

    async def _handle_request(self, request: JsonObject, request_id: str | int) -> None:
        try:
            result = await self._dispatch(request)
        except JsonRpcError as exc:
            await self._send(exc.envelope(request_id))
        except ThreadNotFoundError as exc:
            await self._send(
                JsonRpcError(THREAD_NOT_FOUND, str(exc)).envelope(request_id)
            )
        except ThreadBusyError as exc:
            await self._send(
                JsonRpcError(THREAD_BUSY, str(exc), {"retryable": True}).envelope(
                    request_id
                )
            )
        except ControlAdmissionError as exc:
            await self._send(
                JsonRpcError(
                    SERVER_OVERLOADED,
                    str(exc),
                    {"retryable": True, "failure": exc.failure_type},
                ).envelope(request_id)
            )
        except TurnNotFoundError as exc:
            await self._send(
                JsonRpcError(TURN_NOT_FOUND, str(exc)).envelope(request_id)
            )
        except PluginManagementError as exc:
            await self._send(
                JsonRpcError(PLUGIN_OPERATION_FAILED, str(exc)).envelope(request_id)
            )
        except ValueError as exc:
            await self._send(
                JsonRpcError(INVALID_PARAMS, str(exc)).envelope(request_id)
            )
        except RuntimeClosedError as exc:
            await self._send(
                JsonRpcError(SERVER_OVERLOADED, str(exc), {"retryable": True}).envelope(
                    request_id
                )
            )
        except Exception:
            logger.exception("JSON-RPC handler failed request_id=%r", request_id)
            await self._send(
                JsonRpcError(INTERNAL_ERROR, "Internal error").envelope(request_id)
            )
        else:
            await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})
            await self._post_response_notifications(request, result)

    async def _dispatch(self, request: JsonObject) -> object:
        """验证 request envelope 和 method params 后调用 application service。"""

        # 1. 校验 JSON-RPC envelope。
        if request.get("jsonrpc") != "2.0" or not isinstance(
            request.get("method"), str
        ):
            raise JsonRpcError(INVALID_REQUEST, "Invalid Request")
        unknown = set(request) - {"jsonrpc", "id", "method", "params"}
        if unknown:
            raise JsonRpcError(
                INVALID_REQUEST, f"Unknown request fields: {', '.join(sorted(unknown))}"
            )
        method = cast(str, request["method"])
        model_type = self._method_params.get(method)
        if model_type is None:
            raise JsonRpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

        # 2. 在协议边界一次性建立 typed params。
        raw_params = request.get("params", {})
        if not isinstance(raw_params, dict):
            raise JsonRpcError(INVALID_PARAMS, "params must be an object")
        if method == "initialize" and raw_params.get("protocolVersion") != "2.0":
            raise JsonRpcError(
                INCOMPATIBLE_VERSION,
                "Unsupported protocol version",
                {"supported": ["2.0"]},
            )
        try:
            params = model_type.model_validate(raw_params)
        except ValidationError as exc:
            raise JsonRpcError(
                INVALID_PARAMS,
                "Invalid params",
                {"issues": exc.errors(include_url=False)},
            ) from exc

        # 3. initialize 是唯一允许进入 new 状态的请求。
        if method == "initialize":
            if self._state != "new":
                raise JsonRpcError(INVALID_REQUEST, "initialize may only be sent once")
            init = cast(InitializeParams, params)
            result = self._service.initialize(init)
            self._state = "awaiting_initialized"
            return result
        if self._state != "ready":
            raise JsonRpcError(
                NOT_INITIALIZED, "Client must complete initialize/initialized"
            )

        return await self._call_method(method, params)

    async def _call_method(self, method: str, params: StrictModel) -> object:
        operation = self._service.methods.get(method)
        if operation is not None:
            return await operation.call(params)
        values = params.model_dump()
        if method == "server/status":
            return self._service.status()
        if method == "session/create":
            return self._service.create_session()
        if method == "session/list":
            return self._service.list_sessions(values["cursor"], values["limit"])
        if method == "message/read":
            return self._service.read_messages(values["session_id"], values["after_seq"],
                                                values["through_seq"], values["limit"])
        if method == "message/send":
            return await self._service.send_message(cast(MessageSendParams, params))
        if method == "session/follow":
            session_id, subscription_id = values["session_id"], values["subscription_id"]
            async with self._subscription_lock:
                await self._stop_subscription(session_id)
                if self._closed:
                    raise RuntimeError("连接已经关闭")
                self._subscriptions[session_id] = (subscription_id, None)
            return {"version": 2, "session_id": session_id, "subscription_id": subscription_id,
                    "after_seq": values["after_seq"]}
        if method == "session/unfollow":
            async with self._subscription_lock:
                existing = self._subscriptions.get(values["session_id"])
                if existing is not None and existing[0] == values["subscription_id"]:
                    await self._stop_subscription(values["session_id"])
            return {"session_id": values["session_id"], "subscription_id": values["subscription_id"]}
        if method == "plugin/status":
            return self._service.plugin_status()
        if method == "plugin/update":
            return self._service.plugin_update(values["update_id"])
        if method == "plugin/install":
            return await self._service.install_plugin(values["source"], values["marketplace"],
                values["ref"], values["sparse"], values["update_id"])
        if method == "plugin/promote":
            return await self._service.promote_plugin(values["update_id"])
        if method == "plugin/discard":
            return await self._service.discard_plugin(values["update_id"])
        if method == "plugin/disable-and-drain":
            return await self._service.disable_and_drain_plugin(values["plugin_id"])
        if method == "plugin/uninstall":
            return await self._service.uninstall_plugin(values["plugin_id"])
        raise AssertionError(f"unhandled protocol method: {method}")

    async def _post_response_notifications(self, request: JsonObject, result: object) -> None:
        """follow ACK 先进入传输队列，再启动订阅；客户端能建立对应的读取 owner。"""
        if request["method"] != "session/follow":
            return
        assert isinstance(result, dict)
        session_id = cast(str, result["session_id"])
        identity = cast(str, result["subscription_id"])
        if self._closed or self._subscriptions.get(session_id) != (identity, None):
            return
        task = asyncio.create_task(self._forward_session(session_id, identity, cast(int, result["after_seq"])),
                                   name="control-follow:" + session_id)
        self._subscriptions[session_id] = (identity, task)
        task.add_done_callback(self._subscription_finished)

    @staticmethod
    def _subscription_finished(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("消息订阅传输失败", exc_info=error)

    async def _forward_session(self, session_id: str, identity: str, after_seq: int) -> None:
        """每个 Session 独立读取；旧订阅身份不能进入新订阅的消费队列。"""
        try:
            async with aclosing(self._service.follow(session_id, after_seq)) as feed:
                async for event in feed:
                    await self._send({"jsonrpc": "2.0", "method": "session/event",
                        "params": {"subscription_id": identity, "event": event}})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("消息订阅读取失败 session=%s", session_id)
            await self._send({"jsonrpc": "2.0", "method": "session/error", "params": {
                "session_id": session_id, "subscription_id": identity,
                "error": {"type": type(error).__name__, "message": str(error)},
            }})

    async def _stop_subscription(self, session_id: str) -> None:
        previous = self._subscriptions.pop(session_id, None)
        if previous is not None and previous[1] is not None:
            task = previous[1]
            _ = task.cancel()
            # done callback 已记录传输错误；独立订阅的清理仍须全部完成。
            await asyncio.gather(task, return_exceptions=True)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with self._subscription_lock:
            for session_id in tuple(self._subscriptions):
                await self._stop_subscription(session_id)

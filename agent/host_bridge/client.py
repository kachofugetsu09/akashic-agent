from __future__ import annotations

import asyncio
import base64
import contextlib
import uuid
from pathlib import Path
from typing import Any

import grpc

from agent.host_bridge.protocol import SERVICE_NAME
from agent.host_bridge.protocol import decode_message
from agent.host_bridge.protocol import deserialize_message
from agent.host_bridge.protocol import encode_message
from agent.host_bridge.protocol import serialize_message
from agent.tools.unified_exec import ExecutionCleanupFailure
from agent.tools.unified_exec import ExecutionCleanupReport
from agent.tools.unified_exec import ExecutionResult
from agent.tools.base import ToolResult

_HEARTBEAT_INTERVAL_S = 2.0


class HostBridgeShellProcessManager:
    """Preserve ShellProcessManager semantics through a host UDS bridge."""

    def __init__(self, socket_path: Path, boot_id: str, token: str) -> None:
        if not socket_path.is_absolute():
            raise ValueError("Host Bridge socket 必须是绝对路径")
        if not boot_id:
            raise ValueError("Host Bridge boot_id 不能为空")
        if not token:
            raise ValueError("Host Bridge token 不能为空")
        self._socket_path = socket_path
        self._boot_id = boot_id
        self._token = token
        self._manager_id = uuid.uuid4().hex
        self._channel = grpc.aio.insecure_channel(
            f"unix:{socket_path}",
            options=(
                ("grpc.max_receive_message_length", 16 * 1024 * 1024),
                ("grpc.max_send_message_length", 16 * 1024 * 1024),
            ),
        )
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._lease_error: Exception | None = None
        self._closed = False

    async def probe(self) -> dict[str, Any]:
        return await self._call("Probe", {})

    async def exec_command(
        self,
        *,
        command: str,
        argv: list[str],
        cwd: Path | None,
        env: dict[str, str],
        tty: bool,
        yield_time_ms: int,
        max_output_tokens: int,
        hard_timeout_s: int,
        owner_session_key: str,
    ) -> ExecutionResult:
        payload = await self._call(
            "Exec",
            {
                "command": command,
                "argv": argv,
                "cwd": None if cwd is None else str(cwd),
                "env": env,
                "tty": tty,
                "yieldTimeMs": yield_time_ms,
                "maxOutputTokens": max_output_tokens,
                "hardTimeoutS": hard_timeout_s,
                "ownerSessionKey": owner_session_key,
            },
        )
        return _execution_result(payload)

    async def write_stdin(
        self,
        *,
        execution_id: int,
        chars: str,
        yield_time_ms: int,
        max_output_tokens: int,
        owner_session_key: str,
    ) -> ExecutionResult:
        payload = await self._call(
            "WriteStdin",
            {
                "executionId": execution_id,
                "chars": chars,
                "yieldTimeMs": yield_time_ms,
                "maxOutputTokens": max_output_tokens,
                "ownerSessionKey": owner_session_key,
            },
        )
        return _execution_result(payload)

    async def terminate_execution(
        self,
        execution_id: int,
        *,
        owner_session_key: str,
    ) -> bool:
        payload = await self._call(
            "Stop",
            {
                "executionId": execution_id,
                "ownerSessionKey": owner_session_key,
            },
        )
        return bool(payload["stopped"])

    async def terminate_owner(
        self,
        owner_session_key: str,
    ) -> ExecutionCleanupReport:
        payload = await self._call(
            "TerminateOwner",
            {"ownerSessionKey": owner_session_key},
        )
        return _cleanup_report(payload)

    async def shutdown(self) -> ExecutionCleanupReport:
        if self._closed:
            return ExecutionCleanupReport((), (), ())
        payload = await self._call("ShutdownManager", {})
        self._closed = True
        await self.close_transport()
        return _cleanup_report(payload)

    async def close_transport(self) -> None:
        """Close only the client transport after an RPC admission failure."""

        self._closed = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
        await self._channel.close()

    async def active_execution_ids(self) -> list[int]:
        payload = await self._call("ActiveExecutions", {})
        return [int(item) for item in payload["executionIds"]]

    async def execute_file_tool(
        self,
        operation: str,
        *,
        allowed_dir: Path | None,
        arguments: dict[str, Any],
    ) -> str | ToolResult:
        payload = await self._call(
            "FileTool",
            {
                "operation": operation,
                "allowedDir": None if allowed_dir is None else str(allowed_dir),
                "arguments": arguments,
            },
        )
        if payload["resultType"] == "text":
            return str(payload["text"])
        if payload["resultType"] != "toolResult":
            raise RuntimeError("Host Bridge 返回未知文件工具结果")
        return ToolResult(
            text=str(payload["text"]),
            content_blocks=list(payload["contentBlocks"]),
            mobile_attention=payload.get("mobileAttention"),
        )

    async def _call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Host Bridge manager 已关闭")
        if method != "Heartbeat" and self._lease_error is not None:
            raise RuntimeError(f"Host Bridge lease 已失效: {self._lease_error}")
        self._ensure_heartbeat()
        request = encode_message(
            {
                "bootId": self._boot_id,
                "managerId": self._manager_id,
                "token": self._token,
                **payload,
            }
        )
        call = self._channel.unary_unary(
            f"/{SERVICE_NAME}/{method}",
            request_serializer=serialize_message,
            response_deserializer=deserialize_message,
        )
        try:
            response = await call(
                request,
                timeout=5 if method in {"Probe", "Heartbeat"} else None,
            )
        except grpc.aio.AioRpcError as exc:
            raise RuntimeError(
                f"Host Bridge {method} 失败: {exc.code().name}: {exc.details()}"
            ) from exc
        document = decode_message(response)
        if document.get("ok") is not True:
            raise RuntimeError(str(document.get("error") or "Host Bridge 未知错误"))
        return document

    def _ensure_heartbeat(self) -> None:
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(),
                name=f"host-bridge-heartbeat:{self._manager_id}",
            )

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
                await self._call("Heartbeat", {})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._lease_error = exc


def _execution_result(payload: dict[str, Any]) -> ExecutionResult:
    return ExecutionResult(
        output=base64.b64decode(str(payload["outputBase64"]), validate=True),
        wall_time_ms=int(payload["wallTimeMs"]),
        original_token_count=int(payload["originalTokenCount"]),
        output_omitted_bytes=int(payload["outputOmittedBytes"]),
        execution_id=(
            None if payload.get("executionId") is None else int(payload["executionId"])
        ),
        exit_code=None if payload.get("exitCode") is None else int(payload["exitCode"]),
        output_path=(
            None if payload.get("outputPath") is None else str(payload["outputPath"])
        ),
        finish_reason=str(payload["finishReason"]),
    )


def _cleanup_report(payload: dict[str, Any]) -> ExecutionCleanupReport:
    return ExecutionCleanupReport(
        attempted_execution_ids=tuple(int(item) for item in payload["attempted"]),
        cleaned_execution_ids=tuple(int(item) for item in payload["cleaned"]),
        failures=tuple(
            ExecutionCleanupFailure(
                execution_id=int(item["executionId"]),
                error_type=str(item["errorType"]),
                message=str(item["message"]),
            )
            for item in payload["failures"]
        ),
    )

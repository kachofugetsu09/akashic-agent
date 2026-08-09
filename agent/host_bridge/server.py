from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import grpc

from agent.host_bridge.protocol import SERVICE_NAME
from agent.host_bridge.protocol import decode_message
from agent.host_bridge.protocol import deserialize_message
from agent.host_bridge.protocol import encode_message
from agent.host_bridge.protocol import serialize_message
from agent.tools.unified_exec import ExecutionCleanupReport
from agent.tools.unified_exec import ExecutionResult
from agent.tools.unified_exec import ShellProcessManager

_RpcHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class _ManagerLease:
    manager: ShellProcessManager
    last_seen: float


class HostBridgeService:
    """Own host shell managers and reap them when their Core lease expires."""

    def __init__(self, token: str, lease_timeout_s: float) -> None:
        if not token:
            raise ValueError("Host Bridge token 不能为空")
        if lease_timeout_s <= 0:
            raise ValueError("lease timeout 必须大于零")
        self._token = token
        self._lease_timeout_s = lease_timeout_s
        self._managers: dict[tuple[str, str], _ManagerLease] = {}
        self._lock = asyncio.Lock()

    def rpc_handlers(self) -> grpc.GenericRpcHandler:
        methods: dict[str, _RpcHandler] = {
            "Probe": self.probe,
            "Heartbeat": self.heartbeat,
            "Exec": self.exec_command,
            "WriteStdin": self.write_stdin,
            "Stop": self.stop,
            "TerminateOwner": self.terminate_owner,
            "ShutdownManager": self.shutdown_manager,
            "ActiveExecutions": self.active_executions,
        }
        return grpc.method_handlers_generic_handler(
            SERVICE_NAME,
            {
                name: grpc.unary_unary_rpc_method_handler(
                    self._rpc(handler),
                    request_deserializer=deserialize_message,
                    response_serializer=serialize_message,
                )
                for name, handler in methods.items()
            },
        )

    async def reap_expired(self) -> None:
        while True:
            await asyncio.sleep(min(2.0, self._lease_timeout_s / 2))
            cutoff = time.monotonic() - self._lease_timeout_s
            async with self._lock:
                expired = [
                    (key, lease)
                    for key, lease in self._managers.items()
                    if lease.last_seen < cutoff
                ]
                for key, _lease in expired:
                    del self._managers[key]
            for _key, lease in expired:
                await lease.manager.shutdown()

    async def shutdown(self) -> None:
        async with self._lock:
            leases = list(self._managers.values())
            self._managers.clear()
        for lease in leases:
            await lease.manager.shutdown()

    async def probe(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ = await self._lease(payload)
        return {"capabilities": ["exec", "pty", "stdin", "stop", "lease"]}

    async def heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ = await self._lease(payload)
        return {"alive": True}

    async def exec_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease = await self._lease(payload)
        argv = payload.get("argv")
        requested_env = payload.get("env")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(item, str) and item for item in argv
        ):
            raise ValueError("Host Bridge argv 必须是非空 string array")
        if not isinstance(requested_env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in requested_env.items()
        ):
            raise ValueError("Host Bridge env 必须是 string map")
        cwd_text = payload.get("cwd")
        result = await lease.manager.exec_command(
            command=_required_string(payload, "command"),
            argv=argv,
            cwd=None if cwd_text is None else Path(str(cwd_text)),
            env=_host_environment(requested_env),
            tty=bool(payload["tty"]),
            yield_time_ms=int(payload["yieldTimeMs"]),
            max_output_tokens=int(payload["maxOutputTokens"]),
            hard_timeout_s=int(payload["hardTimeoutS"]),
            owner_session_key=_required_string(payload, "ownerSessionKey"),
        )
        return _result_payload(result)

    async def write_stdin(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease = await self._lease(payload)
        result = await lease.manager.write_stdin(
            execution_id=int(payload["executionId"]),
            chars=str(payload.get("chars", "")),
            yield_time_ms=int(payload["yieldTimeMs"]),
            max_output_tokens=int(payload["maxOutputTokens"]),
            owner_session_key=_required_string(payload, "ownerSessionKey"),
        )
        return _result_payload(result)

    async def stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease = await self._lease(payload)
        stopped = await lease.manager.terminate_execution(
            int(payload["executionId"]),
            owner_session_key=_required_string(payload, "ownerSessionKey"),
        )
        return {"stopped": stopped}

    async def terminate_owner(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease = await self._lease(payload)
        report = await lease.manager.terminate_owner(
            _required_string(payload, "ownerSessionKey")
        )
        return _cleanup_payload(report)

    async def shutdown_manager(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = self._identity(payload)
        async with self._lock:
            lease = self._managers.pop(key, None)
        if lease is None:
            return _cleanup_payload(ExecutionCleanupReport((), (), ()))
        return _cleanup_payload(await lease.manager.shutdown())

    async def active_executions(self, payload: dict[str, Any]) -> dict[str, Any]:
        lease = await self._lease(payload)
        return {"executionIds": await lease.manager.active_execution_ids()}

    def _rpc(
        self,
        handler: _RpcHandler,
    ) -> Callable[[Any, grpc.aio.ServicerContext], Awaitable[Any]]:
        async def run(message: Any, context: grpc.aio.ServicerContext) -> Any:
            try:
                document = decode_message(message)
                payload = await handler(document)
                return encode_message({"ok": True, **payload})
            except (KeyError, TypeError, ValueError) as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except PermissionError as exc:
                await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            except Exception as exc:
                await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            raise AssertionError("gRPC abort returned")

        return run

    async def _lease(self, payload: dict[str, Any]) -> _ManagerLease:
        key = self._identity(payload)
        async with self._lock:
            lease = self._managers.get(key)
            if lease is None:
                lease = _ManagerLease(ShellProcessManager(), time.monotonic())
                self._managers[key] = lease
            else:
                lease.last_seen = time.monotonic()
            return lease

    def _identity(self, payload: dict[str, Any]) -> tuple[str, str]:
        token = _required_string(payload, "token")
        if not hmac.compare_digest(token, self._token):
            raise PermissionError("Host Bridge token 无效")
        return (
            _required_string(payload, "bootId"),
            _required_string(payload, "managerId"),
        )


async def serve(socket_path: Path, token: str, lease_timeout_s: float) -> None:
    """Serve one private UDS and clean every leased process on shutdown."""

    # 1. 拒绝覆盖非 socket 路径，清理上次正常退出遗留的 socket。
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        if not socket_path.is_socket():
            raise RuntimeError(f"拒绝覆盖非 socket: {socket_path}")
        socket_path.unlink()

    # 2. 启动 RPC 与 lease watchdog，再发布仅 owner 可访问的 socket。
    service = HostBridgeService(token, lease_timeout_s)
    server = grpc.aio.server()
    server.add_generic_rpc_handlers((service.rpc_handlers(),))
    if server.add_insecure_port(f"unix:{socket_path}") != 1:
        raise RuntimeError(f"无法监听 Host Bridge socket: {socket_path}")
    await server.start()
    os.chmod(socket_path, 0o600)
    watchdog = asyncio.create_task(service.reap_expired(), name="host-bridge-reaper")
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopping.set)
    try:
        await stopping.wait()
    finally:
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)
        await service.shutdown()
        await server.stop(grace=2)
        socket_path.unlink(missing_ok=True)


def _required_string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Host Bridge {name} 必须是非空 string")
    return value


def _host_environment(requested: dict[str, str]) -> dict[str, str]:
    """Keep host identity and import only execution-scoped presentation fields."""

    env = os.environ.copy()
    for name in (
        "AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN",
        "NO_COLOR",
        "TERM",
        "COLORTERM",
        "PAGER",
        "GIT_PAGER",
        "GH_PAGER",
    ):
        if name in requested:
            env[name] = requested[name]
        else:
            env.pop(name, None)
    return env


def _result_payload(result: ExecutionResult) -> dict[str, Any]:
    return {
        "outputBase64": base64.b64encode(result.output).decode("ascii"),
        "wallTimeMs": result.wall_time_ms,
        "originalTokenCount": result.original_token_count,
        "outputOmittedBytes": result.output_omitted_bytes,
        "executionId": result.execution_id,
        "exitCode": result.exit_code,
        "outputPath": result.output_path,
        "finishReason": result.finish_reason,
    }


def _cleanup_payload(report: ExecutionCleanupReport) -> dict[str, Any]:
    return {
        "attempted": list(report.attempted_execution_ids),
        "cleaned": list(report.cleaned_execution_ids),
        "failures": [
            {
                "executionId": item.execution_id,
                "errorType": item.error_type,
                "message": item.message,
            }
            for item in report.failures
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Akashic Host Bridge")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--lease-timeout", type=float, default=10.0)
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    asyncio.run(serve(args.socket.resolve(), token, args.lease_timeout))


if __name__ == "__main__":
    main()

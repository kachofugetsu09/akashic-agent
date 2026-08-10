from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import os
import re
import shlex
import shutil
import signal
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable

import grpc

from agent.host_bridge.protocol import SERVICE_NAME
from agent.host_bridge.protocol import decode_message
from agent.host_bridge.protocol import deserialize_message
from agent.host_bridge.protocol import encode_message
from agent.host_bridge.protocol import serialize_message
from agent.tools.unified_exec import ExecutionCleanupReport
from agent.tools.unified_exec import ExecutionResult
from agent.tools.unified_exec import ShellProcessManager
from agent.tools.base import ToolResult
from agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)

_RpcHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass
class _ManagerLease:
    manager: ShellProcessManager
    last_seen: float
    cleanup_failure: ExecutionCleanupReport | None = None
    reaping: bool = False
    active_operations: int = 0
    operations_drained: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.operations_drained.set()


class HostBridgeService:
    """Own host shell managers and reap them when their Core lease expires."""

    def __init__(
        self,
        token: str,
        lease_timeout_s: float,
        artifact_root: Path,
        *,
        release_commit: str,
        toolchain_digest: str,
        runtime_checkout: Path,
        bridge_python: Path,
    ) -> None:
        if not token:
            raise ValueError("Host Bridge token 不能为空")
        if lease_timeout_s <= 0:
            raise ValueError("lease timeout 必须大于零")
        if _COMMIT_PATTERN.fullmatch(release_commit) is None:
            raise ValueError("Host Bridge release commit 必须是完整 SHA")
        if _SHA256_PATTERN.fullmatch(toolchain_digest) is None:
            raise ValueError("Host Bridge toolchain digest 必须是 SHA256")
        self._token = token
        self._lease_timeout_s = lease_timeout_s
        artifact_root.mkdir(parents=True, exist_ok=True)
        self._artifact_root = artifact_root
        self._release_commit = release_commit
        self._toolchain_digest = toolchain_digest
        self._runtime_cli = _materialize_runtime_cli(
            artifact_root,
            runtime_checkout.resolve(),
            bridge_python.absolute(),
            release_commit,
        )
        self._managers: dict[tuple[str, str], _ManagerLease] = {}
        self._lock = asyncio.Lock()
        self._claim_lock = asyncio.Lock()
        self._active_boot_id: str | None = None

    def rpc_handlers(self) -> grpc.GenericRpcHandler:
        methods: dict[str, _RpcHandler] = {
            "Inspect": self.inspect,
            "ClaimBoot": self.claim_boot,
            "Probe": self.probe,
            "Heartbeat": self.heartbeat,
            "Exec": self.exec_command,
            "WriteStdin": self.write_stdin,
            "Stop": self.stop,
            "TerminateOwner": self.terminate_owner,
            "ShutdownManager": self.shutdown_manager,
            "ActiveExecutions": self.active_executions,
            "FileTool": self.file_tool,
            "SkillRequirements": self.skill_requirements,
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
            async with self._claim_lock:
                async with self._lock:
                    expired = [
                        (key, lease)
                        for key, lease in self._managers.items()
                        if lease.last_seen < cutoff and not lease.reaping
                    ]
                    for _, lease in expired:
                        lease.reaping = True
                for key, lease in expired:
                    await lease.operations_drained.wait()
                    report = await lease.manager.shutdown()
                    async with self._lock:
                        current = self._managers.get(key)
                        if current is not lease:
                            continue
                        if report.failures:
                            lease.cleanup_failure = report
                            lease.last_seen = time.monotonic()
                            lease.reaping = False
                        else:
                            del self._managers[key]

    async def shutdown(self) -> None:
        async with self._claim_lock:
            async with self._lock:
                self._active_boot_id = None
                leases = list(self._managers.items())
                for _, lease in leases:
                    lease.reaping = True
            failures: list[ExecutionCleanupReport] = []
            for key, lease in leases:
                await lease.operations_drained.wait()
                report = await lease.manager.shutdown()
                if report.failures:
                    failures.append(report)
                    continue
                async with self._lock:
                    if self._managers.get(key) is lease:
                        del self._managers[key]
        if failures:
            raise RuntimeError("Host Bridge shutdown 未能确认清理全部 execution")

    async def inspect(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Report Bridge identity without acquiring Core boot ownership."""

        _ = self._identity(payload)
        return self._probe_payload()

    async def claim_boot(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Fence the previous Core generation before admitting one boot owner."""

        boot_id, _ = self._identity(payload)
        async with self._claim_lock:
            # 1. Close admission before waiting for in-flight manager operations.
            async with self._lock:
                previous_boot_id = self._active_boot_id
                if previous_boot_id == boot_id:
                    foreign = [key for key in self._managers if key[0] != boot_id]
                    if foreign:
                        raise RuntimeError(
                            "Host Bridge 当前 boot 下存在外代 manager，拒绝确认 ownership"
                        )
                    return {
                        "ownerBootId": boot_id,
                        "previousBootId": previous_boot_id,
                        "cleanedManagerCount": 0,
                        "cleanedExecutionCount": 0,
                    }
                self._active_boot_id = None
                leases = list(self._managers.items())
                for _, lease in leases:
                    lease.reaping = True

            # 2. Drain every old manager and prove its execution table is empty.
            reports: list[ExecutionCleanupReport] = []
            failed_keys: set[tuple[str, str]] = set()
            for key, lease in leases:
                await lease.operations_drained.wait()
                report = await lease.manager.shutdown()
                active_ids = await lease.manager.active_execution_ids()
                reports.append(report)
                if report.failures or active_ids:
                    failed_keys.add(key)
                    async with self._lock:
                        lease.cleanup_failure = report

            # 3. Publish the new owner only after the old generation is an empty set.
            async with self._lock:
                for key, lease in leases:
                    if key not in failed_keys and self._managers.get(key) is lease:
                        del self._managers[key]
                if failed_keys:
                    raise RuntimeError(
                        "Host Bridge 旧 boot cleanup 未确认，拒绝新 boot ownership"
                    )
                if self._managers:
                    raise RuntimeError(
                        "Host Bridge manager 集合在 boot claim 期间发生变化"
                    )
                self._active_boot_id = boot_id
            return {
                "ownerBootId": boot_id,
                "previousBootId": previous_boot_id,
                "cleanedManagerCount": len(leases),
                "cleanedExecutionCount": sum(
                    len(report.cleaned_execution_ids) for report in reports
                ),
            }

    async def probe(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ = await self._lease(payload)
        return self._probe_payload()

    def _probe_payload(self) -> dict[str, Any]:
        return {
            "releaseCommit": self._release_commit,
            "toolchainDigest": self._toolchain_digest,
            "capabilities": [
                "boot-fencing",
                "exec",
                "pty",
                "stdin",
                "stop",
                "lease",
                "file-tools",
                "raw-bytes",
                "skill-requirements",
            ],
        }

    async def heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ = await self._lease(payload)
        return {"alive": True}

    async def exec_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        argv = payload.get("argv")
        requested_env = payload.get("env")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item for item in argv)
        ):
            raise ValueError("Host Bridge argv 必须是非空 string array")
        if not isinstance(requested_env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in requested_env.items()
        ):
            raise ValueError("Host Bridge env 必须是 string map")
        cwd_text = payload.get("cwd")
        async with self._manager_operation(payload) as manager:
            result = await manager.exec_command(
                command=_required_string(payload, "command"),
                argv=argv,
                cwd=None if cwd_text is None else Path(str(cwd_text)),
                env=_host_environment(
                    requested_env,
                    _required_string(payload, "bootId"),
                    self._runtime_cli,
                ),
                tty=bool(payload["tty"]),
                yield_time_ms=int(payload["yieldTimeMs"]),
                max_output_tokens=int(payload["maxOutputTokens"]),
                hard_timeout_s=int(payload["hardTimeoutS"]),
                owner_session_key=_required_string(payload, "ownerSessionKey"),
            )
        return _result_payload(result)

    async def write_stdin(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._manager_operation(payload) as manager:
            result = await manager.write_stdin(
                execution_id=int(payload["executionId"]),
                chars=str(payload.get("chars", "")),
                yield_time_ms=int(payload["yieldTimeMs"]),
                max_output_tokens=int(payload["maxOutputTokens"]),
                owner_session_key=_required_string(payload, "ownerSessionKey"),
            )
        return _result_payload(result)

    async def stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._manager_operation(payload) as manager:
            stopped = await manager.terminate_execution(
                int(payload["executionId"]),
                owner_session_key=_required_string(payload, "ownerSessionKey"),
            )
        return {"stopped": stopped}

    async def terminate_owner(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._manager_operation(payload) as manager:
            report = await manager.terminate_owner(
                _required_string(payload, "ownerSessionKey")
            )
        return _cleanup_payload(report)

    async def shutdown_manager(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = self._identity(payload)
        async with self._claim_lock:
            async with self._lock:
                self._assert_active_boot(key[0])
                lease = self._managers.get(key)
                if lease is None:
                    return _cleanup_payload(ExecutionCleanupReport((), (), ()))
                lease.reaping = True
            await lease.operations_drained.wait()
            report = await lease.manager.shutdown()
            async with self._lock:
                current = self._managers.get(key)
                if current is lease:
                    if report.failures:
                        lease.cleanup_failure = report
                        lease.last_seen = time.monotonic()
                    else:
                        del self._managers[key]
        return _cleanup_payload(report)

    async def active_executions(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._manager_operation(payload) as manager:
            return {"executionIds": await manager.active_execution_ids()}

    async def file_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute one existing filesystem tool against the host namespace."""

        # 1. Authenticate the Core generation and validate the operation envelope.
        operation = _required_string(payload, "operation")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("Host Bridge FileTool arguments 必须是 object")
        allowed_text = payload.get("allowedDir")
        allowed_dir = None if allowed_text is None else Path(str(allowed_text))

        # 2. Reuse the canonical tool implementation without recursively bridging.
        tool_types = {
            "read_file": ReadFileTool,
            "list_dir": ListDirTool,
            "write_file": WriteFileTool,
            "edit_file": EditFileTool,
        }
        tool_type = tool_types.get(operation)
        if tool_type is None:
            raise ValueError(f"Host Bridge 不支持文件操作: {operation}")
        tool = tool_type(allowed_dir=allowed_dir, enable_bridge=False)
        async with self._manager_operation(payload):
            result = await tool.execute(**arguments)

        # 3. Preserve multimodal raw-byte results across the RPC boundary.
        if isinstance(result, ToolResult):
            return {
                "resultType": "toolResult",
                "text": result.text,
                "contentBlocks": result.content_blocks,
                "mobileAttention": result.mobile_attention,
            }
        return {"resultType": "text", "text": result}

    async def skill_requirements(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Report host capability names without exposing environment values."""

        # 1. Authenticate the Core generation and validate the narrow request.
        key = self._identity(payload)
        async with self._lock:
            self._assert_active_boot(key[0])
        bins = _required_string_array(payload, "bins")
        env = _required_string_array(payload, "env")

        # 2. Partition names using the Bridge process capability environment.
        available_bins = [name for name in bins if shutil.which(name) is not None]
        available_env = [name for name in env if bool(os.environ.get(name))]
        return {
            "available": {
                "bins": available_bins,
                "env": available_env,
            },
            "missing": {
                "bins": [name for name in bins if name not in available_bins],
                "env": [name for name in env if name not in available_env],
            },
        }

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
            self._assert_active_boot(key[0])
            lease = self._managers.get(key)
            if lease is None:
                manager_root = self._artifact_root / key[0] / key[1]
                lease = _ManagerLease(
                    ShellProcessManager(output_dir=manager_root),
                    time.monotonic(),
                )
                self._managers[key] = lease
            else:
                if lease.cleanup_failure is not None:
                    raise RuntimeError("Host Bridge manager cleanup 未确认，拒绝复用")
                if lease.reaping:
                    raise RuntimeError("Host Bridge manager lease 正在回收，拒绝复用")
                lease.last_seen = time.monotonic()
            return lease

    @asynccontextmanager
    async def _manager_operation(
        self,
        payload: dict[str, Any],
    ) -> AsyncGenerator[ShellProcessManager]:
        """Admit one concurrent operation while fencing takeover and lease reaping."""

        lease = await self._lease(payload)
        key = self._identity(payload)
        async with self._lock:
            self._assert_active_boot(key[0])
            if self._managers.get(key) is not lease or lease.reaping:
                raise RuntimeError("Host Bridge manager admission 已关闭，拒绝执行")
            lease.active_operations += 1
            lease.operations_drained.clear()
        try:
            yield lease.manager
        finally:
            async with self._lock:
                lease.active_operations -= 1
                if lease.active_operations == 0:
                    lease.operations_drained.set()

    def _assert_active_boot(self, boot_id: str) -> None:
        if self._active_boot_id != boot_id:
            raise PermissionError(
                "Host Bridge boot 未持有 ownership: "
                f"requested={boot_id} active={self._active_boot_id or 'none'}"
            )

    def _identity(self, payload: dict[str, Any]) -> tuple[str, str]:
        token = _required_string(payload, "token")
        if not hmac.compare_digest(token, self._token):
            raise PermissionError("Host Bridge token 无效")
        if _required_string(payload, "expectedReleaseCommit") != self._release_commit:
            raise PermissionError("Host Bridge release commit 与客户端不一致")
        if (
            _required_string(payload, "expectedToolchainDigest")
            != self._toolchain_digest
        ):
            raise PermissionError("Host Bridge toolchain digest 与客户端不一致")
        return (
            _required_string(payload, "bootId"),
            _required_string(payload, "managerId"),
        )


async def serve(
    socket_path: Path,
    token: str,
    lease_timeout_s: float,
    artifact_root: Path,
    release_commit: str,
    toolchain_digest: str,
    runtime_checkout: Path,
    bridge_python: Path,
) -> None:
    """Serve one private UDS and clean every leased process on shutdown."""

    # 1. 拒绝覆盖非 socket 路径，清理上次正常退出遗留的 socket。
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        if not socket_path.is_socket():
            raise RuntimeError(f"拒绝覆盖非 socket: {socket_path}")
        socket_path.unlink()

    # 2. 启动 RPC 与 lease watchdog，再发布仅 owner 可访问的 socket。
    service = HostBridgeService(
        token,
        lease_timeout_s,
        artifact_root,
        release_commit=release_commit,
        toolchain_digest=toolchain_digest,
        runtime_checkout=runtime_checkout,
        bridge_python=bridge_python,
    )
    server = grpc.aio.server(
        options=(
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
        )
    )
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


def _required_string_array(payload: dict[str, Any], name: str) -> list[str]:
    value = payload.get(name)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"Host Bridge {name} 必须是非空 string 组成的 array")
    return [str(item) for item in value]


def _host_environment(
    requested: dict[str, str], boot_id: str, runtime_cli: Path
) -> dict[str, str]:
    """Keep host identity and import only execution-scoped presentation fields."""

    env = os.environ.copy()
    env["AKASHIC_BOOT_ID"] = boot_id
    for name in (
        "AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN",
        "AKASHIC_PLUGIN_ROLLOUT_CAPABILITY",
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
    env["AKASHIC_RUNTIME_CLI"] = str(runtime_cli)
    env["PATH"] = f"{runtime_cli.parent}:{env.get('PATH', '')}"
    return env


def _materialize_runtime_cli(
    artifact_root: Path,
    runtime_checkout: Path,
    bridge_python: Path,
    release_commit: str,
) -> Path:
    """Write one launcher whose interpreter and checkout cannot be changed by child env."""

    # 1. Refuse a deployment identity that cannot execute the selected release.
    main_path = runtime_checkout / "main.py"
    if not runtime_checkout.is_absolute() or not main_path.is_file():
        raise RuntimeError(f"Host Bridge runtime checkout 无效: {runtime_checkout}")
    if not bridge_python.is_absolute() or not bridge_python.is_file():
        raise RuntimeError(f"Host Bridge Python 无效: {bridge_python}")

    # 2. Publish a literal launcher under the Bridge-owned artifact root.
    launcher_dir = artifact_root / "runtime-cli" / release_commit
    launcher_dir.mkdir(parents=True, exist_ok=True)
    launcher = launcher_dir / "akashic-runtime"
    content = "\n".join(
        (
            "#!/bin/sh",
            "set -eu",
            f"exec env PYTHONPATH={shlex.quote(str(runtime_checkout))} \\",
            f'    {shlex.quote(str(bridge_python))} {shlex.quote(str(main_path))} "$@"',
            "",
        )
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=launcher_dir,
        prefix=f".{launcher.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o500)
    temporary.replace(launcher)
    return launcher


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
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--toolchain-digest", required=True)
    parser.add_argument("--runtime-checkout", type=Path, required=True)
    parser.add_argument("--bridge-python", type=Path, required=True)
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    asyncio.run(
        serve(
            args.socket.resolve(),
            token,
            args.lease_timeout,
            args.artifact_root.resolve(),
            args.release_commit,
            args.toolchain_digest,
            args.runtime_checkout.resolve(),
            args.bridge_python.absolute(),
        )
    )


if __name__ == "__main__":
    main()

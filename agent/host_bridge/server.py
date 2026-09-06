from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import logging
import os
import re
import shlex
import shutil
import signal
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, cast

from google.protobuf.message import Message

import grpc

from agent.host_bridge import host_bridge_pb2 as pb
from agent.host_bridge import host_bridge_pb2_grpc as rpc
from agent.host_bridge.protocol import (
    CHANNEL_OPTIONS,
    encode_execution,
    encode_cleanup,
    encode_file_result,
    require_fields,
    require_text,
    require_positive,
    require_nonnegative,
    require_names,
)
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
from core.common.diagnostic_log import configure_logging
from core.common.diagnostic_log import diagnostic_context
from core.common.diagnostic_log import log_event

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
logger = logging.getLogger(__name__)


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


def _rpc[Request: Message, Reply: Message](
    handler: Callable[["HostBridgeService", Request], Awaitable[Reply]],
) -> Callable[
    ["HostBridgeService", Request, grpc.aio.ServicerContext], Awaitable[Reply]
]:
    """在唯一 RPC 边界校验身份并记录诊断，取消始终向上传播。"""

    @wraps(handler)
    async def run(
        self: "HostBridgeService", request: Request, context: grpc.aio.ServicerContext
    ) -> Reply:
        started = time.perf_counter()
        method = handler.__name__
        identity = cast(pb.RequestContext, cast(Any, request).context)
        try:
            # 1. 先校验认证与结构；后续 manager 只检查实时 lease 状态。
            self._authenticate(identity, context)
            with diagnostic_context(
                session=identity.session_ref or None,
                turn=identity.turn_id or None,
                request_id=identity.request_id,
            ):
                if method != "Heartbeat":
                    log_event(
                        logger,
                        logging.INFO,
                        "host_bridge.rpc_started",
                        method=method,
                        boot_id=identity.boot_id,
                        manager_id=identity.manager_id,
                    )
                reply = await handler(self, request)
                if method != "Heartbeat":
                    log_event(
                        logger,
                        logging.INFO,
                        "host_bridge.rpc_completed",
                        method=method,
                        boot_id=identity.boot_id,
                        manager_id=identity.manager_id,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        outcome="completed",
                    )
                return reply
        except asyncio.CancelledError:
            # 2. 取消只结束本次 RPC 等待，不承诺进程未执行或输入未写入。
            raise
        except (KeyError, TypeError, ValueError) as exc:
            self._log_rpc_failure(
                method,
                identity.request_id,
                identity.boot_id,
                identity.manager_id,
                started,
                exc,
                "invalid_argument",
            )
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except PermissionError as exc:
            self._log_rpc_failure(
                method,
                identity.request_id,
                identity.boot_id,
                identity.manager_id,
                started,
                exc,
                "permission_denied",
            )
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
        except Exception as exc:
            self._log_rpc_failure(
                method,
                identity.request_id,
                identity.boot_id,
                identity.manager_id,
                started,
                exc,
                "internal",
            )
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
        raise AssertionError("gRPC abort returned")

    return run


class HostBridgeService(rpc.HostBridgeServicer):
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
                    log_event(
                        logger,
                        logging.ERROR if report.failures else logging.WARNING,
                        "host_bridge.lease_reaped",
                        boot_id=key[0],
                        manager_id=key[1],
                        outcome="failed" if report.failures else "completed",
                        counts=f"executions:{len(report.attempted_execution_ids)},failures:{len(report.failures)}",
                    )

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

    @_rpc
    async def Inspect(self, request: pb.ContextRequest) -> pb.IdentityReply:
        """报告身份，不获取或续期 boot ownership。"""
        return self._probe_payload()

    @_rpc
    async def ClaimBoot(self, request: pb.ContextRequest) -> pb.ClaimBootReply:
        """Fence the previous Core generation before admitting one boot owner."""

        boot_id = request.context.boot_id
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
                    return pb.ClaimBootReply(
                        owner_boot_id=boot_id,
                        previous_boot_id=previous_boot_id,
                        cleaned_manager_count=0,
                        cleaned_execution_count=0,
                    )
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
            log_event(
                logger,
                logging.INFO,
                "host_bridge.boot_claimed",
                boot_id=boot_id,
                outcome="completed",
                counts=f"managers:{len(leases)},executions:{sum(len(report.cleaned_execution_ids) for report in reports)}",
            )
            return pb.ClaimBootReply(
                owner_boot_id=boot_id,
                previous_boot_id=previous_boot_id,
                cleaned_manager_count=len(leases),
                cleaned_execution_count=sum(
                    len(report.cleaned_execution_ids) for report in reports
                ),
            )

    @_rpc
    async def Probe(self, request: pb.ContextRequest) -> pb.IdentityReply:
        _ = await self._lease(request.context)
        return self._probe_payload()

    def _probe_payload(self) -> pb.IdentityReply:
        return pb.IdentityReply(
            release_commit=self._release_commit,
            toolchain_digest=self._toolchain_digest,
            capabilities=[
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
        )

    @_rpc
    async def Heartbeat(self, request: pb.ContextRequest) -> pb.HeartbeatReply:
        _ = await self._lease(request.context)
        return pb.HeartbeatReply(alive=True)

    @_rpc
    async def Exec(self, request: pb.ExecRequest) -> pb.ExecutionReply:
        """验证公开参数后交给唯一 manager，直接返回原始输出字节。"""
        # 1. protobuf 保证字段类型，这里检查 presence 和公开值域。
        require_fields(
            request, "tty", "yield_time_ms", "max_output_tokens", "hard_timeout_s"
        )
        require_text(request.command, "command")
        require_text(request.owner_session_key, "owner_session_key")
        if not request.argv:
            raise ValueError("Host Bridge argv 必须非空")
        require_names(request.argv, "argv")
        require_nonnegative(request.max_output_tokens, "max_output_tokens")
        require_positive(request.hard_timeout_s, "hard_timeout_s")
        # 2. 进程与输出消费仍由原 manager 拥有。
        async with self._manager_operation(request.context) as manager:
            result = await manager.exec_command(
                command=request.command,
                argv=list(request.argv),
                cwd=Path(request.cwd) if request.HasField("cwd") else None,
                env=_host_environment(
                    dict(request.env), request.context.boot_id, self._runtime_cli
                ),
                tty=request.tty,
                yield_time_ms=request.yield_time_ms,
                max_output_tokens=request.max_output_tokens,
                hard_timeout_s=request.hard_timeout_s,
                owner_session_key=request.owner_session_key,
            )
        log_event(
            logger,
            logging.INFO,
            "host_bridge.execution_yielded",
            command_fp=hashlib.sha256(request.command.encode("utf-8")).hexdigest()[:16],
            command_bytes=len(request.command.encode("utf-8")),
            cwd=request.cwd,
            tty=request.tty,
            execution_id=result.execution_id,
            exit_code=result.exit_code,
            finish_reason=result.finish_reason,
            output_bytes=len(result.output),
            duration_ms=result.wall_time_ms,
            outcome="completed" if result.exit_code is not None else "running",
        )
        return encode_execution(result)

    @_rpc
    async def WriteStdin(self, request: pb.WriteStdinRequest) -> pb.ExecutionReply:
        require_fields(
            request, "execution_id", "chars", "yield_time_ms", "max_output_tokens"
        )
        require_positive(request.execution_id, "execution_id")
        require_text(request.owner_session_key, "owner_session_key")
        require_nonnegative(request.max_output_tokens, "max_output_tokens")
        async with self._manager_operation(request.context) as manager:
            result = await manager.write_stdin(
                execution_id=request.execution_id,
                chars=request.chars,
                yield_time_ms=request.yield_time_ms,
                max_output_tokens=request.max_output_tokens,
                owner_session_key=request.owner_session_key,
            )
        return encode_execution(result)

    @_rpc
    async def Stop(self, request: pb.StopRequest) -> pb.StopReply:
        require_fields(request, "execution_id")
        require_positive(request.execution_id, "execution_id")
        require_text(request.owner_session_key, "owner_session_key")
        async with self._manager_operation(request.context) as manager:
            stopped = await manager.terminate_execution(
                request.execution_id, owner_session_key=request.owner_session_key
            )
        return pb.StopReply(stopped=stopped)

    @_rpc
    async def TerminateOwner(self, request: pb.OwnerRequest) -> pb.CleanupReply:
        require_text(request.owner_session_key, "owner_session_key")
        async with self._manager_operation(request.context) as manager:
            report = await manager.terminate_owner(request.owner_session_key)
        return encode_cleanup(report)

    @_rpc
    async def ShutdownManager(self, request: pb.ContextRequest) -> pb.CleanupReply:
        """排空旧 manager 的操作并确认回收，失败时保留清理事实。"""
        key = (request.context.boot_id, request.context.manager_id)
        async with self._claim_lock:
            async with self._lock:
                self._assert_active_boot(key[0])
                lease = self._managers.get(key)
                if lease is None:
                    return encode_cleanup(ExecutionCleanupReport((), (), ()))
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
        return encode_cleanup(report)

    @_rpc
    async def ActiveExecutions(
        self, request: pb.ContextRequest
    ) -> pb.ActiveExecutionsReply:
        async with self._manager_operation(request.context) as manager:
            return pb.ActiveExecutionsReply(
                execution_ids=await manager.active_execution_ids()
            )

    @_rpc
    async def FileTool(self, request: pb.FileRequest) -> pb.FileReply:
        """校验文件操作后复用已有文件工具，不建立第二套文件实现。"""
        allowed_dir = (
            Path(request.allowed_dir) if request.HasField("allowed_dir") else None
        )
        # 1. 每支 oneof 只拥有对应工具的公开参数。
        match request.WhichOneof("operation"):
            case "read":
                read = request.read
                require_fields(read, "path")
                if read.HasField("offset"):
                    require_nonnegative(read.offset, "offset")
                if read.HasField("limit"):
                    require_positive(read.limit, "limit")
                async with self._manager_operation(request.context):
                    result = ReadFileTool(
                        allowed_dir=allowed_dir, enable_bridge=False
                    ).read_from_disk(
                        read.path,
                        offset=read.offset,
                        limit=read.limit if read.HasField("limit") else None,
                    )
            case "write":
                require_fields(request.write, "path", "content")
                async with self._manager_operation(request.context):
                    result = await WriteFileTool(
                        allowed_dir=allowed_dir, enable_bridge=False
                    ).execute(request.write.path, request.write.content)
            case "edit":
                require_fields(request.edit, "path", "old_text", "new_text")
                async with self._manager_operation(request.context):
                    result = await EditFileTool(
                        allowed_dir=allowed_dir, enable_bridge=False
                    ).execute(
                        request.edit.path,
                        request.edit.old_text,
                        request.edit.new_text,
                        replace_all=request.edit.replace_all,
                    )
            case "list":
                require_fields(request.list, "path")
                async with self._manager_operation(request.context):
                    result = await ListDirTool(
                        allowed_dir=allowed_dir, enable_bridge=False
                    ).execute(request.list.path)
            case _:
                raise ValueError("Host Bridge 文件操作缺失")
        # 2. 模型投影留在 Core；Bridge 只转换已有文件结果。
        return encode_file_result(result)

    @_rpc
    async def SkillRequirements(
        self, request: pb.SkillRequirementsRequest
    ) -> pb.SkillRequirementsReply:
        """只报告宿主能力名称，不返回环境变量值。"""
        async with self._lock:
            self._assert_active_boot(request.context.boot_id)
        require_names(request.bins, "bins")
        require_names(request.env, "env")
        available_bins = [
            name for name in request.bins if shutil.which(name) is not None
        ]
        available_env = [name for name in request.env if bool(os.environ.get(name))]
        return pb.SkillRequirementsReply(
            available=pb.RequirementNames(bins=available_bins, env=available_env),
            missing=pb.RequirementNames(
                bins=[name for name in request.bins if name not in available_bins],
                env=[name for name in request.env if name not in available_env],
            ),
        )

    def _log_rpc_failure(
        self,
        method: str,
        request_id: str,
        boot_id: str,
        manager_id: str,
        started: float,
        error: BaseException,
        reason: str,
    ) -> None:
        """Record a classified RPC failure without request payloads or credentials."""

        with diagnostic_context(request_id=request_id):
            log_event(
                logger,
                logging.ERROR,
                "host_bridge.rpc_failed",
                method=method,
                boot_id=boot_id,
                manager_id=manager_id,
                duration_ms=int((time.perf_counter() - started) * 1000),
                outcome="failed",
                reason=reason,
                error_type=type(error).__name__,
                error_fp=hashlib.sha256(str(error).encode("utf-8")).hexdigest()[:16],
                exc_info=True,
            )

    async def _lease(self, context: pb.RequestContext) -> _ManagerLease:
        key = (context.boot_id, context.manager_id)
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
        context: pb.RequestContext,
    ) -> AsyncGenerator[ShellProcessManager]:
        """Admit one concurrent operation while fencing takeover and lease reaping."""

        lease = await self._lease(context)
        key = (context.boot_id, context.manager_id)
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

    def _authenticate(
        self, identity: pb.RequestContext, context: grpc.aio.ServicerContext
    ) -> None:
        """只在 RPC 入口认证 token 和固定身份，不获取 boot ownership。"""
        metadata = context.invocation_metadata()
        if metadata is None:
            raise PermissionError("Host Bridge token 缺失")
        tokens = [value for key, value in metadata if key == "authorization"]
        if (
            len(tokens) != 1
            or not isinstance(tokens[0], str)
            or not hmac.compare_digest(
                tokens[0].encode("utf-8"), f"Bearer {self._token}".encode("utf-8")
            )
        ):
            raise PermissionError("Host Bridge token 无效")
        for name, value in (
            ("boot_id", identity.boot_id),
            ("manager_id", identity.manager_id),
            ("request_id", identity.request_id),
            ("expected_release_commit", identity.expected_release_commit),
            ("expected_toolchain_digest", identity.expected_toolchain_digest),
        ):
            require_text(value, name)
        if identity.HasField("session_ref"):
            require_text(identity.session_ref, "session_ref")
        if identity.HasField("turn_id"):
            require_text(identity.turn_id, "turn_id")
        if identity.expected_release_commit != self._release_commit:
            raise PermissionError("Host Bridge release commit 与客户端不一致")
        if identity.expected_toolchain_digest != self._toolchain_digest:
            raise PermissionError("Host Bridge toolchain digest 与客户端不一致")


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
    server = grpc.aio.server(options=CHANNEL_OPTIONS)
    rpc.add_HostBridgeServicer_to_server(service, server)
    if server.add_insecure_port(f"unix:{socket_path}") != 1:
        raise RuntimeError(f"无法监听 Host Bridge socket: {socket_path}")
    await server.start()
    os.chmod(socket_path, 0o600)
    log_event(
        logger,
        logging.INFO,
        "host_bridge.started",
        release_commit=release_commit,
        toolchain_digest=toolchain_digest,
        outcome="ready",
    )
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
        log_event(logger, logging.INFO, "host_bridge.stopped", outcome="completed")


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


def main() -> None:
    configure_logging()
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

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
from google.protobuf.message import Message

from agent.host_bridge import host_bridge_pb2 as pb
from agent.host_bridge import host_bridge_pb2_grpc as rpc
from agent.host_bridge.protocol import (
    CHANNEL_OPTIONS,
    decode_cleanup,
    decode_execution,
    decode_file_result,
    require_fields,
    require_names,
    require_positive,
    require_text,
)
from agent.tools.base import ToolResult
from agent.tools.unified_exec import ExecutionCleanupReport, ExecutionResult
from core.common.diagnostic_log import current_diagnostic_context

_HEARTBEAT_INTERVAL_S = 2.0


@dataclass(frozen=True)
class SkillRequirementAvailability:
    available_bins: tuple[str, ...]
    missing_bins: tuple[str, ...]
    available_env: tuple[str, ...]
    missing_env: tuple[str, ...]


class HostBridgeSkillCapabilityChecker:
    """通过短生命周期同步 RPC 查询宿主能力名称。"""

    def __init__(
        self,
        socket_path: Path,
        boot_id: str,
        token: str,
        expected_release_commit: str,
        expected_toolchain_digest: str,
    ) -> None:
        _check_client_identity(socket_path, boot_id, token)
        self._socket_path = socket_path
        self._token = token
        self._context = pb.RequestContext(
            boot_id=boot_id,
            manager_id=uuid.uuid4().hex,
            expected_release_commit=expected_release_commit,
            expected_toolchain_digest=expected_toolchain_digest,
        )

    def check_skill_requirements(
        self, bins: list[str], env: list[str]
    ) -> SkillRequirementAvailability:
        """保留同步调用的关闭 owner，并验证名称集合没有缺失或混入值。"""
        request = pb.SkillRequirementsRequest(context=self._context, bins=bins, env=env)
        request.context.request_id = uuid.uuid4().hex
        with grpc.insecure_channel(
            f"unix:{self._socket_path}", options=CHANNEL_OPTIONS
        ) as channel:
            stub = rpc.HostBridgeStub(channel)
            try:
                response: pb.SkillRequirementsReply = stub.SkillRequirements(
                    request,
                    timeout=5,
                    metadata=(("authorization", f"Bearer {self._token}"),),
                )
            except grpc.RpcError as exc:
                raise RuntimeError(
                    f"Host Bridge SkillRequirements 失败: {exc.code().name}: {exc.details()}"
                ) from exc
        require_fields(response, "available", "missing")
        for names in (response.available, response.missing):
            require_names(names.bins, "bins")
            require_names(names.env, "env")
        _validate_requirement_partition(
            bins, list(response.available.bins), list(response.missing.bins), "bins"
        )
        _validate_requirement_partition(
            env, list(response.available.env), list(response.missing.env), "env"
        )
        return SkillRequirementAvailability(
            tuple(response.available.bins),
            tuple(response.missing.bins),
            tuple(response.available.env),
            tuple(response.missing.env),
        )


class HostBridgeShellProcessManager:
    """在 gRPC UDS 上保留 ShellProcessManager 的可观察语义。"""

    def __init__(
        self,
        socket_path: Path,
        boot_id: str,
        token: str,
        expected_release_commit: str,
        expected_toolchain_digest: str,
    ) -> None:
        _check_client_identity(socket_path, boot_id, token)
        self._socket_path = socket_path
        self._boot_id = boot_id
        self._token = token
        self._manager_id = uuid.uuid4().hex
        self._expected_release_commit = expected_release_commit
        self._expected_toolchain_digest = expected_toolchain_digest
        self._channel = grpc.aio.insecure_channel(
            f"unix:{socket_path}", options=CHANNEL_OPTIONS
        )
        self._stub = rpc.HostBridgeStub(self._channel)
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._lease_error: Exception | None = None
        self._unconfirmed_owners: dict[str, str] = {}
        self._closed = False

    def _request_context(self) -> pb.RequestContext:
        correlation = current_diagnostic_context()
        return pb.RequestContext(
            boot_id=self._boot_id,
            manager_id=self._manager_id,
            request_id=uuid.uuid4().hex,
            expected_release_commit=self._expected_release_commit,
            expected_toolchain_digest=self._expected_toolchain_digest,
            session_ref=correlation["session"] or None,
            turn_id=correlation["turn"] or None,
        )

    async def probe(self) -> dict[str, Any]:
        reply: pb.IdentityReply = await self._call(
            self._stub.Probe,
            pb.ContextRequest(context=self._request_context()),
            method="Probe",
            timeout=5,
        )
        return self._identity_reply(reply)

    async def inspect(self) -> dict[str, Any]:
        reply: pb.IdentityReply = await self._call(
            self._stub.Inspect,
            pb.ContextRequest(context=self._request_context()),
            method="Inspect",
            lease=False,
        )
        return self._identity_reply(reply)

    def _identity_reply(self, reply: pb.IdentityReply) -> dict[str, Any]:
        require_text(reply.release_commit, "release_commit")
        require_text(reply.toolchain_digest, "toolchain_digest")
        if not reply.capabilities:
            raise ValueError("Host Bridge capabilities 为空")
        require_names(reply.capabilities, "capabilities")
        if reply.release_commit != self._expected_release_commit:
            raise RuntimeError("Host Bridge release commit 与 Core 不一致")
        if reply.toolchain_digest != self._expected_toolchain_digest:
            raise RuntimeError("Host Bridge toolchain digest 与部署合同不一致")
        return {
            "releaseCommit": reply.release_commit,
            "toolchainDigest": reply.toolchain_digest,
            "capabilities": list(reply.capabilities),
        }

    async def claim_boot(self) -> dict[str, Any]:
        reply: pb.ClaimBootReply = await self._call(
            self._stub.ClaimBoot,
            pb.ContextRequest(context=self._request_context()),
            method="ClaimBoot",
            lease=False,
        )
        require_fields(reply, "cleaned_manager_count", "cleaned_execution_count")
        if reply.owner_boot_id != self._boot_id:
            raise RuntimeError("Host Bridge 未确认请求 Core boot 的 ownership")
        if reply.HasField("previous_boot_id"):
            require_text(reply.previous_boot_id, "previous_boot_id")
        return {
            "ownerBootId": reply.owner_boot_id,
            "previousBootId": (
                reply.previous_boot_id if reply.HasField("previous_boot_id") else None
            ),
            "cleanedManagerCount": reply.cleaned_manager_count,
            "cleanedExecutionCount": reply.cleaned_execution_count,
        }

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
        if owner_session_key in self._unconfirmed_owners:
            raise RuntimeError(
                f"Host Bridge owner={owner_session_key} 的 cleanup 未确认，拒绝创建新 execution: "
                f"{self._unconfirmed_owners[owner_session_key]}"
            )
        request = pb.ExecRequest(
            context=self._request_context(),
            command=command,
            argv=argv,
            cwd=None if cwd is None else str(cwd),
            env=env,
            tty=tty,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
            hard_timeout_s=hard_timeout_s,
            owner_session_key=owner_session_key,
        )
        return decode_execution(
            await self._call(self._stub.Exec, request, method="Exec")
        )

    async def write_stdin(
        self,
        *,
        execution_id: int,
        chars: str,
        yield_time_ms: int,
        max_output_tokens: int,
        owner_session_key: str,
    ) -> ExecutionResult:
        request = pb.WriteStdinRequest(
            context=self._request_context(),
            execution_id=execution_id,
            chars=chars,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
            owner_session_key=owner_session_key,
        )
        return decode_execution(
            await self._call(self._stub.WriteStdin, request, method="WriteStdin")
        )

    async def terminate_execution(
        self, execution_id: int, *, owner_session_key: str
    ) -> bool:
        request = pb.StopRequest(
            context=self._request_context(),
            execution_id=execution_id,
            owner_session_key=owner_session_key,
        )
        reply: pb.StopReply = await self._call(self._stub.Stop, request, method="Stop")
        require_fields(reply, "stopped")
        return reply.stopped

    async def terminate_owner(
        self,
        owner_session_key: str,
    ) -> ExecutionCleanupReport:
        """缺少远端清理确认时阻止同 owner 新命令，成功重试才解除。"""
        try:
            request = pb.OwnerRequest(
                context=self._request_context(), owner_session_key=owner_session_key
            )
            report = decode_cleanup(
                await self._call(
                    self._stub.TerminateOwner, request, method="TerminateOwner"
                )
            )
        except (Exception, asyncio.CancelledError) as error:
            # RPC、损坏响应与取消都不能证明远端进程已清理；明确 report 的隔离归服务端。
            self._unconfirmed_owners[owner_session_key] = f"{type(error).__name__}: {error}"
            raise
        if report.failures:
            self._unconfirmed_owners[owner_session_key] = "; ".join(
                f"execution={failure.execution_id}: {failure.message}"
                for failure in report.failures
            )
        else:
            _ = self._unconfirmed_owners.pop(owner_session_key, None)
        return report

    async def shutdown(self) -> ExecutionCleanupReport:
        if self._closed:
            return ExecutionCleanupReport((), (), ())
        await self._stop_heartbeat()
        reply: pb.CleanupReply = await self._call(
            self._stub.ShutdownManager,
            pb.ContextRequest(context=self._request_context()),
            method="ShutdownManager",
            lease=False,
        )
        report = decode_cleanup(reply)
        if not report.failures:
            await self.close_transport()
        return report

    async def close_transport(self) -> None:
        """只关闭客户端；已登记进程仍由宿主 lease owner 回收。"""
        self._closed = True
        await self._stop_heartbeat()
        await self._channel.close()

    async def _stop_heartbeat(self) -> None:
        if self._heartbeat_task is not None:
            _ = self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

    async def active_execution_ids(self) -> list[int]:
        reply: pb.ActiveExecutionsReply = await self._call(
            self._stub.ActiveExecutions,
            pb.ContextRequest(context=self._request_context()),
            method="ActiveExecutions",
        )
        for execution_id in reply.execution_ids:
            require_positive(execution_id, "execution_id")
        return list(reply.execution_ids)

    async def execute_file_tool(
        self, operation: str, *, allowed_dir: Path | None, arguments: dict[str, Any]
    ) -> str | ToolResult:
        """把已有四种文件工具参数转换为明确的 oneof。"""
        request = pb.FileRequest(
            context=self._request_context(),
            allowed_dir=None if allowed_dir is None else str(allowed_dir),
        )
        match operation:
            case "read_file":
                request.read.CopyFrom(pb.ReadFile(**arguments))
            case "write_file":
                request.write.CopyFrom(pb.WriteFile(**arguments))
            case "edit_file":
                request.edit.CopyFrom(pb.EditFile(**arguments))
            case "list_dir":
                request.list.CopyFrom(pb.ListDir(**arguments))
            case _:
                raise ValueError(f"Host Bridge 不支持文件操作: {operation}")
        return decode_file_result(
            await self._call(self._stub.FileTool, request, method="FileTool")
        )

    async def _call(
        self,
        call: Any,
        request: Message,
        *,
        method: str,
        lease: bool = True,
        timeout: float | None = None,
    ) -> Any:
        """发起一次 RPC；失败或取消均不重放可能已生效的操作。"""
        if self._closed:
            raise RuntimeError("Host Bridge manager 已关闭")
        if method not in {"Heartbeat", "ShutdownManager"} and self._lease_error is not None:
            raise RuntimeError(f"Host Bridge lease 已失效: {self._lease_error}")
        if lease:
            self._ensure_heartbeat()
        try:
            return await call(
                request,
                timeout=timeout,
                metadata=(("authorization", f"Bearer {self._token}"),),
            )
        except grpc.aio.AioRpcError as exc:
            uncertainty = (
                "；操作可能已生效，不得自动重发"
                if method in {"Exec", "WriteStdin", "FileTool"}
                else ""
            )
            raise RuntimeError(
                f"Host Bridge {method} 失败: {exc.code().name}: {exc.details()}{uncertainty}"
            ) from exc

    def _ensure_heartbeat(self) -> None:
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name=f"host-bridge-heartbeat:{self._manager_id}"
            )

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
                reply: pb.HeartbeatReply = await self._call(
                    self._stub.Heartbeat,
                    pb.ContextRequest(context=self._request_context()),
                    method="Heartbeat",
                    timeout=5,
                )
                require_fields(reply, "alive")
                if not reply.alive:
                    raise RuntimeError("Host Bridge 未确认 lease 存活")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._lease_error = exc


def _check_client_identity(socket_path: Path, boot_id: str, token: str) -> None:
    if not socket_path.is_absolute():
        raise ValueError("Host Bridge socket 必须是绝对路径")
    require_text(boot_id, "boot_id")
    require_text(token, "token")


def _validate_requirement_partition(
    requested: list[str], available: list[str], missing: list[str], kind: str
) -> None:
    if sorted(requested) != sorted([*available, *missing]):
        raise RuntimeError(f"Host Bridge {kind} capability partition 不完整")

"""宿主资源控制的公共值与窄授权；代码 owner 校验由内核拥有。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol
from contextlib import AbstractAsyncContextManager
from collections.abc import Mapping
from enum import Enum
import shlex
from pydantic import BaseModel, ConfigDict
from agent.tool_catalog import ToolResult
from pathlib import Path
from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey

WorkloadMode = Literal["candidate", "formal"]


class WorkloadEffectUnknown(RuntimeError):
    """The connection failed after Controller may have changed Docker state."""


@dataclass(frozen=True, slots=True)
class WorkloadEndpoint:
    name: str
    url: str


@dataclass(frozen=True, slots=True)
class WorkloadLease:
    workspace_id: str
    plugin_id: str
    workload: str
    mode: WorkloadMode
    transaction_id: str
    generation_id: str
    container_id: str
    spec_digest: str


@dataclass(frozen=True, slots=True)
class WorkloadStartRequest:
    workspace_id: str
    plugin_id: str
    workload: str
    mode: WorkloadMode
    transaction_id: str
    generation_id: str
    image: str
    command: tuple[str, ...]
    ports: tuple[tuple[str, int], ...]
    data: tuple[tuple[str, str, bool], ...]
    health: tuple[str, str, float]
    limits: tuple[int, float, int]
    loopback_ports: tuple[tuple[str, int], ...] = ()
    user_namespaces: bool = False

    @property
    def spec_digest(self) -> str:
        """计算完整请求规格的固定编码摘要，不包含短命请求身份。"""
        value = {
            "owner": self.plugin_id, "name": self.workload, "image": self.image,
            "command": list(self.command), "ports": [list(item) for item in self.ports],
            "data": [list(item) for item in self.data], "health": list(self.health),
            "limits": list(self.limits), "loopback_ports": [list(item) for item in self.loopback_ports],
            "user_namespaces": self.user_namespaces,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "spec_digest": self.spec_digest}


@dataclass(frozen=True, slots=True)
class WorkloadStartReceipt:
    lease: WorkloadLease
    endpoints: tuple[WorkloadEndpoint, ...]
    adopted_from_generation: str | None


@dataclass(frozen=True, slots=True)
class WorkloadStopReceipt:
    lease: WorkloadLease
    container_absent: bool
    mounts_released: bool


class ControllerGrant(Protocol):
    @property
    def mode(self) -> Literal["candidate", "formal"]: ...
    @property
    def workspace_id(self) -> str: ...
    @property
    def identity(self) -> str: ...
    async def start(self, request: WorkloadStartRequest) -> WorkloadStartReceipt: ...
    async def stop(self, lease: WorkloadLease) -> WorkloadStopReceipt: ...


class ControllerAccess(Protocol):
    def bind(self, ctx: Context) -> ControllerGrant: ...


WORKLOAD_CONTROLLER = ServiceKey[ControllerAccess]("host.workloads.v1")


class HostStatus(Protocol):
    def snapshot(self) -> dict[str, object]: ...


HOST_STATUS = ServiceKey[HostStatus]("host.status.v1")


class ShellProcessManagerProtocol(Protocol):
    """Host-backed or local process owner used by a shell provider."""

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
        shell_snapshot: bool = False,
    ) -> "ExecutionResult": ...

    async def write_stdin(
        self,
        *,
        execution_id: int,
        chars: str,
        yield_time_ms: int,
        max_output_tokens: int,
        owner_session_key: str,
    ) -> "ExecutionResult": ...

    async def terminate_execution(
        self, execution_id: int, *, owner_session_key: str
    ) -> bool: ...

    async def terminate_owner(self, owner_session_key: str) -> "ExecutionCleanupReport": ...

    async def shutdown(self) -> "ExecutionCleanupReport": ...

    async def active_execution_ids(self) -> list[int]: ...


MIN_YIELD_TIME_MS = 250


MIN_EMPTY_YIELD_TIME_MS = 5_000


MAX_YIELD_TIME_MS = 30_000


MAX_WRITE_STDIN_YIELD_TIME_MS = 300_000


DEFAULT_INITIAL_YIELD_TIME_MS = 10_000


DEFAULT_MAX_OUTPUT_TOKENS = 10_000


DEFAULT_HARD_TIMEOUT_S = 4 * 3600


MAX_HARD_TIMEOUT_S = 4 * 3600


class UnknownExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    output: bytes
    wall_time_ms: int
    original_token_count: int
    output_omitted_bytes: int
    execution_id: int | None
    exit_code: int | None
    output_path: str | None
    finish_reason: str


@dataclass(frozen=True)
class ExecutionCleanupFailure:
    execution_id: int
    error_type: str
    message: str


@dataclass(frozen=True)
class ExecutionCleanupReport:
    attempted_execution_ids: tuple[int, ...]
    cleaned_execution_ids: tuple[int, ...]
    failures: tuple[ExecutionCleanupFailure, ...]

    @property
    def failed_execution_ids(self) -> tuple[int, ...]:
        return tuple(failure.execution_id for failure in self.failures)


def clamp_initial_yield_time(yield_time_ms: int) -> int:
    return min(max(yield_time_ms, MIN_YIELD_TIME_MS), MAX_YIELD_TIME_MS)


def clamp_write_stdin_yield_time(
    yield_time_ms: int,
    *,
    has_input: bool,
    max_empty_ms: int = MAX_WRITE_STDIN_YIELD_TIME_MS,
) -> int:
    value = max(yield_time_ms, MIN_YIELD_TIME_MS)
    if has_input:
        return min(value, MAX_YIELD_TIME_MS)
    return min(max(value, MIN_EMPTY_YIELD_TIME_MS), max_empty_ms)


class Processes(Protocol):
    def resolve_shell(self, requested: str | None = None) -> ResolvedShell: ...
    def requirements_checker(self) -> RequirementsChecker | None: ...
    async def exec_command(
        self, ctx: Context, owner_key: str, *, command: str, argv: list[str],
        cwd: Path | None, env: dict[str, str], tty: bool, yield_time_ms: int,
        max_output_tokens: int, hard_timeout_s: int, shell_snapshot: bool = False,
    ) -> ExecutionResult: ...
    async def write_stdin(
        self, ctx: Context, owner_key: str, *, execution_id: int, chars: str,
        yield_time_ms: int, max_output_tokens: int,
    ) -> ExecutionResult: ...
    async def terminate_execution(self, ctx: Context, owner_key: str, execution_id: int) -> bool: ...
    async def terminate_owner(self, ctx: Context, owner_key: str) -> ExecutionCleanupReport: ...


PROCESSES = ServiceKey[Processes]("host.processes.v1")


class ShellKind(str, Enum):
    ZSH = "zsh"
    BASH = "bash"
    POWERSHELL = "powershell"
    SH = "sh"
    CMD = "cmd"


@dataclass(frozen=True)
class ResolvedShell:
    kind: ShellKind
    path: Path

    def derive_argv(self, command: str, *, login: bool, snapshot: Path | None = None) -> list[str]:
        """Build the direct process argv for one shell command."""
        if self.kind in {ShellKind.ZSH, ShellKind.BASH, ShellKind.SH}:
            # 用户环境优先取快照：非交互 shell 只 source 一次性导出的 rc 结果。
            if login and snapshot is not None:
                script = f". {shlex.quote(str(snapshot))} || exit $?; eval {shlex.quote(command)}"
                return [str(self.path), "-c", script]
            return [str(self.path), "-lc" if login else "-c", command]
        if self.kind is ShellKind.POWERSHELL:
            profile_args = [] if login else ["-NoProfile"]
            return [str(self.path), *profile_args, "-Command", command]
        return [str(self.path), "/c", command]


PathStatus = Literal[
    "available", "not_found", "not_directory", "permission_denied",
    "not_file", "too_large", "invalid_text", "io_error", "offline",
]


class DirectoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    path: str


class AgentsChainFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    text: str
    bytes: int


class AgentsChainFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["probe", "read", "budget"]
    path: str | None = None
    status: str | None = None
    error: str | None = None


class AgentsChain(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    root: str | None = None
    files: list[AgentsChainFile]
    failure: AgentsChainFailure | None = None


class PathInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    status: PathStatus
    kind: Literal["file", "directory", "other"] | None = None
    error: str | None = None
    text: str | None = None
    bytes: int | None = None
    items: list[DirectoryEntry] | None = None
    after: str | None = None
    parent: str | None = None
    chain: AgentsChain | None = None


@dataclass(frozen=True)
class RequirementsAvailability:
    available_bins: tuple[str, ...]
    missing_bins: tuple[str, ...]
    available_env: tuple[str, ...]
    missing_env: tuple[str, ...]


class RequirementsChecker(Protocol):
    def check_requirements(self, bins: list[str], env: list[str]) -> RequirementsAvailability: ...


class FileOperation(Protocol):
    async def execute(self, **arguments: Any) -> str | ToolResult: ...
    async def aclose(self) -> None: ...


class PathReader(Protocol):
    async def read(
        self, action: str, path: str, *, base_dir: str | None = None,
        after: str | None = None, limit: int = 100, max_bytes: int = 32768,
    ) -> PathInfo: ...


class Files(Protocol):
    def open(self, name: str, allowed_dir: Path | None = None) -> FileOperation: ...
    def paths(self) -> AbstractAsyncContextManager[PathReader]: ...


FILES = ServiceKey[Files]("host.files.v1")
LIST_DIR_MAX_ENTRIES = 500
LIST_DIR_MAX_BYTES = 10_000

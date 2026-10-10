"""宿主资源控制的公共值与窄授权；代码 owner 校验由内核拥有。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Protocol
from pathlib import Path
from agent.process_runtime import ExecutionResult, ExecutionCleanupReport
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


class Processes(Protocol):
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

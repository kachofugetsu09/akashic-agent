"""宿主授予当前 Context 的执行原子能力，不包含资源目录或启动阶段。"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey
from agent.workloads.model import WorkloadStartRequest, WorkloadStartReceipt, WorkloadLease, WorkloadStopReceipt


class ExecutionGrant(Protocol):
    @property
    def mode(self) -> Literal["candidate", "formal"]: ...
    def command(self, command: tuple[str, ...], cwd: str) -> tuple[str, ...]: ...
    def cwd(self, relative: str) -> Path: ...
    def environment(self, values: Mapping[str, str], candidate_values: Mapping[str, str]) -> dict[str, str]: ...


class ExecutionAccess(Protocol):
    def bind(self, ctx: Context) -> ExecutionGrant: ...


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


EXECUTION = ServiceKey[ExecutionAccess]("host.execution.v1")
WORKLOAD_CONTROLLER = ServiceKey[ControllerAccess]("host.workload_controller.v1")

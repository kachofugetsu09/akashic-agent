"""宿主授予当前 Context 的执行原子能力，不包含资源目录或启动阶段。"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

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
    spec_digest: str
    image: str
    command: tuple[str, ...]
    ports: tuple[tuple[str, int], ...]
    data: tuple[tuple[str, str, bool], ...]
    health: tuple[str, str, float]
    limits: tuple[int, float, int]
    loopback_ports: tuple[tuple[str, int], ...] = ()
    user_namespaces: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def workload_spec_digest(
    *,
    plugin_id: str,
    workload: str,
    image: str,
    command: tuple[str, ...],
    ports: tuple[tuple[str, int], ...],
    data: tuple[tuple[str, str, bool], ...],
    health: tuple[str, str, float],
    limits: tuple[int, float, int],
    loopback_ports: tuple[tuple[str, int], ...] = (),
    user_namespaces: bool = False,
) -> str:
    """Hash the complete immutable Workload spec with one fixed encoding."""

    value = {
        "owner": plugin_id,
        "name": workload,
        "image": image,
        "command": list(command),
        "ports": [list(item) for item in ports],
        "data": [list(item) for item in data],
        "health": list(health),
        "limits": list(limits),
        "loopback_ports": [list(item) for item in loopback_ports],
        "user_namespaces": user_namespaces,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ChildProcess(Protocol):
    """opaque 子进程句柄：stdio 归调用方，进程组终止语义归宿主实现。"""

    process: asyncio.subprocess.Process

    @property
    def group_id(self) -> int | None: ...
    async def terminate(self, *, timeout_s: float) -> None: ...
    async def kill(self, *, timeout_s: float) -> None: ...


class PreparedProcess:
    """宿主签发的冻结执行制品：command/cwd/env 已经过授权校验。

    归属校验靠签发者私有 token，而非可被随意拼造的公开字段；
    制品签发后不可修改，不存在公开的派生/修改入口。
    """

    __slots__ = ("_token", "_command", "_cwd", "_env")

    def __init__(
        self,
        token: object,
        *,
        command: tuple[str, ...],
        cwd: str,
        env: Mapping[str, str],
    ) -> None:
        self._token = token
        self._command = tuple(command)
        self._cwd = cwd
        self._env = MappingProxyType(dict(env))

    @property
    def command(self) -> tuple[str, ...]:
        return self._command

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def env(self) -> Mapping[str, str]:
        return self._env

    def _issued_by(self, token: object) -> bool:
        return self._token is token


class ProcessSpawner(Protocol):
    """受控子进程来源；ExecutionGrant 结构满足，不另立 ServiceKey。"""

    def prepare_process(
        self,
        command: tuple[str, ...],
        cwd: str,
        env: Mapping[str, str],
        candidate_env: Mapping[str, str] = {},
    ) -> PreparedProcess:
        """在边界内执行授权校验并签发冻结的执行制品。"""
        ...
    async def spawn(
        self,
        prepared: PreparedProcess,
        *,
        stdin: object = None,
        stdout: object = None,
        stderr: object = None,
        limit: int | None = None,
    ) -> tuple[ChildProcess, bool]: ...


class ExecutionGrant(ProcessSpawner, Protocol):
    @property
    def mode(self) -> Literal["candidate", "formal"]: ...
    def command(self, command: tuple[str, ...], cwd: str) -> tuple[str, ...]: ...
    def cwd(self, relative: str) -> Path: ...
    def environment(self, values: Mapping[str, str], candidate_values: Mapping[str, str]) -> dict[str, str]: ...
    async def spawn(
        self,
        prepared: PreparedProcess,
        *,
        stdin: object = None,
        stdout: object = None,
        stderr: object = None,
        limit: int | None = None,
    ) -> tuple[ChildProcess, bool]:
        """受控 spawn：只消费本授权签发的 PreparedProcess，返回取消标记。"""
        ...


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

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol
from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey


@dataclass(frozen=True, slots=True)
class WorkloadPort:
    name: str
    number: int
    loopback: int | None = None


@dataclass(frozen=True, slots=True)
class WorkloadData:
    name: str
    target: str
    writable: bool = True


@dataclass(frozen=True, slots=True)
class WorkloadHealth:
    port: str
    path: str = "/health"
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class WorkloadLimits:
    """Limit workload resources; zero leaves that resource unlimited."""

    memory_mb: int
    cpu_count: float
    pids: int


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    image: str
    command: tuple[str, ...]
    ports: tuple[WorkloadPort, ...]
    data: tuple[WorkloadData, ...]
    health: WorkloadHealth
    limits: WorkloadLimits
    user_namespaces: bool = False


class WorkloadHandle(Protocol):
    def url(self, ctx: Context, port: str) -> str: ...
    def borrow(self, ctx: Context) -> AbstractAsyncContextManager[Mapping[str, str]]: ...
    async def aclose(self) -> None: ...


class Workloads(Protocol):
    @property
    def root_instance_token(self) -> object: ...
    async def register(self, ctx: Context, workload: Workload) -> WorkloadHandle: ...
    def urls(self, ctx: Context) -> Mapping[tuple[str, str], str]: ...


WORKLOADS = ServiceKey[Workloads]("core.workloads")

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol
from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey


@dataclass(frozen=True, slots=True)
class ManagedProcessDefinition:
    name: str
    command: tuple[str, ...]
    cwd: str = "."
    env: Mapping[str, str] = field(default_factory=dict)
    candidate_env: Mapping[str, str] = field(default_factory=dict)
    port_env: str = "PORT"
    formal_port: int = 0
    readiness_path: str = "/health"
    startup_timeout_seconds: float = 15.0


class ManagedProcessHandle(Protocol):
    def port(self, ctx: Context) -> int: ...
    def borrow(self, ctx: Context) -> AbstractAsyncContextManager[int]: ...
    async def aclose(self) -> None: ...


class ManagedProcesses(Protocol):
    async def register(self, ctx: Context, definition: ManagedProcessDefinition) -> ManagedProcessHandle: ...


MANAGED_PROCESSES = ServiceKey[ManagedProcesses]("core.managed_processes")

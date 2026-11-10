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
    # formal 进程注册即返回、在后台等就绪：健康项不挡宿主就绪，borrow 等到可用。
    # 直连端口的消费者要先 borrow 一次再访问；candidate 校验仍同步等就绪。
    background_ready: bool = False


class ManagedProcessHandle(Protocol):
    def check(self, ctx: Context) -> None: ...
    def port(self, ctx: Context) -> int: ...
    def borrow(self, ctx: Context) -> AbstractAsyncContextManager[int]: ...
    async def aclose(self) -> None: ...


class ManagedProcesses(Protocol):
    async def register(self, ctx: Context, definition: ManagedProcessDefinition) -> ManagedProcessHandle: ...


MANAGED_PROCESSES = ServiceKey[ManagedProcesses]("processes.managed.v1")

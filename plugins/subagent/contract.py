"""Delegated execution contracts owned by the subagent capability.

Native execution remains a bound program: moving its declaration must not
change saved service identities or replay previously accepted work.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Literal, Protocol
from agent.plugin_composition import Context
from agent.plugin_composition.effect import Effect

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.tasks import Task
from plugins.ledger.contract import Message, MessageReader

if TYPE_CHECKING:
    from .request import Request


SUBAGENT_PROGRAM = ServiceKey[Callable[[Task, MessageReader, "Request"], Awaitable[Message]]](
    "subagent.program.v1"
)


@dataclass(frozen=True, slots=True)
class AgentActivity:
    kind: Literal["status", "message", "tool", "result"]
    text: str


@dataclass(frozen=True, slots=True)
class AgentResult:
    status: Literal["completed", "failed"]
    text: str


@dataclass(frozen=True, slots=True)
class AgentTask:
    id: str
    prompt: str
    cwd: str


@dataclass(frozen=True, slots=True)
class AgentRunner:
    """One installed external harness instance, not a host-model provider.

    The provider owns protocol, authentication and native tools. Cancellation
    must finish child teardown before run returns or raises. Resume, parent
    tool grants and inherited model settings are deliberately not promised.
    """
    name: str
    description: str
    progress: bool
    policy: Literal["read-only", "isolated-write"]
    run: Callable[[AgentTask, Callable[[AgentActivity], Awaitable[None]]], Awaitable[AgentResult]]


class AgentWire(Protocol):
    async def send(self, value: Mapping[str, object]) -> None: ...
    async def receive(self) -> dict[str, object]: ...


class AgentRunners(Protocol):
    def connect(self, ctx: Context, owner: str, argv: tuple[str, ...], cwd: str,
                env: dict[str, str]) -> AbstractAsyncContextManager[AgentWire]: ...
    async def register(self, ctx: Context, runner: AgentRunner) -> Effect: ...
    def catalog(self) -> tuple[Mapping[str, object], ...]: ...


AGENT_RUNNERS = ServiceKey[AgentRunners]("subagent.runners.v1")

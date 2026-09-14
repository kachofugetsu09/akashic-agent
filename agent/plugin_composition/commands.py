"""命令服务的窄协议与值对象；实现由显式选择的 provider 提供。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey

CommandHandler = Callable[
    ["CommandInvocation"],
    "CommandResult | Awaitable[CommandResult]",
]
CommandResultKind = Literal["success", "error"]


@dataclass(frozen=True, slots=True)
class CommandInvocation:
    name: str
    raw_input: str
    session_key: str
    channel: str
    chat_id: str
    sender: str
    message_id: str = ""


@dataclass(frozen=True, slots=True)
class CommandResult:
    kind: CommandResultKind
    text: str


@dataclass(frozen=True, slots=True)
class CommandDefinition:
    name: str
    description: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()
    input_hint: str | None = None
    read_only: bool = False
    recover: CommandHandler | None = None


@dataclass(frozen=True, slots=True)
class CommandDescriptor:
    name: str
    description: str
    aliases: tuple[str, ...]
    input_hint: str | None
    owner: str


@dataclass(frozen=True, slots=True)
class CommandExecution:
    name: str
    result: CommandResult


class CommandRecoveryRequired(RuntimeError):
    """命令效果没有可确认的领域回执，调用者必须保留未知结果。"""


class CommandCatalog(Protocol):
    """已封存目录的发现、精确绑定与执行能力。"""

    @property
    def descriptors(self) -> tuple[CommandDescriptor, ...]: ...

    def bind(self, bindings: Bindings, line: str) -> str | None: ...

    async def execute(
        self, line: str, *, session_key: str, channel: str, chat_id: str,
        sender: str, message_id: str = "", recover: bool = False,
    ) -> CommandExecution | None: ...


class Commands(Protocol):
    """贡献方注册命令，消费者读取 provider 封存的目录。"""

    async def register(self, ctx: Context, definition: CommandDefinition) -> None: ...

    def freeze(self) -> CommandCatalog: ...


COMMANDS = ServiceKey[Commands]("core.commands")

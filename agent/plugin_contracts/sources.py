"""来源提交后的同步通知；监听者返回前必须取得所需活动占位。"""

from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.channels import ChannelInboundMessage
from agent.plugin_composition.events import EmitEventKey
from agent.plugin_composition.messages import MessageReader, MessageWriter, OwnerTransaction
from agent.plugin_composition.tasks import RestartGate, Task, TaskAdmission
from agent.plugin_contracts import (
    ContentPart,
    ContentReferences,
    Control,
    Input,
    Message,
)


@dataclass(frozen=True)
class SourceChanged:
    reader: MessageReader
    source: str


SOURCE_CHANGED = EmitEventKey[SourceChanged]("source.changed.v1")
SOURCE_CHANGED_V2 = EmitEventKey[SourceChanged]("source.changed.v2")


class SourceGuard(Protocol):
    """来源固定的执行前提；输出可以属于另一来源，但不能转移控制权。"""

    def __call__(self, *, transaction: OwnerTransaction | None = None) -> None: ...


CompletionProgram = Callable[[Task, MessageReader, SourceGuard], Awaitable[Message]]


class SourceSessionBase(Protocol):
    async def accept(self, message_id: str, body: Input) -> Message: ...
    async def control(
        self, message_id: str, body: Control, *, expected_head: int, handle: str | None
    ) -> Message: ...
    async def pause(self, message_id: str) -> Message: ...
    async def resume(self, message_id: str, input_id: str) -> Message: ...
    async def start(
        self, program: Callable[[Task, MessageReader, str], Awaitable[object]]
    ) -> Task | None: ...
    async def record_failure(
        self, error: BaseException, *, boundary: int | None = None
    ) -> None: ...
    async def wait_capacity(self) -> None: ...


class SourceSession(SourceSessionBase, Protocol):
    async def complete(
        self, program: Callable[[Task, MessageReader], Awaitable[Message]]
    ) -> Message: ...


class GuardedSourceSession(SourceSessionBase, Protocol):
    async def complete(self, program: CompletionProgram) -> Message: ...


class SessionFactory(Protocol):
    def __call__(
        self,
        *,
        reader: MessageReader,
        inputs: MessageWriter,
        controls: MessageWriter,
        tasks: TaskAdmission,
        changed: Callable[[MessageReader, str], None] | None = None,
        restart_gate: RestartGate | None = None,
    ) -> SourceSession: ...
    @staticmethod
    def needs_reply(
        messages: Sequence[Message] | MessageReader, source: str
    ) -> bool: ...


class GuardedSessionFactory(Protocol):
    def __call__(
        self,
        *,
        reader: MessageReader,
        inputs: MessageWriter,
        controls: MessageWriter,
        tasks: TaskAdmission,
        changed: Callable[[MessageReader, str], None] | None = None,
        restart_gate: RestartGate | None = None,
    ) -> GuardedSourceSession: ...
    @staticmethod
    def needs_reply(
        messages: Sequence[Message] | MessageReader, source: str
    ) -> bool: ...


Accept = Callable[[str, str, ChannelInboundMessage], Awaitable[Message]]


@dataclass(frozen=True)
class Source:
    context: Context
    name: str
    open: Callable[[str], SourceSession]
    needs_reply: Callable[[MessageReader], bool]
    accept: Accept | None = None
    channels: tuple[str, ...] | None = ()


class Sources(Protocol):
    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        open: Callable[[str], SourceSession],
        needs_reply: Callable[[MessageReader], bool],
        accept: Accept | None = None,
        channels: tuple[str, ...] | None = (),
    ) -> Effect: ...
    def needs_reply(self, reader: MessageReader, source: str) -> bool: ...
    def entries(self) -> tuple[Source, ...]: ...
    def changes(self) -> AsyncGenerator[tuple[Source, ...], None]: ...
    async def accept(
        self, session_id: str, message_id: str, message: ChannelInboundMessage
    ) -> Message: ...


@dataclass(frozen=True)
class GuardedSource:
    context: Context
    name: str
    open: Callable[[str], GuardedSourceSession]
    needs_reply: Callable[[MessageReader], bool]
    accept: Accept | None = None
    channels: tuple[str, ...] | None = ()


class GuardedSources(Protocol):
    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        open: Callable[[str], GuardedSourceSession],
        needs_reply: Callable[[MessageReader], bool],
        accept: Accept | None = None,
        channels: tuple[str, ...] | None = (),
    ) -> Effect: ...
    def needs_reply(self, reader: MessageReader, source: str) -> bool: ...
    def entries(self) -> tuple[GuardedSource, ...]: ...
    def changes(self) -> AsyncGenerator[tuple[GuardedSource, ...], None]: ...
    async def accept(
        self, session_id: str, message_id: str, message: ChannelInboundMessage
    ) -> Message: ...


class ConversationComplete(Protocol):
    async def __call__(
        self,
        session_id: str,
        program: Callable[[Task, MessageReader], Awaitable[Message]],
    ) -> Message: ...


class ConversationCompleteV2(Protocol):
    async def __call__(
        self, session_id: str, program: CompletionProgram,
    ) -> Message: ...


SOURCES = ServiceKey[Sources]("sources.v2")
SOURCE_INTERRUPT = ServiceKey[
    Callable[[MessageReader, str, str], Awaitable[bool]]
]("source.interrupt.v1")
SOURCE_SESSION = ServiceKey[SessionFactory]("source.session.v1")
SOURCE_CHECK = ServiceKey[Callable[[Task, MessageReader, str, int], None]](
    "source.check.v1"
)

class SourceCheck(Protocol):
    def __call__(
        self, task: Task, reader: MessageReader, source: str, through_seq: int, *,
        transaction: OwnerTransaction | None = None,
    ) -> None:
        """首次效果把检查放在同一 Core transaction；其它阶段只读已提交前提。"""
        ...


# 旧常量保持原值；旧归档不会因导入当前 Core 而隐式升级合同。
SOURCES_V3 = ServiceKey[Sources]("sources.v3")
SOURCES_V4 = ServiceKey[GuardedSources]("sources.v4")
SOURCE_SESSION_V2 = ServiceKey[SessionFactory]("source.session.v2")
SOURCE_SESSION_V3 = ServiceKey[GuardedSessionFactory]("source.session.v3")
SOURCE_CHECK_V2 = ServiceKey[SourceCheck]("source.check.v2")
SOURCE_INTERRUPT_V2 = ServiceKey[
    Callable[[MessageReader, str, str], Awaitable[bool]]
]("source.interrupt.v2")

CONVERSATION_COMPLETE = ServiceKey[ConversationComplete]("conversation.complete.v1")
CONVERSATION_COMPLETE_V2 = ServiceKey[ConversationCompleteV2]("conversation.complete.v2")
CONVERSATION_COMMANDS = ServiceKey[
    Callable[[Task, MessageReader, str], Awaitable[Message | None]]
]("conversation.commands.v1")


class OriginCheck(Protocol):
    def __call__(self, part: ContentPart) -> ContentReferences: ...


CHECK_ORIGIN = ServiceKey[OriginCheck]("conversation.check_origin.v1")

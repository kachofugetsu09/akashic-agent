"""来源提交后的同步通知；监听者返回前必须取得所需活动占位。"""

from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.channels import ChannelInboundMessage
from agent.plugin_composition.events import EmitEventKey
from agent.plugin_composition.messages import MessageReader, MessageWriter
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


class SourceSession(Protocol):
    async def accept(self, message_id: str, body: Input) -> Message: ...
    async def control(
        self, message_id: str, body: Control, *, expected_head: int, handle: str | None
    ) -> Message: ...
    async def pause(self, message_id: str) -> Message: ...
    async def resume(self, message_id: str, input_id: str) -> Message: ...
    async def complete(
        self, program: Callable[[Task, MessageReader], Awaitable[Message]]
    ) -> Message: ...
    async def start(
        self, program: Callable[[Task, MessageReader, str], Awaitable[object]]
    ) -> Task | None: ...
    async def record_failure(
        self, error: BaseException, *, boundary: int | None = None
    ) -> None: ...
    async def wait_capacity(self) -> None: ...


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


class ConversationComplete(Protocol):
    async def __call__(
        self,
        session_id: str,
        program: Callable[[Task, MessageReader], Awaitable[Message]],
    ) -> Message: ...


SOURCES = ServiceKey[Sources]("sources.v2")
SOURCE_SESSION = ServiceKey[SessionFactory]("source.session.v1")
SOURCE_CHECK = ServiceKey[Callable[[Task, MessageReader, str, int], None]](
    "source.check.v1"
)
CONVERSATION_COMPLETE = ServiceKey[ConversationComplete]("conversation.complete.v1")
CONVERSATION_COMMANDS = ServiceKey[
    Callable[[Task, MessageReader, str], Awaitable[Message | None]]
]("conversation.commands.v1")


class OriginCheck(Protocol):
    def __call__(self, part: ContentPart) -> ContentReferences: ...


CHECK_ORIGIN = ServiceKey[OriginCheck]("conversation.check_origin.v1")

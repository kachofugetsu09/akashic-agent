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
    Control,
    Input,
    Message,
)


@dataclass(frozen=True)
class SourceChangedV3:
    """来源提交点的待回复判定；监听者同步占位，不重读历史。"""
    reader: MessageReader
    source: str
    pending: bool


SOURCE_CHANGED_V3 = EmitEventKey[SourceChangedV3]("source.changed.v3")


class SourceGuard(Protocol):
    """来源固定的执行前提；输出可以属于另一来源，但不能转移控制权。"""

    def __call__(self, *, transaction: OwnerTransaction | None = None) -> None: ...


CompletionProgram = Callable[[Task, MessageReader, SourceGuard], Awaitable[Message]]


class GuardedSourceSession(Protocol):
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

    async def complete(self, program: CompletionProgram) -> Message: ...


Accept = Callable[[str, str, ChannelInboundMessage], Awaitable[Message]]


class SourceCheck(Protocol):
    def __call__(
        self, task: Task, reader: MessageReader, source: str, through_seq: int, *,
        transaction: OwnerTransaction | None = None,
    ) -> None:
        """首次效果把检查放在同一 Core transaction；其它阶段只读已提交前提。"""
        ...


SOURCE_CHECK_V2 = ServiceKey[SourceCheck]("source.check.v2")
SOURCE_INTERRUPT_V2 = ServiceKey[
    Callable[[MessageReader, str, str], Awaitable[bool]]
]("source.interrupt.v2")

class SessionFactoryV4(Protocol):
    """当前来源读取可等待；通知携带同一提交点的待回复事实。"""
    def __call__(
        self, *, reader: MessageReader, inputs: MessageWriter, controls: MessageWriter,
        tasks: TaskAdmission,
        changed: Callable[[MessageReader, str, bool], None] | None = None,
        restart_gate: RestartGate | None = None,
    ) -> GuardedSourceSession: ...
    @staticmethod
    async def needs_reply(messages: Sequence[Message] | MessageReader, source: str) -> bool: ...


@dataclass(frozen=True)
class AsyncSource:
    context: Context
    name: str
    open: Callable[[str], GuardedSourceSession]
    needs_reply: Callable[[MessageReader], Awaitable[bool]]
    accept: Accept | None = None
    channels: tuple[str, ...] | None = ()


class SourcesV5(Protocol):
    async def register(
        self, ctx: Context, *, name: str, open: Callable[[str], GuardedSourceSession],
        needs_reply: Callable[[MessageReader], Awaitable[bool]], accept: Accept | None = None,
        channels: tuple[str, ...] | None = (),
    ) -> Effect: ...
    async def needs_reply(self, reader: MessageReader, source: str) -> bool: ...
    def entries(self) -> tuple[AsyncSource, ...]: ...
    def changes(self) -> AsyncGenerator[tuple[AsyncSource, ...], None]: ...
    async def accept(self, session_id: str, message_id: str, message: ChannelInboundMessage) -> Message: ...


SOURCES_V5 = ServiceKey[SourcesV5]("sources.v5")
SOURCE_SESSION_V4 = ServiceKey[SessionFactoryV4]("source.session.v4")

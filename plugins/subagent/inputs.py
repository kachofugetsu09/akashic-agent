"""subagent 消费的内容、工具与投递能力。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.tasks import Task
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.messages import MessageReader, MessageWriter
from agent.plugin_contracts import Body, ContentPart, ContentReferences, Message, CallRef


@dataclass(frozen=True)
class Result:
    outcome: Literal["success", "denied", "error", "interrupted"]
    parts: tuple[ContentPart, ...]


class ContentChecks(Protocol):
    def check_text(self, part: ContentPart) -> ContentReferences: ...


class ToolRef(Protocol):
    @property
    def name(self) -> str: ...


class ToolView(Protocol):
    @property
    def refs(self) -> tuple[ToolRef, ...]: ...


class ToolCatalog(Protocol):
    async def declare_group(self, ctx: Context, *, description: str) -> Effect: ...
    async def register(
        self, ctx: Context, *, name: str, description: str, parameters: Mapping[str, object],
        open: Callable[[Mapping[str, object]], AbstractAsyncContextManager[object]],
        capture: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
        idempotent: bool = False, risk: Literal["read-only", "read-write", "external-side-effect"] = "read-write",
    ) -> ToolRef: ...
    def bind(self, ref: ToolRef, bindings: Bindings) -> str: ...


class Selection(Protocol):
    @property
    def sinks(self) -> tuple[str, ...]: ...


class Receipt(Protocol):
    @property
    def status(self) -> str: ...
    @property
    def error(self) -> str | None: ...


class Deliveries(Protocol):
    def prepare(self, reader: MessageReader, message: Message,
                sinks: tuple[Mapping[str, object], ...]) -> Selection: ...
    def receipt(self, message_id: str, sink: str) -> Receipt | None: ...
    async def send(self, message_id: str, sink: str) -> Receipt: ...


class Delivery(Protocol):
    def open(self, consumer: Context) -> Deliveries: ...


class Senders(Protocol):
    def bind_all(self, bindings: Bindings) -> Mapping[str, str]: ...


CONTENT = ServiceKey[ContentChecks]("content.v2")
TOOLS = ServiceKey[ToolCatalog]("tools.v1")
ALL_TOOLS = ServiceKey[Callable[[], ToolView]]("tools.all.v1")
DELIVERY = ServiceKey[Delivery]("delivery.v1")
DELIVERY_SENDERS = ServiceKey[Senders]("delivery.senders.v1")


class CallSource(Protocol):
    @property
    def call_ref(self) -> CallRef: ...
    @property
    def messages(self) -> tuple[Message, ...]: ...


class Conversation(Protocol):
    async def complete(self, program: Callable[[Task, MessageReader], Awaitable[Message]]) -> Message: ...


class BindSavedTool(Protocol):
    async def __call__(self, bindings: Bindings, binding_id: str, *, configuration: Mapping[str, object]) -> str: ...


CONVERSATION = ServiceKey[Callable[[str], Conversation]]("conversation.v1")
CHECK_ORIGIN = ServiceKey[Callable[[ContentPart], ContentReferences]]("conversation.check_origin.v1")
REPLY_PROGRAM = ServiceKey[Callable[[Task, MessageReader, str, Sequence[Mapping[str, object]]], Awaitable[Message]]]("reply.program.v2")
TOOL_BIND_SAVED = ServiceKey[BindSavedTool]("tools.bind-saved.v1")

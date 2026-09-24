"""plugin_update 消费的内容、工具与投递能力。"""
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
    def select(self, name: str) -> ToolRef: ...


class ToolCatalog(Protocol):
    async def declare_group(self, ctx: Context, *, description: str) -> Effect: ...
    async def register(
        self, ctx: Context, *, name: str, description: str, parameters: Mapping[str, object],
        open: Callable[[Mapping[str, object]], AbstractAsyncContextManager[object]],
        capture: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
        idempotent: bool = False, risk: Literal["read-only", "read-write", "external-side-effect"] = "read-write",
    ) -> ToolRef: ...
    def view(self, *refs: ToolRef) -> ToolView: ...


class Selection(Protocol):
    @property
    def sinks(self) -> tuple[str, ...]: ...


class Receipt(Protocol):
    @property
    def status(self) -> str: ...
    @property
    def error(self) -> str | None: ...


class Deliveries(Protocol):
    def publish(self, writer: MessageWriter, message_id: str, body: Body,
                sinks: tuple[Mapping[str, object], ...], *, passive: bool = False) -> tuple[Message, Selection]: ...
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


class InputOrigin(Protocol):
    def __call__(self, reader: MessageReader, source: str, *, through_seq: int) -> tuple[str, str] | None: ...


INPUT_ORIGIN = ServiceKey[InputOrigin]("delivery.input-origin.v1")

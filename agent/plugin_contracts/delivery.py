"""投递选择、发送回执和 sender 注册的公共合同。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from datetime import datetime
from typing import Literal, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.messages import MessageReader, MessageWriter
from agent.plugin_composition.tasks import Task
from agent.plugin_contracts import Body, Message

Status = Literal["delivered", "rejected", "failed"]


class Sink(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def binding_id(self) -> str: ...
    @property
    def address(self) -> str: ...


class Receipt(Protocol):
    @property
    def status(self) -> Status: ...
    @property
    def provider_ids(self) -> tuple[str, ...]: ...
    @property
    def error(self) -> str | None: ...


class Selection(Protocol):
    @property
    def session_id(self) -> str: ...
    @property
    def recovery_owner(self) -> str: ...
    @property
    def passive(self) -> bool: ...
    @property
    def sinks(self) -> tuple[str, ...]: ...


class Sender(Protocol):
    @property
    def idempotent(self) -> bool: ...
    async def send(self, key: str, address: str, message: Message) -> Receipt: ...
    async def query(self, key: str, address: str) -> Receipt | None:
        """None 表示没有确认回执，不能推断外部效果没有发生。"""
        ...


class Senders(Protocol):
    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        idempotent: bool,
        open: Callable[[], AbstractAsyncContextManager[Sender]],
    ) -> Effect: ...
    def registered_names(self) -> tuple[str, ...]: ...
    def bind(self, name: str, bindings: Bindings) -> str: ...
    def bind_all(self, bindings: Bindings) -> Mapping[str, str]: ...
    def open(
        self, metadata: Mapping[str, object]
    ) -> AbstractAsyncContextManager[Sender]: ...


class Deliveries(Protocol):
    """已授权调用者的投递入口；恢复沿原选择与回执，不重选目标。"""

    def prepare(
        self,
        reader: MessageReader,
        message: Message,
        sinks: tuple[Mapping[str, object], ...],
        *,
        passive: bool = False,
    ) -> Selection: ...
    def publish(
        self,
        writer: MessageWriter,
        message_id: str,
        body: Body,
        sinks: tuple[Mapping[str, object], ...],
        *,
        passive: bool = False,
    ) -> tuple[Message, Selection]: ...
    def consume(
        self,
        reader: MessageReader,
        message: Message,
        sinks: tuple[Mapping[str, object], ...] | None,
        *,
        passive: bool = False,
    ) -> Selection | None: ...
    def cursor(self, session_id: str) -> int: ...
    def selection(self, message_id: str) -> Selection | None: ...
    def add(self, message_id: str, sink: Mapping[str, object]) -> None: ...
    def destination(self, message_id: str, sink: str) -> Sink: ...
    def receipt(self, message_id: str, sink: str) -> Receipt | None: ...
    def pending(self) -> tuple[tuple[str, str], ...]: ...
    def activity(self, channel: str, address: str) -> AbstractContextManager[None]: ...
    async def wait_idle(self, channel: str, address: str) -> None: ...
    async def start(
        self,
        message_id: str,
        sink: str,
        *,
        before_start: Callable[[], str | None] | None = None,
    ) -> Task: ...
    async def send(
        self,
        message_id: str,
        sink: str,
        *,
        before_start: Callable[[], str | None] | None = None,
    ) -> Receipt: ...
    async def retry(self, message_id: str, sink: str) -> Receipt: ...
    async def cancel_prepared(
        self, message_id: str, sink: str, reason: str
    ) -> bool: ...


class Delivery(Protocol):
    def open(self, consumer: Context) -> Deliveries: ...


class DeliveredMessage(Protocol):
    @property
    def message(self) -> Message: ...
    @property
    def confirmed_at(self) -> datetime: ...


class DeliveryHistory(Protocol):
    def recent(
        self,
        *,
        since: datetime,
        until: datetime,
        limit: int,
        excluded_sources: frozenset[str] = frozenset(),
        visibility: Literal["listed", "internal"] | None = None,
    ) -> tuple[DeliveredMessage, ...]: ...
    def status(self, message_id: str, sink: str) -> Mapping[str, object] | None: ...


class FinalOutputTurn(Protocol):
    @property
    def source(self) -> str: ...
    @property
    def ending_message_id(self) -> str | None: ...
    @property
    def message_ids(self) -> tuple[str, ...]: ...


class FinalOutputWaiter(Protocol):
    async def wait(self, reader: MessageReader, turn: FinalOutputTurn) -> None: ...


class FinalOutputDelivery(FinalOutputWaiter, Protocol):
    def register(self, source: str, provider: FinalOutputWaiter) -> None: ...
    def unregister(self, source: str, provider: FinalOutputWaiter) -> None: ...


DELIVERY = ServiceKey[Delivery]("delivery.v1")
DELIVERY_SENDERS = ServiceKey[Senders]("delivery.senders.v1")
DELIVERY_READ = ServiceKey[DeliveryHistory]("delivery.read.v1")
FINAL_OUTPUT_DELIVERY = ServiceKey[FinalOutputDelivery]("delivery.final_output.v1")


class InputOrigin(Protocol):
    def __call__(
        self, reader: MessageReader, source: str, *, through_seq: int
    ) -> tuple[str, str] | None: ...


INPUT_ORIGIN = ServiceKey[InputOrigin]("delivery.input-origin.v1")

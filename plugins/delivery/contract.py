"""投递选择、发送回执和 sender 注册的公共合同。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
import re
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
    async def candidate(self, ctx: Context, *, name: str, title: str, route: str, status: Callable[[], Mapping[str, object]]) -> Effect: ...
    def candidates(self) -> tuple[Mapping[str, object], ...]: ...
    def bind(self, name: str, bindings: Bindings) -> str: ...
    def bind_all(self, bindings: Bindings) -> Mapping[str, str]: ...
    def open(
        self, metadata: Mapping[str, object]
    ) -> AbstractAsyncContextManager[Sender]: ...


StartGuard = Callable[[], AbstractAsyncContextManager[str | None]]


class GuardedDeliveries(Protocol):
    """持久准备可等待；领域 guard 覆盖前提检查到首次 started 提交。"""

    async def prepare_async(self, reader: MessageReader, message: Message,
                            sinks: tuple[Mapping[str, object], ...], *, passive: bool = False) -> Selection: ...
    async def publish_async(self, writer: MessageWriter, message_id: str, body: Body,
                            sinks: tuple[Mapping[str, object], ...], *, passive: bool = False) -> tuple[Message, Selection]: ...
    async def consume_async(self, reader: MessageReader, message: Message,
                            sinks: tuple[Mapping[str, object], ...] | None, *, passive: bool = False) -> Selection | None: ...
    async def consume_batch_async(self, reader: MessageReader,
                                  items: tuple[tuple[Message, tuple[Mapping[str, object], ...] | None], ...],
                                  *, passive: bool = False) -> tuple[Selection | None, ...]: ...
    async def add_async(self, message_id: str, sink: Mapping[str, object]) -> None: ...
    def cursor(self, session_id: str) -> int: ...
    def selection(self, message_id: str) -> Selection | None: ...
    def destination(self, message_id: str, sink: str) -> Sink: ...
    def receipt(self, message_id: str, sink: str) -> Receipt | None: ...
    def pending(self) -> tuple[tuple[str, str], ...]: ...
    def activity(self, channel: str, address: str) -> AbstractContextManager[None]: ...
    async def wait_idle(self, channel: str, address: str) -> None: ...
    async def start(self, message_id: str, sink: str, *,
                    start_guard: StartGuard | None = None) -> Task: ...
    async def send(self, message_id: str, sink: str, *,
                   start_guard: StartGuard | None = None) -> Receipt: ...
    async def retry(self, message_id: str, sink: str) -> Receipt: ...
    async def cancel_prepared(
        self, message_id: str, sink: str, reason: str
    ) -> bool: ...


class GuardedDelivery(Protocol):
    def open(self, consumer: Context) -> GuardedDeliveries: ...


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


DELIVERY_GUARDED_START = ServiceKey[GuardedDelivery]("delivery.guarded-start.v1")
DELIVERY_SENDERS = ServiceKey[Senders]("delivery.senders.v1")
DELIVERY_READ = ServiceKey[DeliveryHistory]("delivery.read.v1")
FINAL_OUTPUT_DELIVERY = ServiceKey[FinalOutputDelivery]("delivery.final_output.v1")


@dataclass(frozen=True, slots=True)
class SenderDefinition:
    """固定 sender 的 owner 与幂等协议；归档后不重新选择目标。"""

    name: str
    owner: str
    idempotent: bool

    @staticmethod
    def key(name: str) -> ServiceKey[SenderDefinition]:
        """Wake 从已选渠道名声明依赖；sender 注册在同一寿命发布此事实。"""
        if re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) is None:
            raise ValueError("发送 adapter 名称无效")
        return ServiceKey[SenderDefinition](f"delivery.sender.{name}.v1")

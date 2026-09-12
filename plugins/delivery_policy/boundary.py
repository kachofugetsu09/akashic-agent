"""delivery_policy 需要的最小普通插件边界。

这些类型只描述消费者实际使用的能力。Delivery 仍拥有持久记录、发送
回执和未知效果恢复；策略只提交目的地结构并读取结果。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Literal, Protocol

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.messages import MessageReader, MessageWriter
from agent.plugin_contracts import Body, ContentPart, ContentReferences, Message
from agent.plugin_composition.tasks import ExternalRootPermit


SinkInput = Mapping[str, object]


class OriginCheck(Protocol):
    """conversation owner 对已接纳 channel.origin 的校验入口。"""

    def __call__(self, part: ContentPart) -> ContentReferences: ...


ORIGIN_CHECK = ServiceKey[OriginCheck]("conversation.check_origin.v1")


class SinkView(Protocol):
    """Delivery 返回的固定目的地只读视图。"""

    name: str
    binding_id: str
    address: str


class ReceiptView(Protocol):
    """Delivery 在真实效果边界产生的结构回执。"""

    status: Literal["delivered", "rejected", "failed"]
    provider_ids: tuple[str, ...]
    error: str | None


class SelectionView(Protocol):
    """首次选路的只读结果。"""

    session_id: str
    recovery_owner: str
    passive: bool
    sinks: tuple[str, ...]


class DeliveryTask(Protocol):
    """Delivery 返回的真实发送任务句柄。"""

    def on_done(self, callback: Callable[[], None]) -> None: ...


class DeliveryExecution(Protocol):
    """策略可用的发送 owner 入口，不暴露 records 或 provider。"""

    def prepare(
        self,
        reader: MessageReader,
        message: Message,
        sinks: tuple[SinkInput, ...],
        *,
        passive: bool = False,
    ) -> SelectionView: ...

    def publish(
        self,
        writer: MessageWriter,
        message_id: str,
        body: Body,
        sinks: tuple[SinkInput, ...],
        *,
        passive: bool = False,
    ) -> tuple[Message, SelectionView]: ...

    def consume(
        self,
        reader: MessageReader,
        message: Message,
        sinks: tuple[SinkInput, ...] | None,
        *,
        passive: bool = False,
    ) -> SelectionView | None: ...

    def cursor(self, session_id: str) -> int: ...

    def selection(self, message_id: str) -> SelectionView | None: ...

    def destination(self, message_id: str, sink: str) -> SinkView: ...

    def receipt(self, message_id: str, sink: str) -> ReceiptView | None: ...

    def pending(self) -> tuple[tuple[str, str], ...]: ...

    def activity(self, channel: str, address: str) -> AbstractContextManager[None]: ...

    async def start(self, message_id: str, sink: str) -> DeliveryTask: ...

    async def send(self, message_id: str, sink: str) -> ReceiptView: ...


class DeliveryAdmission(Protocol):
    def open(self, consumer: Context) -> DeliveryExecution: ...


DELIVERY = ServiceKey[DeliveryAdmission]("delivery.v1")


class SenderRegistry(Protocol):
    def bind(self, name: str, bindings: Bindings) -> str: ...


DELIVERY_SENDERS = ServiceKey[SenderRegistry]("delivery.senders.v1")


class FinalOutputTurn(Protocol):
    source: str
    ending_message_id: str | None
    message_ids: tuple[str, ...]


class FinalOutputWaiter(Protocol):
    async def wait(self, reader: MessageReader, turn: FinalOutputTurn) -> None: ...


class FinalOutputRegistry(Protocol):
    def register(self, source: str, provider: FinalOutputWaiter) -> None: ...


FINAL_OUTPUT_DELIVERY = ServiceKey[FinalOutputRegistry]("delivery.final_output.v1")


class Completion(Protocol):
    def activity(self, reader: MessageReader, source: str) -> AbstractContextManager[None]: ...

    def __call__(
        self,
        reader: MessageReader,
        source: str,
        *,
        child_permit: Callable[[], ExternalRootPermit] | None = None,
    ) -> AbstractAsyncContextManager[None]: ...


REPLY_COMPLETION = ServiceKey[Completion]("reply.completion.v1")

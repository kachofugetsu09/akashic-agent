"""message_push 依赖的局部能力边界。

ServiceKey 只按稳定名称匹配，真实实现仍由各 owner 注册；这里不复制
Delivery、Tools、TurnProjection 或 Content 的内部对象。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.messages import MessageReader, MessageWriter
from agent.plugin_contracts import Body, CallRef, ContentPart, ContentReferences, Message


class CallSource(Protocol):
    """工具 owner 提供的只读调用前缀。"""

    call_ref: CallRef
    messages: tuple[Message, ...]


ToolOutcome = Literal["success", "denied", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """message_push 自有的结构结果；tools owner 在执行边界重新校验。"""

    outcome: ToolOutcome
    parts: tuple[ContentPart, ...]


class BoundTool(Protocol):
    @property
    def idempotent(self) -> bool: ...

    async def prepare(
        self,
        arguments: Mapping[str, object],
        source: CallSource | None = None,
    ) -> Mapping[str, object] | str: ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResult: ...

    async def query(self, key: str) -> ToolResult | None: ...


class ToolRef(Protocol):
    name: str
    description: Mapping[str, object]


class ToolCatalog(Protocol):
    async def declare_group(
        self,
        ctx: Context,
        *,
        always_on: bool = False,
        description: str = "未声明用途",
    ) -> Effect: ...

    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, object],
        open: Callable[[Mapping[str, object]], AbstractAsyncContextManager[BoundTool]],
        capture: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
        public: bool = True,
        idempotent: bool = False,
        risk: Literal["read-only", "read-write", "external-side-effect"] = "read-write",
        search_hint: str | None = None,
    ) -> ToolRef: ...


TOOLS = ServiceKey[ToolCatalog]("tools.v1")


class ReceiptView(Protocol):
    status: Literal["delivered", "rejected", "failed"]
    provider_ids: tuple[str, ...]
    error: str | None


class SelectionView(Protocol):
    sinks: tuple[str, ...]


class DeliveryExecution(Protocol):
    def publish(
        self,
        writer: MessageWriter,
        message_id: str,
        body: Body,
        sinks: tuple[Mapping[str, object], ...],
        *,
        passive: bool = False,
    ) -> tuple[Message, SelectionView]: ...

    def selection(self, message_id: str) -> SelectionView | None: ...

    async def send(self, message_id: str, sink: str) -> ReceiptView: ...


class DeliveryAdmission(Protocol):
    def open(self, consumer: Context) -> DeliveryExecution: ...


DELIVERY = ServiceKey[DeliveryAdmission]("delivery.v1")


class SenderRegistry(Protocol):
    def bind_all(self, bindings: Bindings) -> Mapping[str, str]: ...


DELIVERY_SENDERS = ServiceKey[SenderRegistry]("delivery.senders.v1")


class ContentView(Protocol):
    @property
    def checks(self) -> Mapping[str, Callable[[ContentPart], ContentReferences]]: ...


class ContentProvider(Protocol):
    def bind(self) -> AbstractAsyncContextManager[ContentView]: ...


CONTENT = ServiceKey[ContentProvider]("content.v2")


class FinalOutputTurn(Protocol):
    source: str
    ending_message_id: str | None
    message_ids: tuple[str, ...]


class FinalOutputWaiter(Protocol):
    async def wait(self, reader: MessageReader, turn: FinalOutputTurn) -> None: ...


class FinalOutputDelivery(Protocol):
    def register(self, source: str, provider: FinalOutputWaiter) -> None: ...

    async def wait(self, reader: MessageReader, turn: FinalOutputTurn) -> None: ...


FINAL_OUTPUT_DELIVERY = ServiceKey[FinalOutputDelivery]("delivery.final_output.v1")


class ProjectedTurn(Protocol):
    source: str
    ending_message_id: str | None
    status: Literal["open", "complete", "quiet", "abandoned"]
    message_ids: tuple[str, ...]


class TurnProjection(Protocol):
    def project(self, messages: Sequence[Message], source: str) -> tuple[ProjectedTurn, ...]: ...


TURN_PROJECTION = ServiceKey[TurnProjection]("turn.projection.v1")

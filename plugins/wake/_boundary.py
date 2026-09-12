"""Wake 使用的外部能力窄边界。

这里仅声明 Wake 实际消费的结构和操作。ServiceKey 按稳定名称连接真实
owner；Delivery、Tools、Content 和兴趣服务的实现与内部模型不进入 Wake。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypedDict

from agent.plugin_composition import Context, Effect, EmitEventKey, ServiceKey
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.messages import MessageReader, MessageWriter
from agent.plugin_contracts import Body, CallRef, ContentPart, ContentReferences, Message


class SinkValue(TypedDict):
    """Wake 归档的发送目的地；Delivery owner 在入口重新校验。"""

    name: str
    binding_id: str
    address: str


class CallSource(Protocol):
    """工具 owner 提供的只读调用前缀。"""

    call_ref: CallRef
    messages: tuple[Message, ...]


ToolOutcome = Literal["success", "denied", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class ToolResultValue:
    """Wake 工具返回的结构值；tools owner 在执行边界归一化它。"""

    outcome: ToolOutcome
    parts: tuple[ContentPart, ...]


class ToolRef(Protocol):
    """Tools owner 返回的真实注册引用；Wake 不构造它。"""

    name: str
    description: Mapping[str, object]


class ToolView(Protocol):
    """Tools owner 根据真实引用创建的短期目录视图。"""

    refs: tuple[ToolRef, ...]

    def select(self, name: str) -> ToolRef: ...


class ToolCatalog(Protocol):
    """Wake 需要的注册、组合和 binding 入口。"""

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
        open: Callable[[Mapping[str, object]], AbstractAsyncContextManager[object]],
        public: bool = True,
        idempotent: bool = False,
    ) -> ToolRef: ...

    def view(self, *refs: ToolRef) -> ToolView: ...

    def bind(self, ref: ToolRef, bindings: Bindings) -> str: ...


TOOLS = ServiceKey[ToolCatalog]("tools.v1")
AKASHA_TOOLS = ServiceKey[ToolView]("akasha.tools.v1")
STANDARD_WEB_TOOLS = ServiceKey[ToolView]("standard-web.tools.v1")
WAKE_TOOLS_VIEW = ServiceKey[ToolView]("wake.tools.v1")


class ReceiptView(Protocol):
    status: Literal["delivered", "rejected", "failed"]


class SelectionView(Protocol):
    sinks: tuple[str, ...]


class DeliveryExecution(Protocol):
    def wait_idle(self, channel: str, address: str) -> Awaitable[None]: ...

    def selection(self, message_id: str) -> SelectionView | None: ...

    def receipt(self, message_id: str, sink: str) -> ReceiptView | None: ...

    def publish(
        self,
        writer: MessageWriter,
        message_id: str,
        body: Body,
        sinks: tuple[SinkValue, ...],
    ) -> tuple[Message, SelectionView]: ...

    def prepare(
        self,
        reader: MessageReader,
        message: Message,
        sinks: tuple[SinkValue, ...],
    ) -> SelectionView: ...

    def send(
        self,
        message_id: str,
        sink: str,
        *,
        before_start: Callable[[], str | None] | None = None,
    ) -> Awaitable[ReceiptView]: ...

    def cancel_prepared(self, message_id: str, sink: str, reason: str) -> Awaitable[bool]: ...


class DeliveryAdmission(Protocol):
    def open(self, consumer: Context) -> DeliveryExecution: ...


DELIVERY = ServiceKey[DeliveryAdmission]("delivery.v1")


class DeliveryHistoryEntry(Protocol):
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
    ) -> tuple[DeliveryHistoryEntry, ...]: ...

    def status(self, message_id: str, sink: str) -> Mapping[str, object] | None: ...


DELIVERY_READ = ServiceKey[DeliveryHistory]("delivery.read.v1")


class SenderRegistry(Protocol):
    def bind(self, name: str, bindings: Bindings) -> str: ...


DELIVERY_SENDERS = ServiceKey[SenderRegistry]("delivery.senders.v1")


class ContentCapability(Protocol):
    def check_text(self, part: ContentPart) -> ContentReferences: ...

    async def register(
        self,
        ctx: Context,
        definition: Mapping[str, object],
        *,
        prepare: Callable[[], object] | None = None,
    ) -> Effect: ...


CONTENT = ServiceKey[ContentCapability]("content.v2")


class SemanticInterest(Protocol):
    async def score(self, texts: Sequence[str], *, cutoff: str) -> Sequence[object]: ...


SEMANTIC_INTEREST = ServiceKey[SemanticInterest]("akasha.semantic-interest.v1")
DRIFT_CHANGED = EmitEventKey[None]("drift.changed")

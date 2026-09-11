"""投递能力的公开结构合同。

`delivery.final_output.v1` 是「按来源等待最终 Output 完成外部送达」的公开名字，
`delivery.v1` 是投递准入能力的公开名字。值模型、Sender Protocol 与注册表
Protocol 由合同层拥有；实现（`FinalOutputDelivery` 注册表、`DeliveryAdmission`）
留在 `plugins/delivery/`。

`FinalOutputWaiter`/`Sender` 是插件实现方可由第三方替换的接口，因此用
`@runtime_checkable` 描述；消费者只依赖合同。
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition.model import ServiceKey
from agent.plugin_contracts.message import Message

Text = Annotated[str, Field(min_length=1)]
Status = Literal["delivered", "rejected", "failed"]


class Sink(BaseModel):
    """发送 owner 固定的目的地；恢复不重新选择地址或 adapter。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: Text
    binding_id: Text
    address: Text


class Receipt(BaseModel):
    """一次外部送达的结算回执。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    status: Status
    provider_ids: tuple[Text, ...] = ()
    error: Text | None = None


@runtime_checkable
class Sender(Protocol):
    """一个具体渠道的发送适配器。"""

    @property
    def idempotent(self) -> bool:
        """同一 key 是否可安全重放。"""
        ...

    async def send(self, key: str, address: str, message: Message) -> Receipt:
        """发送并返回结算回执。"""
        ...

    async def query(self, key: str, address: str) -> Receipt | None:
        """只查询原效果；None 表示缺少可确认回执，不证明没有发送。"""
        ...


OpenSender = Callable[[str], AbstractAsyncContextManager[Sender]]


@runtime_checkable
class FinalOutputWaiter(Protocol):
    """等待一个已投影 Turn 的最终 Output 完成其外部送达。"""

    async def wait(self, reader: object, turn: object) -> None:
        """等待该 Turn 的最终 Output 送达完成。"""
        ...


@runtime_checkable
class FinalOutputDeliveryPort(Protocol):
    """按来源注册/等待最终 Output 送达能力。"""

    def register(self, source: str, provider: FinalOutputWaiter) -> None:
        """为某来源登记 provider；同一来源只有一个 owner。"""
        ...

    def unregister(self, source: str, provider: FinalOutputWaiter) -> None:
        """只移除仍由同一 provider 登记的所有权。"""
        ...

    async def wait(self, reader: object, turn: object) -> None:
        """等待该 Turn 的最终 Output。"""
        ...


@runtime_checkable
class DeliveriesPort(Protocol):
    """一个消费者的投递范围；发送记录与 Task 仍由投递能力独占。"""

    async def deliver(self, *args: object, **kwargs: object) -> object:
        """在已签发范围内执行一次投递。"""
        ...


@runtime_checkable
class DeliveryAdmissionPort(Protocol):
    """投递准入：按消费者签发恢复范围。"""

    def open(self, consumer: object) -> DeliveriesPort:
        """为一个消费者签发投递范围。"""
        ...


FINAL_OUTPUT_DELIVERY = ServiceKey[FinalOutputDeliveryPort]("delivery.final_output.v1")
DELIVERY = ServiceKey[DeliveryAdmissionPort]("delivery.v1")


@runtime_checkable
class DeliverySendersPort(Protocol):
    """普通渠道的出站注册表对 Core 与插件可见的方法子集。"""

    async def register(self, ctx: object, *, name: str, idempotent: bool, open: object) -> object:
        """登记一个发送 adapter；配置随真实 owner 归档。"""
        ...

    def registered_names(self) -> tuple[str, ...]:
        """只读当前可用名称。"""
        ...

    def bind(self, name: str, bindings: object) -> str:
        """固定一个名字到当前 binding。"""
        ...

    def bind_all(self, bindings: object) -> dict[str, str]:
        """固定当前全部可选发送者。"""
        ...


DELIVERY_SENDERS = ServiceKey[DeliverySendersPort]("delivery.senders.v1")

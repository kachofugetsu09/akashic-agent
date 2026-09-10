from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Literal, Protocol

from agent.plugin_composition import ServiceKey
from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_contracts.turn_projection import Turn
from session.log import MessageReader
from agent.plugin_contracts import Message

Text = Annotated[str, Field(min_length=1)]
Status = Literal["delivered", "rejected", "failed"]


class Sink(BaseModel):
    """发送 owner 固定的目的地；恢复不重新选择地址或 adapter。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: Text
    binding_id: Text
    address: Text


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    status: Status
    provider_ids: tuple[Text, ...] = ()
    error: Text | None = None


class Sender(Protocol):
    @property
    def idempotent(self) -> bool: ...

    async def send(self, key: str, address: str, message: Message) -> Receipt: ...

    async def query(self, key: str, address: str) -> Receipt | None:
        """只查询原效果；None 表示缺少可确认回执，不证明没有发送。"""
        ...


OpenSender = Callable[[str], AbstractAsyncContextManager[Sender]]


class FinalOutputWaiter(Protocol):
    """等待一个已投影 Turn 的最终 Output 完成其外部送达。"""

    async def wait(self, reader: MessageReader, turn: Turn) -> None: ...


class FinalOutputDelivery:
    """按来源选择最终 Output 的普通 delivery 能力。"""

    def __init__(self) -> None:
        self._providers: dict[str, FinalOutputWaiter] = {}

    def register(self, source: str, provider: FinalOutputWaiter) -> None:
        if not source or source in self._providers:
            raise ValueError("最终 Output provider 已有 owner")
        self._providers[source] = provider

    def unregister(self, source: str, provider: FinalOutputWaiter) -> None:
        """只移除仍由同一 optional child owner 登记的 provider。"""
        if self._providers.get(source) is provider:
            del self._providers[source]

    async def wait(self, reader: MessageReader, turn: Turn) -> None:
        provider = self._providers.get(turn.source)
        if provider is None:
            raise ValueError(f"没有来源 {turn.source!r} 的最终 Output provider")
        await provider.wait(reader, turn)


FINAL_OUTPUT_DELIVERY = ServiceKey[FinalOutputDelivery]("delivery.final_output.v1")

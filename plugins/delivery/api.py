from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition.messages import MessageReader
from agent.plugin_contracts.delivery import (
    FINAL_OUTPUT_DELIVERY as FINAL_OUTPUT_DELIVERY,
    FinalOutputTurn as FinalOutputTurn,
    FinalOutputWaiter as FinalOutputWaiter,
    Receipt as SenderResult,  # noqa: F401 - 显式再导出给本插件消费者。
    Sender as Sender,
)

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


OpenSender = Callable[[str], AbstractAsyncContextManager[Sender]]


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

    async def wait(self, reader: MessageReader, turn: FinalOutputTurn) -> None:
        provider = self._providers.get(turn.source)
        if provider is None:
            raise ValueError(f"没有来源 {turn.source!r} 的最终 Output provider")
        await provider.wait(reader, turn)

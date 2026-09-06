from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from session.message import Message

Text = Annotated[str, Field(min_length=1)]
Status = Literal["delivered", "rejected", "unknown"]


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

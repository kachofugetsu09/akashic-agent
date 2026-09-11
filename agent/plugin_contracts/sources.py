"""来源注册与渠道接纳的公开结构合同。

`source.v1`（`SOURCES`）是「按来源注册接纳与控制」的公开名字，`source.changed.v1`
（`SOURCE_CHANGED`）是「来源事实变化」的通知 key。`Source` 是值模型、
`SourcesPort` 是消费者可见的只读/注册接口；实现留在 `plugins/sources`。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey
from agent.plugin_contracts.conversation import Changed, ConversationPort
from agent.plugin_contracts.message import Message

if TYPE_CHECKING:
    from agent.plugin_composition.channels import ChannelInboundMessage
    from agent.plugin_composition.context import Context
    from agent.plugin_composition.effect import Effect

Accept = Callable[[str, str, "ChannelInboundMessage"], Awaitable[Message]]


@dataclass(frozen=True)
class Source:
    """一个来源的接纳入口与它独占的输入渠道。"""

    name: str
    open: Callable[[str], ConversationPort]
    accept: Accept | None = None
    channels: tuple[str, ...] | None = ()


@runtime_checkable
class SourcesPort(Protocol):
    """来源注册表对消费者可见的方法子集。"""

    async def register(self, ctx: Context, source: Source) -> Effect:
        """登记一个来源；同名或同渠道已有 owner 时失败。"""
        ...

    def entries(self) -> tuple[Source, ...]:
        """按注册顺序返回全部来源。"""
        ...


SOURCES = ServiceKey[SourcesPort]("sources.v1")
SOURCE_CHANGED = ServiceKey[Changed]("source.changed.v1")

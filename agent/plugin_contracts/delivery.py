"""投递发送者能力的公开结构合同。

`delivery.senders.v1` 是「普通渠道出站注册表」的公开名字：Core 与插件都需要
声明这个依赖，但都不应该 import 具体实现模块。合同层因此拥有 key 与 Protocol，
`plugins/delivery/senders.py` 提供满足该 Protocol 的实现。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey

if TYPE_CHECKING:
    from agent.plugin_composition.bindings import Bindings
    from agent.plugin_composition.context import Context
    from agent.plugin_composition.effect import Effect


@runtime_checkable
class DeliverySenders(Protocol):
    """出站注册表对 Core 与插件可见的方法子集。"""

    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        idempotent: bool,
        open: Callable[[], Awaitable[object]],
    ) -> Effect:
        """登记一个发送 adapter；配置随真实 owner 归档。"""
        ...

    def registered_names(self) -> tuple[str, ...]:
        """只读当前可用名称。"""
        ...

    def bind(self, name: str, bindings: Bindings) -> str:
        """固定一个名字到当前 binding。"""
        ...

    def bind_all(self, bindings: Bindings) -> Mapping[str, str]:
        """固定当前全部可选发送者。"""
        ...


DELIVERY_SENDERS = ServiceKey[DeliverySenders]("delivery.senders.v1")

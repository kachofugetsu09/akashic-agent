"""standard_tools 与材料 owner 之间的最小本地边界。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_contracts import Message


class MaterialRegistry(Protocol):
    """context owner 提供的材料注册入口；skills 不依赖其实现类。"""

    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        prepare: Callable[[tuple[Message, ...], str], Awaitable[Mapping[str, object]]],
        priority: int = 0,
        prompt: bool = False,
        reduce: object | None = None,
    ) -> object: ...


MATERIALS = ServiceKey[MaterialRegistry]("context.materials.v3")

"""tool_search 与 tools owner 之间的最小本地边界。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_contracts import ContentPart
from agent.plugin_contracts.tools import (
    TOOLS as TOOLS,
    CallSource as CallSource,
    ToolCatalog as ToolCatalog,
    ToolPresentation as ToolPresentation,
    ToolRef as ToolRef,
    ToolView as ToolView,
)

ToolOutcome = Literal["success", "denied", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class ToolResultValue:
    """tool_search 的本地结果；tools owner 在执行边界重新校验。"""

    outcome: ToolOutcome
    parts: tuple[ContentPart, ...]


class BoundTool(Protocol):
    """tools owner 打开的真实工具；搜索插件只消费其最小方法集。"""

    @property
    def idempotent(self) -> bool: ...

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object] | str: ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue: ...

    async def query(self, key: str) -> ToolResultValue | None: ...


__all__ = [
    "BoundTool",
    "CallSource",
    "ToolCatalog",
    "ToolPresentation",
    "ToolRef",
    "ToolResultValue",
    "ToolView",
    "TOOLS",
]

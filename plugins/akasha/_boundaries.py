"""Akasha 只依赖外部插件发布的窄能力和结构值。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from agent.plugin_contracts import ContentPart, Message
from agent.plugin_contracts.content import (
    CONTENT as CONTENT,
    Content as ContentCapability,  # noqa: F401 - 显式再导出给本插件消费者。
)
from agent.plugin_contracts.tools import (
    TOOLS as TOOLS,
    CallSource as CallSource,
    ToolCatalog as ToolCatalog,
    ToolRef as ToolRef,
    ToolView as ToolView,
)
from agent.plugin_contracts.turns import (
    TURN_PROJECTION as TURN_PROJECTION,
    Turn as Turn,
    TurnProjection as TurnProjection,
)

Outcome = Literal["success", "denied", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class Result:
    """工具 provider 可消费的结构结果。"""

    outcome: Outcome
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "denied", "error", "interrupted"}:
            raise ValueError("工具结果状态无效")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("工具结果必须是内容块")
        object.__setattr__(self, "parts", parts)


PostCommitReader = Callable[[Message], str | None]

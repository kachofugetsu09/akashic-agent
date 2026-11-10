"""Akasha 只依赖外部插件发布的窄能力和结构值。"""
from __future__ import annotations

from collections.abc import Callable

from plugins.ledger.contract import Message
from plugins.content.contract import (
    CONTENT as CONTENT,
    Content as ContentCapability,  # noqa: F401 - 显式再导出给本插件消费者。
)
from plugins.tools.contract import (
    TOOLS as TOOLS,
    CallSource as CallSource,
    ToolCatalog as ToolCatalog,
    Outcome as Outcome,
    Result as Result,
    ToolRef as ToolRef,
    ToolView as ToolView,
)
from plugins.turn_projection.contract import (
    TURN_PROJECTION as TURN_PROJECTION,
    Turn as Turn,
    TurnProjection as TurnProjection,
)

PostCommitReader = Callable[[Message], str | None]

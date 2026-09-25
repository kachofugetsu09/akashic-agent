from agent.plugin_contracts.tools import (
    TOOLS as TOOLS,
    CallSource as CallSource,
    ToolCatalog as ToolCatalog,
)
from agent.plugin_contracts.turns import (
    TURN_PROJECTION as TURN_PROJECTION,
    Turn as Turn,
    TurnProjection as TurnProjection,
)

"""Computer 只读取工具调用前缀和 Turn 身份。"""
from dataclasses import dataclass
from typing import Literal

from agent.plugin_contracts import ContentPart


@dataclass(frozen=True)
class Result:
    outcome: Literal["success", "error"]
    parts: tuple[ContentPart, ...]

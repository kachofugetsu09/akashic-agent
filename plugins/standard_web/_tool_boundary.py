"""standard_web 与 tools owner 之间的最小本地边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.plugin_contracts import ContentPart
from agent.plugin_contracts.tools import (
    TOOLS as TOOLS,
    CallSource as CallSource,
    ToolCatalog as ToolCatalog,
    ToolRef as ToolRef,
    ToolView as ToolView,
)

ToolOutcome = Literal["success", "denied", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class ToolResultValue:
    """standard_web 的本地结果；tools owner 在执行边界重新校验。"""

    outcome: ToolOutcome
    parts: tuple[ContentPart, ...]

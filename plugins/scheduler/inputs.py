"""调度来源消费的内容、工具与投递能力。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.plugin_contracts import ContentPart
from agent.plugin_contracts.content import (
    CONTENT as CONTENT,
)
from agent.plugin_contracts.delivery import (
    DELIVERY as DELIVERY,
    DELIVERY_SENDERS as DELIVERY_SENDERS,
    Deliveries as Deliveries,
    Delivery as Delivery,
    Receipt as Receipt,
    Selection as Selection,
    Senders as Senders,
)
from agent.plugin_contracts.tools import (
    ALL_TOOLS as ALL_TOOLS,
    TOOLS as TOOLS,
    ToolCatalog as ToolCatalog,
    ToolRef as ToolRef,
    ToolView as ToolView,
)


@dataclass(frozen=True)
class Result:
    outcome: Literal["success", "denied", "error", "interrupted"]
    parts: tuple[ContentPart, ...]

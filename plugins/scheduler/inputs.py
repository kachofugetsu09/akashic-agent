"""调度来源消费的内容、工具与投递能力。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.plugin_contracts import ContentPart
from plugins.content.contract import (
    CONTENT as CONTENT,
)
from plugins.delivery.contract import (
    DELIVERY_GUARDED_START as DELIVERY,
    DELIVERY_SENDERS as DELIVERY_SENDERS,
    GuardedDeliveries as Deliveries,
    GuardedDelivery as Delivery,
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

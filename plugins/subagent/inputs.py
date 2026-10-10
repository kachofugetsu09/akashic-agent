"""subagent 消费的内容、工具与投递能力。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.plugin_contracts import (
    ContentPart,
)
from agent.plugin_contracts.content import (
    CONTENT as CONTENT,
)
from agent.plugin_contracts.delivery import (
    DELIVERY_GUARDED_START as DELIVERY,
    DELIVERY_SENDERS as DELIVERY_SENDERS,
    GuardedDeliveries as Deliveries,
    GuardedDelivery as Delivery,
    Receipt as Receipt,
    Selection as Selection,
    Senders as Senders,
)
from plugins.reply.contract import (
    REPLY_PROGRAM_V3 as REPLY_PROGRAM,
)
from agent.plugin_contracts.sources import (
    CHECK_ORIGIN as CHECK_ORIGIN,
    CONVERSATION_COMPLETE_V2 as CONVERSATION_COMPLETE,
    ConversationCompleteV2 as ConversationComplete,
)
from agent.plugin_contracts.tools import (
    ALL_TOOLS as ALL_TOOLS,
    TOOL_BIND_SAVED as TOOL_BIND_SAVED,
    TOOLS as TOOLS,
    BindSavedTool as BindSavedTool,
    CallSource as CallSource,
    ToolCatalog as ToolCatalog,
    ToolRef as ToolRef,
    ToolView as ToolView,
)


@dataclass(frozen=True)
class Result:
    outcome: Literal["success", "denied", "error", "interrupted"]
    parts: tuple[ContentPart, ...]

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Protocol, cast

from agent.plugin_composition.models import BoundChatModel, ModelRequest
from agent.plugin_contracts.content import Reference
from agent.plugin_contracts import CallRef, ContentPart, ContentReferences, Control, Message, Output, ToolCall, ToolResult



# 预置上下文词汇表与 Protocol 的拥有者已移到结构合同层；这里保留再导出。
from agent.plugin_contracts.context import (  # noqa: E402,F401
    ContextModel,
    ContextOverflow,
    Materials,
    Reminder,
    Summary,
    SummaryReducer,
    check_summary,
    settled_prefixes,
    summary_range,
)

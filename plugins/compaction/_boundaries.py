"""Compaction 与其它插件之间只交换已校验的结构值。"""
from __future__ import annotations

from collections.abc import Mapping

from plugins.compaction.contract import (
    COMPACTION_READER as COMPACTION_READER,
    COMPACTION_SUMMARIES as COMPACTION_SUMMARIES,
    CompactionReader as CompactionReader,
    StoredSummary as StoredSummary,
    SummaryLookup as SummaryLookup,
)
from plugins.context.contract import (
    CONTEXT as CONTEXT,
    MATERIALS_V4 as MATERIALS,
    ContextBuilder as ContextBuilder,
)
from agent.plugin_contracts.models import ContextModel as ContextModel
from plugins.turn_projection.contract import (
    TURN_PROJECTION as TURN_PROJECTION,
    Turn as Turn,
    TurnProjection as TurnProjection,
)

MaterialData = Mapping[str, object]
SummaryData = Mapping[str, object]

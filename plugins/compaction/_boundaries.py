"""Compaction 与其它插件之间只交换已校验的结构值。"""
from __future__ import annotations

from collections.abc import Mapping

from agent.plugin_contracts.compaction import (
    COMPACTION_READER as COMPACTION_READER,
    COMPACTION_SUMMARIES as COMPACTION_SUMMARIES,
    CompactionReader as CompactionReader,
    StoredSummary as StoredSummary,
    SummaryLookup as SummaryLookup,
)
from agent.plugin_contracts.context import (
    CONTEXT as CONTEXT,
    MATERIALS as MATERIALS,
    ContextBuilder as ContextBuilder,
)
from agent.plugin_contracts.models import ContextModel as ContextModel
from agent.plugin_contracts.turns import (
    TURN_PROJECTION as TURN_PROJECTION,
    Turn as Turn,
    TurnProjection as TurnProjection,
)

MaterialData = Mapping[str, object]
SummaryData = Mapping[str, object]

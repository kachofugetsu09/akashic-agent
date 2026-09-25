"""Markdown memory 消费的外部能力边界。"""
from __future__ import annotations

from agent.plugin_contracts.compaction import (
    COMPACTION_READER as COMPACTION_READER,
    COMPACTION_SUMMARIES as COMPACTION_SUMMARIES,
    CompactionReader as CompactionReader,
    PartitionedSummary as PartitionedSummary,
    StoredSummary as StoredSummary,
    SummaryLookup as SummaryLookup,
)
from agent.plugin_contracts.content import (
    CONTENT as CONTENT,
    Content as ContentFacts,  # noqa: F401 - 显式再导出给本插件消费者。
)
from agent.plugin_contracts.context import (
    CONTEXT as CONTEXT,
    MATERIALS as MATERIALS,
    ContextBuilder as ContextBuilder,
)
from agent.plugin_contracts.turns import (
    TURN_PROJECTION as TURN_PROJECTION,
    Turn as Turn,
    TurnProjection as TurnProjection,
)

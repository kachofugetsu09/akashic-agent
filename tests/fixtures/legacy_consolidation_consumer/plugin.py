from __future__ import annotations

from agent.plugin_composition import Context
from plugins.markdown_memory.plugin import (
    LEGACY_CONSOLIDATION_COMMITTED,
    LegacyConsolidationCommitted,
)

api_version = 3
name = "legacy_consolidation_consumer"
version = "1.0.0"
inject = ()

observed: list[LegacyConsolidationCommitted] = []
fail = False


def _consume(event: LegacyConsolidationCommitted) -> None:
    if fail:
        raise RuntimeError("legacy consumer write failed")
    observed.append(event)


async def apply(ctx: Context, config: object) -> None:
    _ = config
    observed.clear()
    _ = await ctx.on(LEGACY_CONSOLIDATION_COMMITTED, _consume)

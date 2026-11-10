"""Optional user-information adapter; no Alert/Content consumption capability."""
from collections.abc import Mapping
from datetime import UTC, datetime
from plugins.userinformation.contract import USER_INFORMATION
from .contract import EVENTMAIL_CONTEXT_READ


class ContextInformation:
    def __init__(self, reader):
        self.reader = reader

    async def describe(self):
        return {"queries": {"current": {"description": "有效的当前来源状态（包括睡眠、Steam）；缺失不等于否定。source_ids 可筛选来源。",
            "parameters": {"type": "object", "properties": {"source_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 32}}, "additionalProperties": False}}}}

    async def query(self, query: str, arguments: Mapping[str, object]):
        sources = arguments.get("source_ids")
        if query != "current" or set(arguments) - {"source_ids"} or (sources is not None and
            (not isinstance(sources, (list, tuple)) or len(sources) > 32 or any(not isinstance(s, str) for s in sources))):
            return {"status": "invalid_query", "error": "Use current with optional source_ids string array"}
        now = datetime.now(UTC)
        rows = await self.reader.active_context(now)
        items = [dict(row) for row in rows if sources is None or row['source_id'] in sources]
        present = {row['source_id'] for row in items}
        return {"as_of": now.isoformat(), "items": items,
                "missing_sources": [] if sources is None else [s for s in sources if s not in present],
                "semantics": "Latest unexpired observations, not live measurements. Missing can mean expired, unknown or unavailable."}


async def contribute(ctx):
    await ctx.require(USER_INFORMATION).register(ctx, ContextInformation(ctx.require(EVENTMAIL_CONTEXT_READ)))

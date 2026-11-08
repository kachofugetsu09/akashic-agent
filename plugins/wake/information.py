"""Read configured information once before persisting a Wake request."""
import asyncio
import json
from datetime import UTC, datetime
from agent.plugin_composition import Context
from plugins.ledger.contract import json_value
from plugins.userinformation.contract import USER_INFORMATION_READ
from .api import InformationQuery


async def snapshot_information(ctx: Context, queries: tuple[InformationQuery, ...]) -> tuple[dict[str, object], ...]:
    reader = ctx.require(USER_INFORMATION_READ)

    async def read(query: InformationQuery) -> dict[str, object]:
        evidence: dict[str, object] = {"request": query.model_dump(mode="json"),
                                       "queried_at": datetime.now(UTC).isoformat()}
        try:
            async with asyncio.timeout(5):
                result = json_value(await reader.query(query.provider, query.query, query.arguments))
                if len(json.dumps(result, ensure_ascii=False).encode()) > 8192:
                    return {**evidence, "status": "unavailable", "reason": "size_limit"}
                return {**evidence, "response": result}
        except Exception:
            # Read failures are evidence, not an admission failure. Cancellation still propagates.
            return {**evidence, "status": "unavailable", "reason": "query_failed_or_timed_out"}

    return tuple(await asyncio.gather(*(read(query) for query in queries)))

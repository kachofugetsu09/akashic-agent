from __future__ import annotations

from typing import cast

from plugins.default_proactive.gateway import GatewayDeps
from proactive_v2 import mcp_sources
from proactive_v2.mcp_sources import McpGateway


class McpGatewaySource:
    def __init__(self, pool: McpGateway, *, content_limit: int) -> None:
        self._pool = pool
        self._content_limit = content_limit

    def build_gateway_deps(
        self,
        *,
        web_fetch_tool: object | None,
        max_chars: int,
    ) -> GatewayDeps:
        return GatewayDeps(
            alert_fn=self.alert_fn,
            feed_fn=self.feed_fn,
            context_fn=self.context_fn,
            web_fetch_tool=web_fetch_tool,
            max_chars=max_chars,
            content_limit=self._content_limit,
        )

    async def alert_fn(self) -> list[dict[str, object]]:
        return cast(
            list[dict[str, object]],
            await mcp_sources.fetch_alert_events_async(self._pool),
        )

    async def feed_fn(self, limit: int = 5) -> list[dict[str, object]]:
        events = await mcp_sources.fetch_content_events_async(self._pool)
        return cast(list[dict[str, object]], events[:limit])

    async def context_fn(self) -> list[dict[str, object]]:
        rows = await mcp_sources.fetch_context_data_async(self._pool)
        if not isinstance(rows, list):
            return []
        return cast(list[dict[str, object]], rows)

    async def ack_fn(self, compound_key: str, feedback: str) -> None:
        parts = compound_key.split(":", 1)
        if len(parts) != 2:
            return
        ack_server, item_id = parts
        await mcp_sources.acknowledge_content_entries_async(
            self._pool,
            [(f"mcp:{ack_server}", item_id)],
            feedback=feedback,
        )

    async def alert_ack_fn(self, compound_key: str) -> None:
        parts = compound_key.split(":", 1)
        if len(parts) != 2:
            return
        await mcp_sources.acknowledge_events_async(self._pool, [(parts[0], parts[1])])

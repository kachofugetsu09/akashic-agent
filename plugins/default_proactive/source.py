from __future__ import annotations

import asyncio
from typing import cast

from agent.plugins.specs import RegisteredProactiveSource
from plugins.default_proactive.gateway import GatewayDeps
from proactive_v2 import mcp_sources
from proactive_v2.mcp_sources import McpGateway


class McpGatewaySource:
    def __init__(
        self,
        pool: McpGateway,
        sources: list[RegisteredProactiveSource],
        *,
        content_limit: int,
    ) -> None:
        self._pool = pool
        self._sources = sources
        self._content_limit = content_limit
        self._snapshot_task: asyncio.Task[dict[str, list[dict]]] | None = None

    def build_gateway_deps(
        self,
        *,
        web_fetch_tool: object | None,
        max_chars: int,
    ) -> GatewayDeps:
        return GatewayDeps(
            begin_fn=self.begin_tick,
            alert_fn=self.alert_fn,
            feed_fn=self.feed_fn,
            context_fn=self.context_fn,
            web_fetch_tool=web_fetch_tool,
            max_chars=max_chars,
            content_limit=self._content_limit,
        )

    def begin_tick(self) -> None:
        self._snapshot_task = None

    async def _snapshot(self) -> dict[str, list[dict]]:
        if self._snapshot_task is None:
            self._snapshot_task = asyncio.create_task(
                mcp_sources.fetch_sources_async(self._pool, self._sources)
            )
        return await self._snapshot_task

    async def alert_fn(self) -> list[dict[str, object]]:
        return cast(list[dict[str, object]], (await self._snapshot())["alert"])

    async def feed_fn(self, limit: int = 5) -> list[dict[str, object]]:
        events = (await self._snapshot())["content"]
        return cast(list[dict[str, object]], events[:limit])

    async def context_fn(self) -> list[dict[str, object]]:
        return cast(list[dict[str, object]], (await self._snapshot())["context"])

    async def ack_fn(self, compound_key: str, feedback: str) -> None:
        parsed = self._parse_compound_key(compound_key)
        if parsed is None:
            return
        source_id, item_id = parsed
        await mcp_sources.acknowledge_async(
            self._pool,
            self._sources,
            source_id,
            [item_id],
            feedback=feedback,
        )

    async def alert_ack_fn(self, compound_key: str) -> None:
        parsed = self._parse_compound_key(compound_key)
        if parsed is None:
            return
        source_id, item_id = parsed
        await mcp_sources.acknowledge_async(
            self._pool,
            self._sources,
            source_id,
            [item_id],
        )

    def _parse_compound_key(self, compound_key: str) -> tuple[str, str] | None:
        keys = sorted(
            (mcp_sources.source_key(source) for source in self._sources),
            key=len,
            reverse=True,
        )
        for source_id in keys:
            prefix = f"{source_id}:"
            if compound_key.startswith(prefix):
                return source_id, compound_key[len(prefix):]
        return None

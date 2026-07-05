from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, cast

from proactive_v2 import mcp_sources
from proactive_v2.config import ProactiveConfig
from proactive_v2.frame import ProactiveFrame
from proactive_v2.gateway import GatewayDeps
from proactive_v2.mcp_sources import McpClientPool

logger = logging.getLogger(__name__)


class McpRuntimeModule:
    slot = "proactive.source.mcp_runtime"
    phase = "proactive.source"

    def __init__(
        self,
        *,
        workspace: Path | None,
        cfg: ProactiveConfig,
        extra_server_configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._cfg = cfg
        self._pool = McpClientPool(
            workspace,
            extra_server_configs=extra_server_configs,
        )
        self._poll_lock = asyncio.Lock()
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None

    @property
    def pool(self) -> McpClientPool:
        return self._pool

    async def start(self) -> None:
        self._running = True
        await self._pool.connect_all()
        await self._poll_once()
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        await self._pool.disconnect_all()
        logger.info("[proactive] mcp pool 已关闭")

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        return frame

    async def _poll_once(self) -> None:
        if self._poll_lock.locked():
            logger.debug("[proactive] feed poll 仍在进行,跳过本次")
            return
        async with self._poll_lock:
            try:
                await mcp_sources.poll_content_feeds_async(self._pool)
                logger.info("[proactive] feed poll 完成")
            except Exception as e:
                logger.warning("[proactive] feed poll 系统级失败: %s", e)

    async def _poll_loop(self) -> None:
        while self._running:
            interval = max(
                1,
                int(self._cfg.feed_poller_interval_seconds),
            )
            await asyncio.sleep(interval)
            if not self._running:
                break
            await self._poll_once()


class McpGatewaySource:
    def __init__(
        self,
        pool: McpClientPool,
        *,
        content_limit: int,
    ) -> None:
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

    async def ack_fn(self, compound_key: str, ttl_hours: int) -> None:
        parts = compound_key.split(":", 1)
        if len(parts) != 2:
            return
        ack_server, item_id = parts
        source_key = f"mcp:{ack_server}"
        await mcp_sources.acknowledge_content_entries_async(
            self._pool,
            [(source_key, item_id)],
            ttl_hours=ttl_hours,
        )

    async def alert_ack_fn(self, compound_key: str) -> None:
        parts = compound_key.split(":", 1)
        if len(parts) != 2:
            return
        ack_server, ack_id = parts
        await mcp_sources.acknowledge_events_async(self._pool, [(ack_server, ack_id)])

from __future__ import annotations

import asyncio
import logging
from proactive_v2 import mcp_sources
from proactive_v2.config import ProactiveConfig
from proactive_v2.frame import ProactiveFrame
from proactive_v2.mcp_sources import McpGateway

logger = logging.getLogger(__name__)


class McpRuntimeModule:
    slot = "proactive.source.mcp_runtime"

    def __init__(
        self,
        *,
        cfg: ProactiveConfig,
        gateway: McpGateway,
    ) -> None:
        self._cfg = cfg
        self._pool = gateway
        self._poll_lock = asyncio.Lock()
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None

    @property
    def pool(self) -> McpGateway:
        return self._pool

    async def start(self) -> None:
        self._running = True
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
        logger.info("[proactive] source poll 已关闭")

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

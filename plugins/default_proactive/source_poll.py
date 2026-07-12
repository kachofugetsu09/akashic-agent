from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from agent.plugins.specs import RegisteredProactiveSource
from proactive_v2 import mcp_sources
from proactive_v2.frame import ProactiveFrame
from proactive_v2.mcp_sources import McpGateway

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshotLease

logger = logging.getLogger(__name__)


class DefaultSourcePollModule:
    slot = "default.source.poll"

    def __init__(
        self,
        *,
        gateway: McpGateway,
        sources: list[RegisteredProactiveSource],
        runtime_snapshot_lease: RuntimeSnapshotLease | None = None,
    ) -> None:
        self._gateway = gateway
        self._sources = [source for source in sources if source.spec.poll_tool]
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._runtime_snapshot_lease = runtime_snapshot_lease

    async def start(self) -> None:
        """首次拉取全部 pull source，并启动各自的后台轮询。"""

        # 1. 先完成首次拉取，保证随后运行的默认流程能看到最新数据。
        self._running = True
        try:
            for source in self._sources:
                await self._poll_once(source)
                # 2. 每个 source 独立计时，避免慢源阻塞其他来源。
                ready = asyncio.Event()
                self._tasks.append(
                    asyncio.create_task(
                        self._poll_loop(source, ready),
                        name=f"default_proactive_poll:{mcp_sources.source_key(source)}",
                    )
                )
                _ = await ready.wait()
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            _ = task.cancel()
        if self._tasks:
            _ = await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        return frame

    async def _poll_once(self, source: RegisteredProactiveSource) -> None:
        key = mcp_sources.source_key(source)
        try:
            await mcp_sources.poll_source_async(self._gateway, source)
            logger.info("[default-proactive] source poll 完成: %s", key)
        except Exception as exc:
            logger.warning("[default-proactive] source poll 失败 %s: %s", key, exc)

    async def _poll_loop(
        self,
        source: RegisteredProactiveSource,
        ready: asyncio.Event,
    ) -> None:
        """在正确的插件快照作用域中持续轮询一个 source。"""

        # 1. 普通启动不需要绑定快照。
        source_lease = self._runtime_snapshot_lease
        if source_lease is None:
            ready.set()
            await self._poll_loop_bound(source)
            return

        # 2. 热重载模式让后台任务持有自己的快照租约。
        lease = source_lease.fork()
        from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

        async with lease:
            token = bind_runtime_snapshot(lease)
            ready.set()
            try:
                await self._poll_loop_bound(source)
            finally:
                reset_runtime_snapshot(token)

    async def _poll_loop_bound(self, source: RegisteredProactiveSource) -> None:
        interval = max(1, int(source.spec.poll_interval_seconds))
        while self._running:
            await asyncio.sleep(interval)
            if self._running:
                await self._poll_once(source)

"""
proactive/feed_poller.py — 后台 Feed 轮询器。

独立 asyncio 任务，以固定间隔从 FeedRegistry 拉取并写入 FeedBuffer。
ProactiveEngine 通过 SensePort.fetch_items() 从 FeedBuffer 读取，
不再在 tick 内直接拉取，解耦决策频率与 feed 拉取频率。

启动预热：run() 开始时立即执行一次 _poll_once()，
确保冷启动第一轮 tick 不会读到空 buffer。
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from feeds.registry import FeedRegistry
from feeds.buffer import FeedBuffer

if TYPE_CHECKING:
    from proactive.loop import ProactiveConfig

logger = logging.getLogger(__name__)


class FeedPoller:
    """定期从 FeedRegistry 拉取 feed 并写入 FeedBuffer 的后台循环。"""

    def __init__(
        self,
        registry: FeedRegistry,
        buffer: FeedBuffer,
        cfg: "ProactiveConfig",
    ) -> None:
        self._registry = registry
        self._buffer = buffer
        self._cfg = cfg
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info(
            "[feed_poller] 启动 interval=%ds fetch_limit=%d buffer_ttl=%dh max_per_source=%d",
            self._cfg.feed_poller_interval_seconds,
            self._cfg.feed_poller_fetch_limit,
            self._cfg.feed_poller_buffer_ttl_hours,
            self._cfg.feed_poller_buffer_max_per_source,
        )
        # 启动预热：立即拉取一次，避免冷启动第一轮 tick 读空 buffer
        await self._poll_once()
        while self._running:
            await asyncio.sleep(self._cfg.feed_poller_interval_seconds)
            if not self._running:
                break
            await self._poll_once()

    async def _poll_once(self) -> int:
        """执行一次拉取：先淘汰过期条目，再写入新条目，返回新增数量。

        顺序关键：先 evict 再 add，防止过期同 item_id 条目挡住本轮写入：
        若先 add：过期旧条目 iid 仍在 _entries → add 跳过 → evict 删旧 → buffer 丢失该条目。
        """
        try:
            items = await self._registry.fetch_all(self._cfg.feed_poller_fetch_limit)
            evicted = self._buffer.evict_expired()
            added = self._buffer.add(items)
            logger.info(
                "[feed_poller] poll 完成 fetched=%d added=%d evicted=%d buffer_total=%d stats=%s",
                len(items),
                added,
                evicted,
                self._buffer.size(),
                self._buffer.stats(),
            )
            return added
        except Exception:
            logger.exception("[feed_poller] _poll_once 失败")
            return 0

    def stop(self) -> None:
        self._running = False

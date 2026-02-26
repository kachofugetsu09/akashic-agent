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
from typing import TYPE_CHECKING, Any

from feeds.registry import FeedRegistry
from feeds.buffer import FeedBuffer

if TYPE_CHECKING:
    from proactive.loop import ProactiveConfig
    from proactive.source_scorer import SourceScorer
    from feeds.store import FeedStore

logger = logging.getLogger(__name__)


class FeedPoller:
    """定期从 FeedRegistry 拉取 feed 并写入 FeedBuffer 的后台循环。"""

    def __init__(
        self,
        registry: FeedRegistry,
        buffer: FeedBuffer,
        cfg: "ProactiveConfig",
        source_scorer: "SourceScorer | None" = None,
        feed_store: "FeedStore | None" = None,
        memory_provider: Any | None = None,
    ) -> None:
        self._registry = registry
        self._buffer = buffer
        self._cfg = cfg
        self._scorer = source_scorer
        self._feed_store = feed_store
        self._memory_provider = memory_provider  # MemoryStore，用于读取 memory_text
        self._running = False

    async def run(self) -> None:
        self._running = True
        scorer_status = "enabled" if self._scorer is not None else "disabled"
        logger.info(
            "[feed_poller] 启动 interval=%ds fetch_limit=%d buffer_ttl=%dh max_per_source=%d source_scorer=%s",
            self._cfg.feed_poller_interval_seconds,
            self._cfg.feed_poller_fetch_limit,
            self._cfg.feed_poller_buffer_ttl_hours,
            self._cfg.feed_poller_buffer_max_per_source,
            scorer_status,
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
            per_source_limits = await self._get_per_source_limits()
            items = await self._registry.fetch_all(
                self._cfg.feed_poller_fetch_limit,
                per_source_limits=per_source_limits,
            )
            evicted = self._buffer.evict_expired()
            added = self._buffer.add(items)
            logger.debug(
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

    async def _get_per_source_limits(self) -> dict[str, int] | None:
        """向 SourceScorer 获取本轮的每源配额，失败时返回 None（均等分配降级）。"""
        if self._scorer is None or self._feed_store is None:
            return None
        try:
            subs = self._feed_store.list_enabled()
            memory_text = ""
            if self._memory_provider is not None:
                try:
                    memory_text = self._memory_provider.read_long_term().strip()
                except Exception:
                    pass
            limits = await self._scorer.get_limits(
                subscriptions=subs,
                memory_text=memory_text,
                total_budget=getattr(self._cfg, "source_scorer_total_budget", 60),
                min_per_source=getattr(self._cfg, "source_scorer_min_per_source", 2),
                max_per_source=getattr(self._cfg, "source_scorer_max_per_source", 20),
            )
            logger.debug(
                "[feed_poller] source_scorer 配额: %s",
                {k[:8]: v for k, v in limits.items()},
            )
            return limits
        except Exception as e:
            logger.warning("[feed_poller] source_scorer 失败，降级均等分配: %s", e)
            return None

    def stop(self) -> None:
        self._running = False

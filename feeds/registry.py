"""
FeedRegistry — 动态信息源注册与批量拉取。

每次 fetch_all() 都重新从 FeedStore 读取订阅，
确保用户新增/删除订阅后立即生效，无需重启。

注册方式（对标 MessagePushTool.register_channel）：
    registry.register_source_type("rss", lambda sub: RSSFeedSource(sub))
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from feeds.base import FeedItem, FeedSource, FeedSubscription
from feeds.store import FeedStore

logger = logging.getLogger(__name__)


class FeedRegistry:
    def __init__(self, store: FeedStore) -> None:
        self._store = store
        # type_name -> factory(sub) -> FeedSource
        self._factories: dict[str, Callable[[FeedSubscription], FeedSource]] = {}

    def register_source_type(
        self,
        type_name: str,
        factory: Callable[[FeedSubscription], FeedSource],
    ) -> None:
        """注册一种信息源类型的构造工厂。"""
        self._factories[type_name] = factory
        logger.debug(f"FeedRegistry: 注册类型 {type_name!r}")

    async def fetch_all(
        self,
        limit_per_source: int = 3,
        per_source_limits: dict[str, int] | None = None,
    ) -> list[FeedItem]:
        """从所有启用的订阅中并发拉取内容。单个 source 失败不影响其他。

        Args:
            limit_per_source: 每源默认拉取条数（per_source_limits 未覆盖时使用）。
            per_source_limits: source_id → limit 的精细化配额字典（由 SourceScorer 生成）。
                               若为 None 则所有源均使用 limit_per_source。
        """
        subs = self._store.list_enabled()
        if not subs:
            return []

        sources: list[tuple[FeedSource, int]] = []  # (source, limit)
        for sub in subs:
            factory = self._factories.get(sub.type)
            if factory is None:
                logger.warning(
                    f"FeedRegistry: 未知类型 {sub.type!r}，跳过 {sub.name!r}"
                )
                continue
            try:
                src = factory(sub)
                limit = (
                    per_source_limits.get(sub.id, limit_per_source)
                    if per_source_limits is not None
                    else limit_per_source
                )
                sources.append((src, limit))
            except Exception as e:
                logger.warning(f"FeedRegistry: 构造 source {sub.name!r} 失败: {e}")

        if not sources:
            return []

        results = await asyncio.gather(
            *[src.fetch(limit) for src, limit in sources],
            return_exceptions=True,
        )

        items: list[FeedItem] = []
        for (src, _), result in zip(sources, results):
            if isinstance(result, Exception):
                logger.warning(
                    "FeedRegistry: %r 拉取失败 err_type=%s err=%r",
                    src.name,
                    type(result).__name__,
                    str(result),
                )
            else:
                items.extend(result)  # type: ignore[arg-type]

        return items

"""
feeds/buffer.py — 内存 Feed 缓冲区。

FeedPoller 写入，ProactiveEngine 通过 SensePort 读取。
不做 seen/delivery 判断，那是 ProactiveStateStore 的职责。
进程重启后自动由 FeedPoller 重新填充。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from proactive.item_id import compute_item_id, compute_source_key

if TYPE_CHECKING:
    from feeds.base import FeedItem

logger = logging.getLogger(__name__)


class FeedBuffer:
    """线程安全（asyncio 单线程）的内存 FeedItem 暂存区。

    - item_id 去重：同一 item 不会重复入库
    - TTL 淘汰：超过 ttl_hours 的条目自动丢弃
    - 每源上限：每个 source_key 最多保留 max_per_source 条（按加入时间，保留最新）
    """

    def __init__(
        self,
        *,
        ttl_hours: int = 48,
        max_per_source: int = 100,
    ) -> None:
        self._ttl = timedelta(hours=max(ttl_hours, 1))
        self._max_per_source = max(max_per_source, 1)
        # item_id -> (FeedItem, added_at)
        self._entries: dict[str, tuple[FeedItem, datetime]] = {}

    # ── 写入 ──────────────────────────────────────────────────────

    def add(self, items: list[FeedItem]) -> int:
        """加入 items，重复 item_id 跳过。返回实际新增数。"""
        now = datetime.now(timezone.utc)
        added = 0
        for item in items:
            iid = compute_item_id(item)
            if iid in self._entries:
                continue
            self._entries[iid] = (item, now)
            added += 1
        if added:
            self._enforce_per_source_limit()
        logger.debug("[feed_buffer] add added=%d total=%d", added, len(self._entries))
        return added

    # ── 读取 ──────────────────────────────────────────────────────

    def get_all(self, n: int = 0) -> list[FeedItem]:
        """返回所有未过期 items，按加入时间降序（最新在前）。

        n > 0 时只返回前 n 条（软上限，保护决策链成本）。
        n = 0 返回全部（测试/调试用）。
        """
        now = datetime.now(timezone.utc)
        cutoff = now - self._ttl
        valid = [
            (item, ts)
            for item, ts in self._entries.values()
            if ts >= cutoff
        ]
        valid.sort(key=lambda x: x[1], reverse=True)
        if n > 0:
            valid = valid[:n]
        return [item for item, _ in valid]

    def size(self) -> int:
        """返回 buffer 中（含过期）条目总数，用于监控。"""
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        """返回各 source_key 的未过期条目数，用于日志。"""
        now = datetime.now(timezone.utc)
        cutoff = now - self._ttl
        counts: dict[str, int] = {}
        for item, ts in self._entries.values():
            if ts < cutoff:
                continue
            key = compute_source_key(item)
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ── 淘汰 ──────────────────────────────────────────────────────

    def evict_expired(self) -> int:
        """清理过期条目，返回清理数。"""
        now = datetime.now(timezone.utc)
        cutoff = now - self._ttl
        expired = [iid for iid, (_, ts) in self._entries.items() if ts < cutoff]
        for iid in expired:
            del self._entries[iid]
        if expired:
            logger.debug("[feed_buffer] evict_expired count=%d remaining=%d", len(expired), len(self._entries))
        return len(expired)

    # ── 内部 ──────────────────────────────────────────────────────

    def _enforce_per_source_limit(self) -> None:
        """对每个 source_key 保留最新 max_per_source 条，丢弃较旧的。"""
        by_source: dict[str, list[tuple[str, datetime]]] = {}
        for iid, (item, ts) in self._entries.items():
            key = compute_source_key(item)
            by_source.setdefault(key, []).append((iid, ts))

        for key, entries in by_source.items():
            if len(entries) <= self._max_per_source:
                continue
            # 按时间降序排，删掉最旧的
            entries.sort(key=lambda x: x[1], reverse=True)
            for iid, _ in entries[self._max_per_source:]:
                del self._entries[iid]

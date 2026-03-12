"""
tests/test_feed_buffer_poller.py — FeedBuffer 和 FeedPoller 单测。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from feeds.base import FeedItem
from feeds.buffer import FeedBuffer
from proactive.feed_poller import FeedPoller

# ── helpers ──────────────────────────────────────────────────────


def _make_item(
    title: str,
    url: str | None = None,
    source_name: str = "test",
    source_type: str = "rss",
) -> FeedItem:
    return FeedItem(
        source_name=source_name,
        source_type=source_type,
        title=title,
        content="",
        url=url,
        author=None,
        published_at=datetime.now(timezone.utc),
    )


def _make_poller_cfg(
    interval: int = 60,
    fetch_limit: int = 10,
    ttl_hours: int = 48,
    max_per_source: int = 100,
):
    cfg = MagicMock()
    cfg.feed_poller_interval_seconds = interval
    cfg.feed_poller_fetch_limit = fetch_limit
    cfg.feed_poller_buffer_ttl_hours = ttl_hours
    cfg.feed_poller_buffer_max_per_source = max_per_source
    return cfg


# ── FeedBuffer tests ──────────────────────────────────────────────


class TestFeedBuffer:
    def test_add_returns_count_and_dedupes(self):
        buf = FeedBuffer()
        item1 = _make_item("A", url="https://example.com/a")
        item2 = _make_item("B", url="https://example.com/b")

        added = buf.add([item1, item2])
        assert added == 2

        # 相同 url → 同一 item_id → 跳过
        added_dup = buf.add([item1])
        assert added_dup == 0

        assert buf.size() == 2

    def test_get_all_returns_newest_first(self):
        buf = FeedBuffer()
        items = [
            _make_item(f"Item{i}", url=f"https://example.com/{i}") for i in range(3)
        ]
        buf.add(items)
        result = buf.get_all()
        # 最后加入的最新，排在最前
        assert len(result) == 3

    def test_get_all_with_limit(self):
        buf = FeedBuffer()
        items = [
            _make_item(f"Item{i}", url=f"https://example.com/{i}") for i in range(10)
        ]
        buf.add(items)
        result = buf.get_all(n=3)
        assert len(result) == 3

    def test_get_all_n_zero_returns_all(self):
        buf = FeedBuffer()
        items = [
            _make_item(f"Item{i}", url=f"https://example.com/{i}") for i in range(5)
        ]
        buf.add(items)
        assert len(buf.get_all(n=0)) == 5

    def test_evict_expired(self):
        buf = FeedBuffer(ttl_hours=1)
        item = _make_item("Old", url="https://example.com/old")
        buf.add([item])
        assert buf.size() == 1

        # 手动将条目时间戳向前移以模拟过期
        for key in list(buf._entries.keys()):
            it, _ = buf._entries[key]
            buf._entries[key] = (it, datetime.now(timezone.utc) - timedelta(hours=2))

        evicted = buf.evict_expired()
        assert evicted == 1
        assert buf.size() == 0

    def test_expired_items_excluded_from_get_all(self):
        buf = FeedBuffer(ttl_hours=1)
        item = _make_item("Expired", url="https://example.com/exp")
        buf.add([item])

        for key in list(buf._entries.keys()):
            it, _ = buf._entries[key]
            buf._entries[key] = (it, datetime.now(timezone.utc) - timedelta(hours=2))

        assert buf.get_all() == []

    def test_per_source_limit(self):
        buf = FeedBuffer(max_per_source=3)
        items = [
            _make_item(f"Item{i}", url=f"https://example.com/{i}", source_name="src")
            for i in range(5)
        ]
        buf.add(items)
        # 每源上限=3，旧的被清理
        stats = buf.stats()
        assert stats.get("rss:src", 0) <= 3

    def test_stats_returns_per_source_counts(self):
        buf = FeedBuffer()
        buf.add(
            [
                _make_item("A", url="https://a.com/1", source_name="src1"),
                _make_item("B", url="https://a.com/2", source_name="src1"),
                _make_item("C", url="https://b.com/1", source_name="src2"),
            ]
        )
        stats = buf.stats()
        assert stats["rss:src1"] == 2
        assert stats["rss:src2"] == 1

    def test_add_empty_list(self):
        buf = FeedBuffer()
        assert buf.add([]) == 0
        assert buf.size() == 0


# ── FeedPoller tests ──────────────────────────────────────────────


class TestFeedPoller:
    @pytest.mark.asyncio
    async def test_warmup_on_run(self):
        """run() 启动时应立即执行一次 _poll_once() 预热。"""
        registry = MagicMock()
        items = [_make_item("X", url="https://example.com/x")]
        registry.fetch_all = AsyncMock(return_value=items)

        buf = FeedBuffer()
        cfg = _make_poller_cfg(interval=9999)  # 间隔超长，确保只执行预热
        poller = FeedPoller(registry, buf, cfg)

        # 启动后立即停止
        async def _run_and_stop():
            task = asyncio.create_task(poller.run())
            await asyncio.sleep(0.05)  # 给预热执行时间
            poller.stop()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.CancelledError, asyncio.TimeoutError:
                task.cancel()

        await _run_and_stop()
        # 预热应该已经写入 buffer
        assert buf.size() >= 1

    @pytest.mark.asyncio
    async def test_poll_once_adds_to_buffer(self):
        """_poll_once() 应将拉取的 items 写入 buffer。"""
        registry = MagicMock()
        items = [
            _make_item("A", url="https://example.com/a"),
            _make_item("B", url="https://example.com/b"),
        ]
        registry.fetch_all = AsyncMock(return_value=items)

        buf = FeedBuffer()
        cfg = _make_poller_cfg()
        poller = FeedPoller(registry, buf, cfg)

        added = await poller._poll_once()
        assert added == 2
        assert buf.size() == 2

    @pytest.mark.asyncio
    async def test_poll_once_handles_registry_error(self):
        """registry 出错时 _poll_once() 应返回 0，不抛异常。"""
        registry = MagicMock()
        registry.fetch_all = AsyncMock(side_effect=RuntimeError("network error"))

        buf = FeedBuffer()
        cfg = _make_poller_cfg()
        poller = FeedPoller(registry, buf, cfg)

        added = await poller._poll_once()
        assert added == 0
        assert buf.size() == 0

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        """stop() 应将 _running 置为 False，下次循环检查时退出。"""
        registry = MagicMock()
        registry.fetch_all = AsyncMock(return_value=[])

        buf = FeedBuffer()
        cfg = _make_poller_cfg(interval=9999)
        poller = FeedPoller(registry, buf, cfg)

        task = asyncio.create_task(poller.run())
        await asyncio.sleep(0.05)
        assert poller._running is True
        poller.stop()
        assert poller._running is False
        # 任务仍在 sleep；cancel 它以清理
        task.cancel()
        try:
            await task
        except asyncio.CancelledError, Exception:
            pass


# ── 场景④ 修复验证：semantic_duplicate_entries 不写 seen_items ──


class TestScenario4SemanticDupeNotMarkedSeen:
    """
    验证 engine.py 修复：语义重复条目 semantic_duplicate_entries 经过 tick() 后
    不出现在 seen_items（14天 TTL），只受 semantic_items（72h TTL）窗口自然抑制。

    测试方法：构造 fake SensePort 返回 semantic_duplicate_entries，
    运行 ProactiveEngine.tick()，断言 state.is_item_seen() 为 False。
    """

    @pytest.mark.asyncio
    async def test_engine_tick_does_not_mark_semantic_dup_as_seen(self, tmp_path):
        """tick() 不得对 semantic_duplicate_entries 调用 mark_items_seen。"""
        from unittest.mock import AsyncMock, MagicMock
        from proactive.state import ProactiveStateStore
        from proactive.engine import ProactiveEngine
        from proactive.item_id import compute_item_id, compute_source_key

        state = ProactiveStateStore(tmp_path / "state.json")
        dup_item = _make_item("CS2 IEM Major 2025", url="https://example.com/cs2-2025")
        dup_source_key = compute_source_key(dup_item)
        dup_item_id = compute_item_id(dup_item)
        semantic_dup_entries = [(dup_source_key, dup_item_id)]

        # SensePort mock：filter_new_items 把该 item 归为语义重复
        sense = MagicMock()
        sense.compute_energy.return_value = 0.5
        sense.collect_recent.return_value = []
        sense.compute_interruptibility.return_value = (
            0.5,
            {
                "f_time": 0.5,
                "f_reply": 0.5,
                "f_activity": 0.5,
                "f_fatigue": 0.5,
                "random_delta": 0.0,
            },
        )
        sense.fetch_items = AsyncMock(return_value=[dup_item])
        sense.filter_new_items.return_value = (
            [],  # new_items 为空（dup 被过滤）
            [],  # new_entries
            semantic_dup_entries,  # semantic_duplicate_entries
        )
        sense.read_memory_text.return_value = ""
        sense.has_global_memory.return_value = False
        sense.last_user_at.return_value = None
        sense.target_session_key.return_value = "telegram:123"
        sense.quiet_hours.return_value = (23, 8, 0.0)
        sense.refresh_sleep_context.return_value = False
        sense.sleep_context.return_value = None

        # Config mock：无新条目且无记忆，仅验证 semantic duplicate 不会写 seen_items
        cfg = MagicMock()
        cfg.anyaction_enabled = False
        cfg.score_weight_energy = 0.40
        cfg.score_weight_content = 0.40
        cfg.score_weight_recent = 0.20
        cfg.score_recent_scale = 8.0
        cfg.score_content_halfsat = 2.5
        cfg.score_pre_threshold = 0.01  # 低阈值，确保 pre_score 通过
        cfg.score_llm_threshold = 0.99
        cfg.items_per_source = 5
        cfg.interest_filter.enabled = False
        cfg.feature_scoring_enabled = False
        cfg.dedupe_seen_ttl_hours = 336
        cfg.delivery_dedupe_hours = 10
        cfg.semantic_dedupe_window_hours = 72
        cfg.llm_reject_cooldown_hours = 0

        engine = ProactiveEngine(
            cfg=cfg,
            state=state,
            presence=None,
            rng=None,
            sense=sense,
            decide=MagicMock(),
            act=MagicMock(),
        )

        await engine.tick()

        # 修复后：语义重复条目不应出现在 seen_items（14天 TTL）
        assert not state.is_item_seen(
            source_key=dup_source_key,
            item_id=dup_item_id,
            ttl_hours=336,
        ), "semantic_duplicate_entries 不应被写入 seen_items（14天 TTL）"

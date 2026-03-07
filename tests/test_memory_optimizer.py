"""Tests for current proactive.memory_optimizer behavior."""

import asyncio
import types
from datetime import datetime
from unittest.mock import AsyncMock

from agent.memory import MemoryStore
from core.memory.port import DefaultMemoryPort
from proactive.memory_optimizer import MemoryOptimizer, MemoryOptimizerLoop


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


def _provider_with_responses(*responses: str) -> object:
    provider = types.SimpleNamespace()
    provider.chat = AsyncMock(side_effect=[_Resp(x) for x in responses])
    return provider


def test_optimize_skips_when_memory_pending_history_all_empty(tmp_path):
    memory = DefaultMemoryPort(MemoryStore(tmp_path))
    provider = types.SimpleNamespace()
    provider.chat = AsyncMock()

    optimizer = MemoryOptimizer(memory, provider, "test-model")
    asyncio.run(optimizer.optimize())

    provider.chat.assert_not_called()


def test_optimize_rewrites_memory_from_first_llm_call(tmp_path):
    memory = DefaultMemoryPort(MemoryStore(tmp_path))
    memory.write_long_term("old profile")

    provider = _provider_with_responses("## 用户画像\n- 新版本\n")
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    asyncio.run(optimizer.optimize())

    assert memory.read_long_term().strip() == "## 用户画像\n- 新版本"


def test_optimize_rolls_back_snapshot_when_merge_returns_empty(tmp_path):
    memory = DefaultMemoryPort(MemoryStore(tmp_path))
    memory.write_long_term("old profile")
    memory.append_pending("- pending fact")

    provider = _provider_with_responses("")
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    asyncio.run(optimizer.optimize())

    assert "pending fact" in memory.read_pending()
    assert not memory._store._snapshot_path.exists()


def test_optimize_updates_self_when_self_and_history_exist(tmp_path):
    memory = DefaultMemoryPort(MemoryStore(tmp_path))
    memory.write_long_term("old")
    memory.write_self("## 原 SELF")
    memory.append_history("[2026-03-03 10:00] USER: hi")

    provider = _provider_with_responses(
        "## 新记忆",
        "## 新 SELF",
    )
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    asyncio.run(optimizer.optimize())

    assert memory.read_self().strip() == "## 新 SELF"


def test_seconds_until_next_tick_aligns_to_interval_boundary():
    now = datetime(2026, 2, 23, 12, 34, 56)
    loop = MemoryOptimizerLoop(None, interval_seconds=3600, _now_fn=lambda: now)

    secs = loop._seconds_until_next_tick()

    assert abs(secs - (25 * 60 + 4)) < 0.001


def test_seconds_until_next_tick_always_positive():
    for h in range(24):
        now = datetime(2026, 2, 23, h, 59, 59)
        loop = MemoryOptimizerLoop(None, interval_seconds=300, _now_fn=lambda n=now: n)
        assert loop._seconds_until_next_tick() > 0

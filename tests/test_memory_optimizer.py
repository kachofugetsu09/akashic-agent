"""Tests for current proactive.memory_optimizer behavior."""

import asyncio
import types
from datetime import datetime
from unittest.mock import AsyncMock

from agent.memory import MemoryStore
from core.memory.port import DefaultMemoryPort
from proactive_v2.memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizerLoop,
    _parse_cleanup_json,
)


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
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    provider.chat.assert_not_called()


def test_optimize_rewrites_memory_from_first_llm_call(tmp_path):
    memory = DefaultMemoryPort(MemoryStore(tmp_path))
    memory.write_long_term("old profile")

    provider = _provider_with_responses("## 用户画像\n- 新版本\n")
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    assert memory.read_long_term().strip() == "## 用户画像\n- 新版本"


def test_optimize_rolls_back_snapshot_when_merge_returns_empty(tmp_path):
    memory = DefaultMemoryPort(MemoryStore(tmp_path))
    memory.write_long_term("old profile")
    memory.append_pending("- pending fact")

    provider = _provider_with_responses("")
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    assert "pending fact" in memory.read_pending()
    assert not memory._store._snapshot_path.exists()


def test_optimize_updates_self_using_pending_only(tmp_path):
    memory = DefaultMemoryPort(MemoryStore(tmp_path))
    memory.write_long_term("old")
    memory.write_self("## 原 SELF")
    memory.append_pending("- [preference] 回复保持简洁。")
    memory.append_history("[2026-03-03 10:00] USER: 这段历史不该进入 SELF")

    provider = _provider_with_responses(
        "## 新记忆",
        "# Akashic 的自我认知\n\n## 人格与形象\n\n- 新版人格\n\n## 我对当前用户的理解\n\n- 新版理解\n\n## 我们关系的定义\n\n- 新版关系\n",
    )
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    assert memory.read_self().strip().startswith("# Akashic 的自我认知")
    assert "新版理解" in memory.read_self()

    self_prompt = provider.chat.await_args_list[1].kwargs["messages"][1]["content"]
    assert "- [preference] 回复保持简洁。" in self_prompt
    assert "这段历史不该进入 SELF" not in self_prompt


def test_merge_memory_ignores_history_and_only_uses_pending(tmp_path):
    memory = DefaultMemoryPort(MemoryStore(tmp_path))
    memory.write_long_term("old profile")
    memory.append_pending("- [identity] 新身份")
    memory.append_history("[2026-03-03 10:00] USER: 这段历史不该进入长期记忆")

    provider = _provider_with_responses("## 用户画像\n- 新版本\n")
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    call = provider.chat.await_args_list[0]
    prompt = call.kwargs["messages"][1]["content"]

    assert "近期历史摘要" not in prompt
    assert "- [identity] 新身份" in prompt


def test_parse_cleanup_json_supports_fenced_json():
    ongoing, pending = _parse_cleanup_json(
        '```json\n{"remove_ongoing":["明天返校"],"remove_pending":["确认显卡型号"]}\n```'
    )

    assert ongoing == ["明天返校"]
    assert pending == ["确认显卡型号"]


def test_request_text_response_uses_expected_chat_kwargs(tmp_path):
    memory = DefaultMemoryPort(MemoryStore(tmp_path))
    provider = _provider_with_responses("merged")
    optimizer = MemoryOptimizer(memory, provider, "test-model")

    result = asyncio.run(
        optimizer._request_text_response(
            system_content="system",
            user_content="user",
            max_tokens=123,
        )
    )

    assert result == "merged"
    kwargs = provider.chat.await_args.kwargs
    assert kwargs["tools"] == []
    assert kwargs["model"] == "test-model"
    assert kwargs["max_tokens"] == 123


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

"""覆盖当前 proactive memory optimizer 行为。"""

from typing import Any, cast
import asyncio
import types
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from proactive_v2.memory_optimizer import (
    MemoryOptimizerBusy,
    MemoryOptimizer,
    MemoryOptimizerOutputError,
    MemoryOptimizerLoop,
)
from core.memory.markdown import MarkdownMemoryStore


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


_VALID_MEMORY = """# 用户长期记忆

## 用户事实
- 新版事实

## 用户偏好
- 新版偏好

## 用户明确要求长期记住的关键内容
- 新版要求
"""

_VALID_SELF = """# Akashic 的自我认知

## 人格与形象
- 新版人格

## 我对当前用户的理解
- 新版理解

## 我们关系的定义
- 新版关系
"""


def _provider_with_responses(*responses: str) -> object:
    provider = types.SimpleNamespace()
    provider.chat = AsyncMock(side_effect=[_Resp(x) for x in responses])
    return provider


def test_optimize_skips_when_memory_pending_history_all_empty(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    provider = types.SimpleNamespace()
    provider.chat = AsyncMock()

    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    provider.chat.assert_not_called()


def test_optimize_commits_marker_only_pending_snapshot(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    _ = memory.pending_file.write_text(
        "<!-- consolidation:test:pending -->\n",
        encoding="utf-8",
    )
    provider = types.SimpleNamespace()
    provider.chat = AsyncMock()
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0

    asyncio.run(optimizer.optimize())

    provider.chat.assert_not_called()
    assert not memory._snapshot_path.exists()
    assert memory.pending_file.exists()
    assert memory.read_pending() == ""


def test_optimize_rewrites_memory_from_first_llm_call(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old profile")

    provider = _provider_with_responses(_VALID_MEMORY, "")
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    assert memory.read_long_term().strip() == _VALID_MEMORY.strip()
    assert (memory.memory_dir / "MEMORY.bak.md").read_text(encoding="utf-8") == (
        "old profile"
    )
    history = list((memory.memory_dir / "backups").glob("MEMORY.*.bak.md"))
    assert len(history) == 1
    assert history[0].read_text(encoding="utf-8") == "old profile"


def test_optimize_rejects_invalid_memory_and_restores_pending(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term(_VALID_MEMORY)
    memory.append_pending("- [identity] 新身份")
    provider = _provider_with_responses("好的，已经整理完成。")
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(MemoryOptimizerOutputError, match="MEMORY.md"):
        asyncio.run(optimizer.optimize())

    assert memory.read_long_term() == _VALID_MEMORY
    assert "新身份" in memory.read_pending()
    assert not (memory.memory_dir / "MEMORY.bak.md").exists()


def test_optimize_rolls_back_snapshot_when_merge_returns_empty(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old profile")
    memory.append_pending("- pending fact")

    provider = _provider_with_responses("", "")
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    assert "pending fact" in memory.read_pending()
    assert not memory._snapshot_path.exists()


def test_optimize_rolls_back_snapshot_and_propagates_merge_failure(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old profile")
    memory.append_pending("- pending fact")
    provider = types.SimpleNamespace()
    provider.chat = AsyncMock(side_effect=RuntimeError("merge failed"))
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(RuntimeError, match="merge failed"):
        asyncio.run(optimizer.optimize())

    assert "pending fact" in memory.read_pending()
    assert not memory._snapshot_path.exists()
    assert memory.read_long_term().strip() == "old profile"


def test_optimize_rolls_back_snapshot_when_memory_write_fails(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.append_pending("- pending fact")

    async def break_memory_file(**_kwargs: Any) -> _Resp:
        memory.memory_file.mkdir()
        return _Resp(_VALID_MEMORY)

    provider = types.SimpleNamespace()
    provider.chat = AsyncMock(side_effect=break_memory_file)
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(IsADirectoryError):
        asyncio.run(optimizer.optimize())

    assert "pending fact" in memory.read_pending()
    assert not memory._snapshot_path.exists()


def test_optimize_propagates_cancellation_and_restores_pending(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.append_pending("- pending fact")
    provider = types.SimpleNamespace()
    provider.chat = AsyncMock(side_effect=asyncio.CancelledError)
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(optimizer.optimize())

    assert "pending fact" in memory.read_pending()
    assert not memory._snapshot_path.exists()


def test_optimize_updates_self_using_pending_only(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old")
    memory.write_self("原 SELF")
    memory.append_pending("- [preference] 回复保持简洁。")

    provider = _provider_with_responses(
        _VALID_MEMORY,
        _VALID_SELF,
    )
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    assert memory.read_self().strip().startswith("# Akashic 的自我认知")
    assert "新版理解" in memory.read_self()
    assert (memory.memory_dir / "SELF.bak.md").read_text(encoding="utf-8") == (
        "原 SELF"
    )
    history = list((memory.memory_dir / "backups").glob("SELF.*.bak.md"))
    assert len(history) == 1
    assert history[0].read_text(encoding="utf-8") == "原 SELF"

    self_prompt = provider.chat.await_args_list[1].kwargs["messages"][1]["content"]
    assert "- [preference] 回复保持简洁。" in self_prompt


def test_optimize_propagates_self_update_failure(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old")
    memory.write_self("## 原 SELF")
    memory.append_pending("- [preference] 回复保持简洁。")
    provider = types.SimpleNamespace()
    provider.chat = AsyncMock(
        side_effect=[_Resp(_VALID_MEMORY), RuntimeError("self update failed")]
    )
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(RuntimeError, match="self update failed"):
        asyncio.run(optimizer.optimize())

    assert memory.read_long_term().strip() == _VALID_MEMORY.strip()
    assert memory.read_self().strip() == "## 原 SELF"


def test_optimize_rejects_invalid_self_without_overwriting_file(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old")
    memory.write_self(_VALID_SELF)
    memory.append_pending("- [preference] 回复保持简洁。")
    provider = _provider_with_responses(_VALID_MEMORY, "# Akashic 的自我认知")
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(MemoryOptimizerOutputError, match="SELF.md"):
        asyncio.run(optimizer.optimize())

    assert memory.read_self() == _VALID_SELF
    assert not (memory.memory_dir / "SELF.bak.md").exists()


def test_merge_memory_ignores_history_and_only_uses_pending(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old profile")
    memory.append_pending("- [identity] 新身份")

    provider = _provider_with_responses(_VALID_MEMORY, "")
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    call = provider.chat.await_args_list[0]
    prompt = call.kwargs["messages"][1]["content"]

    assert "近期历史摘要" not in prompt
    assert "- [identity] 新身份" in prompt


def test_request_text_response_uses_expected_chat_kwargs(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    provider = _provider_with_responses("merged")
    optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")

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


def test_optimize_reports_busy_instead_of_waiting(tmp_path):
    async def run_case() -> None:
        memory = MarkdownMemoryStore(tmp_path)
        provider = types.SimpleNamespace()
        provider.chat = AsyncMock()
        optimizer = MemoryOptimizer(memory, cast(Any, provider), "test-model")
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_optimize() -> None:
            started.set()
            await release.wait()

        optimizer._optimize = blocked_optimize  # type: ignore[method-assign]
        running = asyncio.create_task(optimizer.optimize())
        await started.wait()

        assert optimizer.is_running
        with pytest.raises(MemoryOptimizerBusy):
            await optimizer.optimize()

        release.set()
        await running

    asyncio.run(run_case())


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

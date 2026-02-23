"""TDD for proactive/memory_optimizer.py"""
import json
import types
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from agent.memory import MemoryStore
from proactive.memory_optimizer import MemoryOptimizer, MemoryOptimizerLoop, _format_questions


# ── helpers ──────────────────────────────────────────────────────

class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


def _mock_provider(memory_reply: str = "", questions_reply: str = "") -> object:
    """两次调用：第一次返回 memory rewrite，第二次返回 questions JSON。"""
    provider = types.SimpleNamespace()
    provider.chat = AsyncMock(side_effect=[
        _Resp(memory_reply),
        _Resp(questions_reply),
    ])
    return provider


def _q_json(questions: list[str]) -> str:
    return json.dumps({"questions": questions}, ensure_ascii=False)


# ── _format_questions ─────────────────────────────────────────────

def test_format_questions_numbers_each():
    qs = ["问题A", "问题B", "问题C"]
    result = _format_questions(qs)
    assert "1. 问题A" in result
    assert "2. 问题B" in result
    assert "3. 问题C" in result


def test_format_questions_has_header():
    result = _format_questions(["Q"])
    assert "## " in result


def test_format_questions_empty_returns_header_only():
    result = _format_questions([])
    assert result.strip() != ""   # header still present


# ── MemoryOptimizer ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_optimize_rewrites_memory(tmp_path):
    """optimize() 应用 LLM 重写 MEMORY.md。"""
    memory = MemoryStore(tmp_path)
    memory.write_long_term("旧的冗长记忆内容")

    new_mem = "## 用户画像\n- 测试用户\n"
    provider = _mock_provider(
        memory_reply=new_mem,
        questions_reply=_q_json(["Q1", "Q2", "Q3", "Q4", "Q5"]),
    )
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    await optimizer.optimize()

    assert memory.read_long_term().strip() == new_mem.strip()


@pytest.mark.asyncio
async def test_optimize_writes_five_questions(tmp_path):
    """optimize() 应生成并写入问题列表。"""
    memory = MemoryStore(tmp_path)
    memory.write_long_term("用户信息")

    questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    provider = _mock_provider(
        memory_reply="## 用户画像\n- 用户\n",
        questions_reply=_q_json(questions),
    )
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    await optimizer.optimize()

    result = memory.read_questions()
    for q in questions:
        assert q in result


@pytest.mark.asyncio
async def test_optimize_overwrites_previous_questions(tmp_path):
    """每次 optimize 都应覆盖写入，不追加。"""
    memory = MemoryStore(tmp_path)
    memory.write_long_term("用户")
    memory.write_questions("## 旧问题\n\n1. 旧问题1\n")

    provider = _mock_provider(
        memory_reply="## 用户画像\n- 用户\n",
        questions_reply=_q_json(["新Q1", "新Q2", "新Q3", "新Q4", "新Q5"]),
    )
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    await optimizer.optimize()

    result = memory.read_questions()
    assert "旧问题1" not in result
    assert "新Q1" in result


@pytest.mark.asyncio
async def test_optimize_skips_when_empty_memory_and_history(tmp_path):
    """无记忆无历史时跳过，不调 LLM。"""
    memory = MemoryStore(tmp_path)
    provider = types.SimpleNamespace()
    provider.chat = AsyncMock()

    optimizer = MemoryOptimizer(memory, provider, "test-model")
    await optimizer.optimize()

    provider.chat.assert_not_called()


@pytest.mark.asyncio
async def test_optimize_calls_llm_twice(tmp_path):
    """optimize() 应调用 LLM 两次：一次重写记忆，一次生成问题。"""
    memory = MemoryStore(tmp_path)
    memory.write_long_term("有内容")

    provider = _mock_provider(
        memory_reply="## 用户画像\n- 用户\n",
        questions_reply=_q_json(["Q1", "Q2", "Q3", "Q4", "Q5"]),
    )
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    await optimizer.optimize()

    assert provider.chat.await_count == 2


@pytest.mark.asyncio
async def test_optimize_handles_llm_failure_gracefully(tmp_path):
    """LLM 失败时不应抛异常，保留原有记忆。"""
    memory = MemoryStore(tmp_path)
    memory.write_long_term("原始记忆")

    provider = types.SimpleNamespace()
    provider.chat = AsyncMock(side_effect=RuntimeError("LLM error"))

    optimizer = MemoryOptimizer(memory, provider, "test-model")
    await optimizer.optimize()   # 不应抛异常

    # 原记忆保留（因为重写失败返回空字符串，不会写回）
    assert memory.read_long_term() == "原始记忆"


@pytest.mark.asyncio
async def test_optimize_truncates_to_five_questions(tmp_path):
    """即使 LLM 返回超过 5 个问题，也只保留前 5 个。"""
    memory = MemoryStore(tmp_path)
    memory.write_long_term("用户")

    too_many = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
    provider = _mock_provider(
        memory_reply="## 用户画像\n- 用户\n",
        questions_reply=_q_json(too_many),
    )
    optimizer = MemoryOptimizer(memory, provider, "test-model")
    await optimizer.optimize()

    result = memory.read_questions()
    assert "Q5" in result
    assert "Q6" not in result


# ── MemoryOptimizerLoop scheduling ────────────────────────────────

def test_seconds_until_midnight_at_11pm():
    """23:00 → 1 小时后到午夜。"""
    now = datetime(2026, 2, 23, 23, 0, 0)
    loop = MemoryOptimizerLoop(None, _now_fn=lambda: now)
    secs = loop._seconds_until_midnight()
    assert abs(secs - 3600) < 1


def test_seconds_until_midnight_at_1am():
    """01:00 → 23 小时后到午夜。"""
    now = datetime(2026, 2, 23, 1, 0, 0)
    loop = MemoryOptimizerLoop(None, _now_fn=lambda: now)
    secs = loop._seconds_until_midnight()
    assert abs(secs - 82800) < 1


def test_seconds_until_midnight_at_noon():
    """12:00 → 12 小时后到午夜。"""
    now = datetime(2026, 2, 23, 12, 0, 0)
    loop = MemoryOptimizerLoop(None, _now_fn=lambda: now)
    secs = loop._seconds_until_midnight()
    assert abs(secs - 43200) < 1


def test_seconds_until_midnight_always_positive():
    """任何时刻结果应 > 0。"""
    for h in range(24):
        now = datetime(2026, 2, 23, h, 30, 0)
        loop = MemoryOptimizerLoop(None, _now_fn=lambda n=now: n)
        assert loop._seconds_until_midnight() > 0

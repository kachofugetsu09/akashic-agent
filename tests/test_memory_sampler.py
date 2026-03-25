"""
TDD for proactive/memory_sampler.py

随机记忆抽取：把 MEMORY.md 文本切成语义块，随机挑 N 条注入 LLM prompt。
纯函数，无 I/O，无 LLM。
"""

import random
import pytest
from proactive_v2.memory_sampler import split_memory_chunks, sample_memory_chunks

MEMORY_MD = """\
## 用户偏好

- 喜欢单机 RPG，尤其是魂类游戏
- 不喜欢电竞和 MOBA
- 关注 AI 前沿进展

## 工作习惯

- 通常早上 10 点开始工作
- 用 Python 和 TypeScript
- 不喜欢被打断

## 最近聊过的话题

上周聊了 Elden Ring DLC 的剧情，很感兴趣。
讨论过 Claude 3.5 的上下文窗口限制。
"""


# ── split_memory_chunks ───────────────────────────────────────────


def test_split_returns_nonempty_list():
    chunks = split_memory_chunks(MEMORY_MD)
    assert len(chunks) > 0


def test_split_empty_string_returns_empty():
    assert split_memory_chunks("") == []
    assert split_memory_chunks("   \n  ") == []


def test_split_chunks_are_nonempty_strings():
    for chunk in split_memory_chunks(MEMORY_MD):
        assert isinstance(chunk, str)
        assert chunk.strip()


def test_split_separates_sections():
    """## 标题 应成为独立块或与紧跟内容合并，不能把两个 section 混在一块。"""
    chunks = split_memory_chunks(MEMORY_MD)
    # 不应有超过 400 字符的块（说明没有合并过多内容）
    for chunk in chunks:
        assert len(chunk) <= 400, f"chunk too long: {chunk[:80]!r}"


def test_split_preserves_content():
    """所有原始内容应能在合并后的 chunks 里找到。"""
    chunks = split_memory_chunks(MEMORY_MD)
    merged = "\n".join(chunks)
    assert "魂类游戏" in merged
    assert "Elden Ring" in merged
    assert "Python" in merged


def test_split_single_line_memory():
    text = "用户叫小明，喜欢打篮球。"
    chunks = split_memory_chunks(text)
    assert len(chunks) == 1
    assert "小明" in chunks[0]


# ── sample_memory_chunks ──────────────────────────────────────────


def test_sample_returns_requested_count():
    chunks = split_memory_chunks(MEMORY_MD)
    rng = random.Random(1)
    result = sample_memory_chunks(MEMORY_MD, n=2, rng=rng)
    assert len(result) == 2


def test_sample_returns_fewer_when_not_enough_chunks():
    text = "只有一行记忆。"
    result = sample_memory_chunks(text, n=5)
    assert len(result) == 1


def test_sample_returns_empty_for_empty_memory():
    result = sample_memory_chunks("", n=3)
    assert result == []


def test_sample_is_deterministic_with_seed():
    rng1 = random.Random(99)
    rng2 = random.Random(99)
    r1 = sample_memory_chunks(MEMORY_MD, n=2, rng=rng1)
    r2 = sample_memory_chunks(MEMORY_MD, n=2, rng=rng2)
    assert r1 == r2


def test_sample_varies_across_different_seeds():
    results = set()
    for seed in range(20):
        rng = random.Random(seed)
        r = sample_memory_chunks(MEMORY_MD, n=1, rng=rng)
        results.add(r[0])
    assert (
        len(results) > 1
    ), "sampling should produce different results with different seeds"


def test_sample_n_zero_returns_empty():
    assert sample_memory_chunks(MEMORY_MD, n=0) == []


def test_sample_chunks_are_original_content():
    """抽取的每个片段都应该是原文的一部分。"""
    rng = random.Random(7)
    result = sample_memory_chunks(MEMORY_MD, n=3, rng=rng)
    all_chunks = split_memory_chunks(MEMORY_MD)
    for chunk in result:
        assert chunk in all_chunks

"""
HyDE 检索增强单元测试。

覆盖：
  1. hypothesis 生成超时 → 降级返回 raw 结果，无异常
  2. raw 结果完整保留（union_dedup：id 不丢，score 不改）
  3. used_hyde 只在 HyDE 实际追加条目时标记为 True
"""

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

from agent.provider import LLMResponse
from memory2.hyde_enhancer import HyDEEnhancer, _union_dedup


def test_hypothesis_timeout_falls_back_to_raw():
    raw_items = [{"id": "a", "score": 0.7}, {"id": "b", "score": 0.6}]

    async def slow_chat(**kwargs):
        await asyncio.sleep(10)  # 超过 timeout
        return LLMResponse(content="假想条目", tool_calls=[])

    provider = MagicMock()
    provider.chat = slow_chat

    enhancer = HyDEEnhancer(
        light_provider=cast(Any, provider),
        light_model="qwen-flash",
        timeout_s=0.05,  # 50ms，必然超时
    )

    async def fake_retrieve(query, **kwargs):
        return raw_items

    results, used_hyde = asyncio.run(
        enhancer.augment(
            raw_query="测试问题",
            context="",
            retrieve_fn=fake_retrieve,
            top_k=6,
        )
    )

    assert results == raw_items
    assert used_hyde is False


# ── 2. raw 结果完整保留（id 不丢，score 不变）────────────────────────────────


def test_union_dedup_raw_preserved():
    raw = [{"id": "a", "score": 0.7}, {"id": "b", "score": 0.6}]
    hyde = [{"id": "b", "score": 0.9}, {"id": "c", "score": 0.8}]  # b 重复，c 新增

    result = _union_dedup(raw, hyde)

    ids = [r["id"] for r in result]
    scores = {r["id"]: r["score"] for r in result}

    # raw 条目全部存在
    assert "a" in ids
    assert "b" in ids
    # hyde 新增条目追加
    assert "c" in ids
    # 总数正确（去重后 3 条）
    assert len(result) == 3
    # score 值未被修改
    assert scores["a"] == 0.7
    assert scores["b"] == 0.6  # 保持 raw 的分数，不被 hyde 的 0.9 覆盖
    # raw 条目在前
    assert result[0]["id"] == "a"
    assert result[1]["id"] == "b"


# ── 3. used_hyde 标记 ─────────────────────────────────────────────────────────


def test_used_hyde_true_when_hyde_appended_new_item():
    """HyDE 追加了新条目时 used_hyde=True。"""
    raw_items = [{"id": "a", "score": 0.7}]
    hyde_items = [{"id": "b", "score": 0.8}]  # 全新条目

    async def fake_chat(**kwargs):
        return LLMResponse(content="假想条目", tool_calls=[])

    async def fake_retrieve(query, **kwargs):
        if query == "假想条目":
            return hyde_items
        return raw_items

    provider = MagicMock()
    provider.chat = fake_chat

    enhancer = HyDEEnhancer(
        light_provider=cast(Any, provider),
        light_model="qwen-flash",
        timeout_s=2.0,
    )

    results, used_hyde = asyncio.run(
        enhancer.augment(
            raw_query="原始问题",
            context="",
            retrieve_fn=fake_retrieve,
            top_k=6,
        )
    )

    assert used_hyde is True
    assert len(cast(Any, results)) == 2


def test_used_hyde_false_when_hyde_adds_nothing_new():
    """HyDE 命中条目全部已在 raw 中时 used_hyde=False。"""
    items = [{"id": "a", "score": 0.7}]

    async def fake_chat(**kwargs):
        return LLMResponse(content="假想条目", tool_calls=[])

    async def fake_retrieve(query, **kwargs):
        return items  # 两路返回相同条目

    provider = MagicMock()
    provider.chat = fake_chat

    enhancer = HyDEEnhancer(
        light_provider=cast(Any, provider),
        light_model="qwen-flash",
        timeout_s=2.0,
    )

    results, used_hyde = asyncio.run(
        enhancer.augment(
            raw_query="原始问题",
            context="",
            retrieve_fn=fake_retrieve,
            top_k=6,
        )
    )

    assert used_hyde is False
    assert len(cast(Any, results)) == 1

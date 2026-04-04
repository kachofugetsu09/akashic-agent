"""
HyDE 检索增强单元测试。

覆盖：
  1. HyDE 关闭时 AgentLoop 不构建 enhancer
  2. light_model 未配置时即使 hyde_enabled=True 也禁用（不回退主模型）
  3. hypothesis 生成超时 → 降级返回 raw 结果，无异常
  4. raw 结果完整保留（union_dedup：id 不丢，score 不改）
  5. used_hyde / scope_mode 标记只在 HyDE 实际追加条目时才加 +hyde 后缀
  6. retrieve_history_items 在 hyde_enhancer=None 时走原有路径
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig, MemoryConfig
from agent.memory import MemoryStore
from agent.provider import LLMResponse
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from core.memory.port import DefaultMemoryPort
from memory2.hyde_enhancer import HyDEEnhancer, _union_dedup
from memory2.injection_planner import retrieve_history_items

# ── 测试工具 ──────────────────────────────────────────────────────────────────


class _NoopTool(Tool):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "noop"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs) -> str:
        return "ok"


class _FakeProvider:
    def __init__(self, response: str = "") -> None:
        self._response = response

    async def chat(self, **kwargs):
        return LLMResponse(content=self._response, tool_calls=[])


def _make_loop(
    light_model: str = "qwen-flash",
    *,
    memory_hyde_enabled: bool = False,
    memory_hyde_timeout_ms: int = 2000,
    **_unused: Any,
) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(_NoopTool())
    workspace = Path(tempfile.mkdtemp(prefix="hyde-test-"))
    provider = cast(Any, _FakeProvider())
    light_provider = cast(Any, _FakeProvider())
    return AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=provider,
            light_provider=light_provider,
            tools=tools,
            session_manager=MagicMock(),
            workspace=workspace,
            memory_port=DefaultMemoryPort(MemoryStore(workspace)),
        ),
        AgentLoopConfig(
            llm=LLMConfig(light_model=light_model),
            memory=MemoryConfig(
                hyde_enabled=memory_hyde_enabled,
                hyde_timeout_ms=memory_hyde_timeout_ms,
            ),
        ),
    )


# ── 1. HyDE 关闭时不构建 enhancer ─────────────────────────────────────────────


def test_hyde_disabled_no_enhancer():
    loop = _make_loop(memory_hyde_enabled=False)
    assert loop._hyde_enhancer is None


# ── 2. 无 light_model 时 hyde_enabled=True 自动禁用（不回退主模型）─────────────


def test_hyde_enabled_without_light_model_is_disabled(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="agent.loop"):
        loop = _make_loop(light_model="", memory_hyde_enabled=True)
    assert loop._hyde_enhancer is None
    assert "HyDE 已自动禁用" in caplog.text


# ── 3. hypothesis 超时 → 降级返回 raw 结果，无异常 ────────────────────────────


def test_hypothesis_timeout_falls_back_to_raw():
    import asyncio

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


# ── 4. raw 结果完整保留（id 不丢，score 不变）────────────────────────────────


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


# ── 5. used_hyde / scope_mode 标记 ────────────────────────────────────────────


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
    assert len(results) == 2


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
    assert len(results) == 1


# ── 6. retrieve_history_items scope_mode 标记 ─────────────────────────────────


def test_scope_mode_no_hyde_suffix_when_enhancer_none():
    """hyde_enhancer=None 时 scope_mode 为 'global'，无 +hyde 后缀。"""
    raw_items = [{"id": "a", "score": 0.7}]

    memory = MagicMock()
    memory.retrieve_related = AsyncMock(return_value=raw_items)

    items, scope_mode = asyncio.run(
        retrieve_history_items(
            memory,
            "测试",
            memory_types=["event"],
            top_k=6,
            hyde_enhancer=None,
        )
    )

    assert scope_mode == "global"
    assert items == raw_items


def test_scope_mode_has_hyde_suffix_when_hyde_appended():
    """HyDE 实际追加条目时 scope_mode 为 'global+hyde'。"""
    raw_items = [{"id": "a", "score": 0.7}]
    hyde_items = [{"id": "b", "score": 0.8}]

    async def fake_chat(**kwargs):
        return LLMResponse(content="假想条目", tool_calls=[])

    call_count = 0

    async def fake_retrieve(query, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return raw_items
        return hyde_items

    provider = MagicMock()
    provider.chat = fake_chat

    enhancer = HyDEEnhancer(
        light_provider=cast(Any, provider),
        light_model="qwen-flash",
        timeout_s=2.0,
    )

    memory = MagicMock()
    memory.retrieve_related = fake_retrieve

    items, scope_mode = asyncio.run(
        retrieve_history_items(
            memory,
            "测试",
            memory_types=["event"],
            top_k=6,
            context="近期对话内容",
            hyde_enhancer=enhancer,
        )
    )

    assert scope_mode == "global+hyde"
    assert len(items) == 2


def test_scope_mode_global_fallback_without_hyde_suffix_in_scoped_fast_path():
    """prefer_scoped=True 的快速路径下，global fallback 不走 HyDE 增强。"""
    provider = MagicMock()
    provider.chat = AsyncMock(
        return_value=LLMResponse(content="不会被使用的假想条目", tool_calls=[])
    )
    enhancer = HyDEEnhancer(
        light_provider=cast(Any, provider),
        light_model="qwen-flash",
        timeout_s=2.0,
    )

    memory = MagicMock()
    memory.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    memory.retrieve_related_vec = AsyncMock(
        side_effect=[
            [],
            [{"id": "g1", "score": 0.8}],
        ]
    )

    items, scope_mode = asyncio.run(
        retrieve_history_items(
            memory,
            "测试",
            memory_types=["event"],
            top_k=6,
            prefer_scoped=True,
            scope_channel="telegram",
            scope_chat_id="123",
            allow_global=True,
            context="近期对话内容",
            hyde_enhancer=enhancer,
        )
    )

    assert scope_mode == "global-fallback"
    assert items == [{"id": "g1", "score": 0.8}]
    provider.chat.assert_not_called()


def test_scope_mode_no_hyde_suffix_when_hyde_disabled_by_config():
    """hyde_enabled=False 时 AgentLoop._hyde_enhancer 为 None。"""
    loop = _make_loop(
        light_model="qwen-flash",
        memory_hyde_enabled=False,
    )
    assert loop._hyde_enhancer is None

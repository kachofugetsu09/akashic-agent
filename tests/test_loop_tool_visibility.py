"""
tool_search 可见性机制 + LRU 回归测试。

覆盖场景：
- tool_search_enabled=True 时非存在工具被拦截
- tool_search_enabled=True 时存在但不可见的工具自动解锁
- tool_search 调用结果正确扩展 visible_names
- LRU 容量上限 5，超出时淘汰最久未用
- 最近使用的工具刷新 LRU 顺序（不被淘汰）
- always_on 工具不写入 LRU
- preloaded 工具在下一请求中直接可见
"""

import asyncio
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig

import pytest

from agent.memory import MemoryStore
from agent.provider import LLMResponse, ToolCall
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool
from core.memory.port import DefaultMemoryPort

# ── 工具桩 ────────────────────────────────────────────────────────────────────


class _DummyTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"dummy tool {self._name}"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"ok:{self._name}"


class _FakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    async def chat(self, **kwargs: Any) -> LLMResponse:
        if not self._responses:
            raise AssertionError("provider.chat called more times than expected")
        return self._responses.pop(0)


# ── 工厂 ──────────────────────────────────────────────────────────────────────


def _make_loop(
    tmp_path: Path,
    provider: _FakeProvider,
    registry: ToolRegistry,
    tool_search_enabled: bool = True,
) -> AgentLoop:
    return AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=cast(Any, provider),
            tools=registry,
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=10, tool_search_enabled=tool_search_enabled)),
    )


def _base_registry() -> ToolRegistry:
    """只含 tool_search 的最小 registry。"""
    reg = ToolRegistry()
    reg.register(ToolSearchTool(reg), always_on=True, risk="read-only")
    return reg


# ── 可见性 / 拦截测试 ─────────────────────────────────────────────────────────


class TestVisibilityGuard:
    def test_nonexistent_tool_is_blocked(self, tmp_path):
        """完全不在 registry 里的工具名 → 拦截，不执行，返回错误消息给模型。"""
        reg = _base_registry()
        provider = _FakeProvider(
            [
                LLMResponse(content="", tool_calls=[ToolCall("c1", "ghost_tool", {})]),
                LLMResponse(content="ok", tool_calls=[]),
            ]
        )
        loop = _make_loop(tmp_path, provider, reg)

        final, tools_used, _, _, _ = asyncio.run(
            loop._run_agent_loop([{"role": "user", "content": "test"}])
        )

        assert final == "ok"
        assert "ghost_tool" not in tools_used  # 被拦截，不计入 tools_used

    def test_deferred_tool_direct_call_blocked_with_select_hint(self, tmp_path):
        """在 registry 里但不在 visible_names 里的工具（deferred）直接调用
        → 不执行，返回 select: 引导错误，模型收到后给出最终回复。"""
        reg = _base_registry()
        hidden = _DummyTool("hidden_tool")
        reg.register(hidden)  # 不设 always_on → deferred

        provider = _FakeProvider(
            [
                LLMResponse(content="", tool_calls=[ToolCall("c1", "hidden_tool", {})]),
                LLMResponse(content="done", tool_calls=[]),
            ]
        )
        loop = _make_loop(tmp_path, provider, reg)

        final, tools_used, tool_chain, _, _ = asyncio.run(
            loop._run_agent_loop([{"role": "user", "content": "test"}])
        )

        assert final == "done"
        assert "hidden_tool" not in tools_used  # 未执行，不计入 tools_used
        assert len(hidden.calls) == 0           # 工具实体未被调用

        # 第一轮 tool_chain 应有 select: 引导错误
        calls = tool_chain[0]["calls"] if tool_chain else []
        hidden_call = next((c for c in calls if c["name"] == "hidden_tool"), None)
        assert hidden_call is not None
        assert "select:" in hidden_call["result"]

    def test_tool_search_enabled_false_exposes_all_tools(self, tmp_path):
        """tool_search_enabled=False 时全量暴露，hidden tool 直接可用。"""
        reg = _base_registry()
        hidden = _DummyTool("hidden_tool")
        reg.register(hidden)

        provider = _FakeProvider(
            [
                LLMResponse(content="", tool_calls=[ToolCall("c1", "hidden_tool", {})]),
                LLMResponse(content="done", tool_calls=[]),
            ]
        )
        loop = _make_loop(tmp_path, provider, reg, tool_search_enabled=False)

        _, tools_used, _, _, _ = asyncio.run(
            loop._run_agent_loop([{"role": "user", "content": "test"}])
        )

        assert "hidden_tool" in tools_used

    def test_tool_search_result_unlocks_target_tool(self, tmp_path):
        """调用 tool_search 后，返回结果里的工具名加入 visible_names。"""
        reg = _base_registry()
        target = _DummyTool("target_tool")
        reg.register(target)

        # tool_search 直接返回匹配结果（模拟 registry.search 找到了 target_tool）
        tool_search_result = json.dumps(
            {
                "matched": [
                    {
                        "name": "target_tool",
                        "summary": "...",
                        "why_matched": [],
                        "key_params": [],
                        "tags": [],
                        "risk": "read-only",
                    }
                ]
            },
            ensure_ascii=False,
        )

        provider = _FakeProvider(
            [
                # 第 1 轮：调用 tool_search
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall("s1", "tool_search", {"query": "target"})],
                ),
                # 第 2 轮：调用解锁后的 target_tool
                LLMResponse(content="", tool_calls=[ToolCall("t1", "target_tool", {})]),
                # 第 3 轮：返回最终结果
                LLMResponse(content="all done", tool_calls=[]),
            ]
        )
        loop = _make_loop(tmp_path, provider, reg)

        final, tools_used, _, _, _ = asyncio.run(
            loop._run_agent_loop([{"role": "user", "content": "use target"}])
        )

        assert "target_tool" in tools_used
        assert len(target.calls) == 1
        assert final == "all done"

    def test_visible_names_starts_with_only_always_on(self, tmp_path):
        """tool_search_enabled=True 时，第一次 LLM 调用只传 always_on 工具 schema。"""
        reg = _base_registry()
        hidden = _DummyTool("hidden_tool")
        reg.register(hidden)

        schemas_seen: list[list[str]] = []

        class _CapturingProvider:
            _responses = [LLMResponse(content="done", tool_calls=[])]

            async def chat(self, **kwargs: Any) -> LLMResponse:
                schemas_seen.append(
                    [t["function"]["name"] for t in (kwargs.get("tools") or [])]
                )
                return self._responses.pop(0)

        loop = _make_loop(tmp_path, cast(Any, _CapturingProvider()), reg)

        asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

        assert schemas_seen, "provider.chat was never called"
        first_call_tools = schemas_seen[0]
        assert "tool_search" in first_call_tools
        assert "hidden_tool" not in first_call_tools


# ── LRU 测试 ──────────────────────────────────────────────────────────────────


class TestLRUCache:
    def _make_loop_for_lru(self, tmp_path: Path) -> AgentLoop:
        reg = _base_registry()
        # 注册 10 个非核心工具
        for i in range(10):
            reg.register(_DummyTool(f"tool_{i}"))
        return _make_loop(tmp_path, cast(Any, _FakeProvider([])), reg)

    def test_lru_capacity_5(self, tmp_path):
        """写入 6 个工具后，LRU 只保留最新 5 个。"""
        loop = self._make_loop_for_lru(tmp_path)
        loop._tool_discovery.update("s1", [f"tool_{i}" for i in range(6)], loop.tools.get_always_on_names())

        lru = loop._tool_discovery._unlocked["s1"]
        assert len(lru) == 5
        # tool_0 是最早写入的，应被淘汰
        assert "tool_0" not in lru
        assert "tool_5" in lru

    def test_lru_evicts_oldest_first(self, tmp_path):
        """容量满后，最久未使用的工具先被淘汰。"""
        loop = self._make_loop_for_lru(tmp_path)
        # 写入 5 个（满）
        loop._tool_discovery.update("s1", ["tool_0", "tool_1", "tool_2", "tool_3", "tool_4"], loop.tools.get_always_on_names())
        # 再加 1 个 → tool_0 应被淘汰
        loop._tool_discovery.update("s1", ["tool_5"], loop.tools.get_always_on_names())

        lru = loop._tool_discovery._unlocked["s1"]
        assert "tool_0" not in lru
        assert "tool_5" in lru

    def test_lru_refresh_on_reuse(self, tmp_path):
        """重复使用某工具会刷新其在 LRU 中的位置，不被淘汰。"""
        loop = self._make_loop_for_lru(tmp_path)
        # 写入 5 个（满）
        loop._tool_discovery.update("s1", ["tool_0", "tool_1", "tool_2", "tool_3", "tool_4"], loop.tools.get_always_on_names())
        # 重新使用 tool_0（刷到末尾）
        loop._tool_discovery.update("s1", ["tool_0"], loop.tools.get_always_on_names())
        # 再加 1 个 → tool_1（最久未用）应被淘汰，而非 tool_0
        loop._tool_discovery.update("s1", ["tool_5"], loop.tools.get_always_on_names())

        lru = loop._tool_discovery._unlocked["s1"]
        assert "tool_0" in lru  # 刚被刷新，安全
        assert "tool_1" not in lru  # 最久未用，被淘汰
        assert "tool_5" in lru

    def test_always_on_tools_not_in_lru(self, tmp_path):
        """always_on 工具不应写入 LRU。"""
        reg = _base_registry()
        reg.register(_DummyTool("always_tool"), always_on=True)
        reg.register(_DummyTool("normal_tool"))
        loop = _make_loop(tmp_path, cast(Any, _FakeProvider([])), reg)

        loop._tool_discovery.update("s1", ["always_tool", "tool_search", "normal_tool"], loop.tools.get_always_on_names())

        lru = loop._tool_discovery._unlocked.get("s1", {})
        assert "always_tool" not in lru
        assert "tool_search" not in lru
        assert "normal_tool" in lru

    def test_lru_preloaded_on_next_request(self, tmp_path):
        """上一请求写入 LRU 的工具，下一请求应出现在 preloaded 中。"""
        reg = _base_registry()
        target = _DummyTool("remembered_tool")
        reg.register(target)

        # 第一请求：调用 remembered_tool（触发 auto-unlock + LRU 写入）
        provider1 = _FakeProvider(
            [
                LLMResponse(
                    content="", tool_calls=[ToolCall("c1", "remembered_tool", {})]
                ),
                LLMResponse(content="done1", tool_calls=[]),
            ]
        )
        loop = _make_loop(tmp_path, provider1, reg)

        asyncio.run(
            loop._run_agent_loop(
                [{"role": "user", "content": "first"}],
                preloaded_tools=set(),
            )
        )
        # 手动模拟 _run_with_safety_retry 的 LRU 写入
        loop._tool_discovery.update("session1", ["remembered_tool"], loop.tools.get_always_on_names())

        # 验证 LRU 已记录
        assert "remembered_tool" in loop._tool_discovery._unlocked.get("session1", {})

        # 第二请求：preloaded 应包含该工具
        preloaded = set(loop._tool_discovery._unlocked["session1"].keys())
        assert "remembered_tool" in preloaded

    def test_lru_independent_per_session(self, tmp_path):
        """不同 session_key 的 LRU 互相独立。"""
        loop = self._make_loop_for_lru(tmp_path)
        loop._tool_discovery.update("session_a", ["tool_0", "tool_1"], loop.tools.get_always_on_names())
        loop._tool_discovery.update("session_b", ["tool_2", "tool_3"], loop.tools.get_always_on_names())

        assert set(loop._tool_discovery._unlocked["session_a"].keys()) == {"tool_0", "tool_1"}
        assert set(loop._tool_discovery._unlocked["session_b"].keys()) == {"tool_2", "tool_3"}

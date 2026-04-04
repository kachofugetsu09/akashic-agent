"""
tool_search 路由机制单元测试。

覆盖两条 runtime 行为：

  U1  unknown_tool_error_hint
      工具不在 registry → 错误消息包含可用作 tool_search query 的关键词

  U2  known_invisible_auto_unlock_no_search
      工具在 registry 但不在 visible_names → 自动解锁执行，tool_search 不被调用
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig
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
            raise AssertionError("provider.chat 被调用次数超过预期")
        return self._responses.pop(0)


# ── 工厂 ──────────────────────────────────────────────────────────────────────


def _make_loop(
    tmp_path: Path,
    provider: _FakeProvider,
    registry: ToolRegistry,
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
        AgentLoopConfig(llm=LLMConfig(max_iterations=10, tool_search_enabled=True)),
    )


def _base_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSearchTool(reg), always_on=True, risk="read-only")
    return reg


# ── U1: 未知工具错误消息含 query hint ─────────────────────────────────────────


class TestUnknownToolErrorHint:
    def test_error_message_contains_suggested_query(self, tmp_path: Path) -> None:
        """U1: 模型幻觉调 'rss_manage'（不在 registry 且不在 visible_names），
        错误消息应引导使用 select: 加载，并包含工具名和 tool_search 关键词。
        """
        reg = _base_registry()
        # provider: 第一步幻觉调用不存在工具，第二步收到错误后直接结束
        provider = _FakeProvider(
            [
                LLMResponse(content="", tool_calls=[ToolCall("c1", "rss_manage", {})]),
                LLMResponse(content="好的", tool_calls=[]),
            ]
        )
        loop = _make_loop(tmp_path, provider, reg)

        # 捕获工具调用结果（通过查 tool_chain）
        _, _, tool_chain, _, _ = asyncio.run(
            loop._run_agent_loop([{"role": "user", "content": "管理RSS"}])
        )

        # 找到 rss_manage 的调用结果
        error_result = None
        for step in tool_chain:
            for call in step.get("calls", []):
                if call["name"] == "rss_manage":
                    error_result = call["result"]
                    break

        assert error_result is not None, "rss_manage 调用记录不存在"
        # 新语义：错误消息应包含 select:rss_manage 的加载引导
        assert "select:rss_manage" in error_result, (
            f"错误消息未包含 select: 加载引导，当前消息：{error_result!r}"
        )
        assert "tool_search" in error_result, (
            f"错误消息未引导调用 tool_search，当前消息：{error_result!r}"
        )


# ── U2: deferred 工具直调 → 不执行，返回 select: 引导 ────────────────────────


class TestKnownInvisibleAutoUnlock:
    def test_deferred_direct_call_blocked_with_select_hint(self, tmp_path: Path) -> None:
        """U2: 模型直接调用在 registry 但不在 visible_names 的工具（deferred），
        runtime 不应自动解锁执行，应返回 select: 引导错误，工具实体不被调用。
        """
        reg = _base_registry()
        hidden = _DummyTool("schedule")
        reg.register(
            hidden,
            risk="write",
        )

        provider = _FakeProvider(
            [
                # 模型直接调 schedule（deferred 工具，未经 tool_search 加载）
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall("c1", "schedule", {"action": "remind", "at": "08:00"})],
                ),
                LLMResponse(content="已设置提醒", tool_calls=[]),
            ]
        )
        loop = _make_loop(tmp_path, provider, reg)

        _, tools_used, tool_chain, _, _ = asyncio.run(
            loop._run_agent_loop([{"role": "user", "content": "明天8点提醒我"}])
        )

        assert "schedule" not in tools_used, "deferred 工具未加载，不应计入 tools_used"
        assert len(hidden.calls) == 0, "deferred 工具未加载，不应被调用"

        # 错误消息应包含 select: 加载引导
        calls = tool_chain[0]["calls"] if tool_chain else []
        schedule_call = next((c for c in calls if c["name"] == "schedule"), None)
        assert schedule_call is not None
        assert "select:schedule" in schedule_call["result"], (
            f"错误消息未含 select: 引导，当前消息：{schedule_call['result']!r}"
        )

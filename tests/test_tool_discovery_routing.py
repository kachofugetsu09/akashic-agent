"""未知工具调用的恢复提示合同。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent.context import ContextBuilder
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig
from agent.plugin_composition import LLMResponse, ToolCall
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool
from bus.queue import MessageBus
from tests.compaction_fakes import run_test_agent_loop
from tests.memory_fakes import FakeMemoryEngine
from tests.provider_fakes import ProviderContextBudgetStub


class _FakeProvider(ProviderContextBudgetStub):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    async def chat(self, **kwargs: Any) -> LLMResponse:
        if not self._responses:
            raise AssertionError("provider.chat 被调用次数超过预期")
        return self._responses.pop(0)


def _make_loop(
    tmp_path: Path,
    provider: _FakeProvider,
    registry: ToolRegistry,
) -> AgentLoop:
    return AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            tools=registry,
            session_manager=MagicMock(),
            workspace=tmp_path,
            context=ContextBuilder(tmp_path),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=10, tool_search_enabled=True)),
    )


def test_unknown_tool_error_contains_recovery_query(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSearchTool(registry),
        always_on=True,
        risk="read-only",
    )
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "rss_manage", {})],
            ),
            LLMResponse(content="好的", tool_calls=[]),
        ]
    )

    _, _, tool_chain, _, _ = asyncio.run(
        run_test_agent_loop(
            _make_loop(tmp_path, provider, registry),
            provider,
            [{"role": "user", "content": "管理RSS"}],
        )
    )

    calls = [
        call
        for step in tool_chain
        for call in step.get("calls", [])
        if call["name"] == "rss_manage"
    ]
    assert len(calls) == 1
    assert "select:rss_manage" in calls[0]["result"]
    assert "tool_search" in calls[0]["result"]

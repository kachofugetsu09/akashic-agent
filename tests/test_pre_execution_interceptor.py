import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from agent.looping.core import AgentLoop
from agent.memory import MemoryStore
from agent.provider import LLMResponse, ToolCall
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from core.memory.port import DefaultMemoryPort


class _DummyTool(Tool):
    def __init__(self, name: str = "web_fetch") -> None:
        self._name = name
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "dummy tool"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        }

    async def execute(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "fetched"


class _FakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _HintMemory:
    def __init__(self, items_per_call: list[list[dict]]) -> None:
        self._items_per_call = list(items_per_call)

    def keyword_match_procedures(self, action_tokens: list[str]) -> list[dict]:
        if not self._items_per_call:
            return []
        return list(self._items_per_call.pop(0))


def _make_loop(
    tmp_path: Path,
    provider: _FakeProvider,
    tool: Tool,
    memory: _HintMemory | None = None,
) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(tool)
    return AgentLoop(
        bus=MagicMock(),
        provider=cast(Any, provider),
        tools=tools,
        session_manager=MagicMock(),
        workspace=tmp_path,
        max_iterations=5,
        memory_port=cast(
            Any,
            memory or DefaultMemoryPort(MemoryStore(tmp_path)),
        ),
    )


def test_tool_not_executed_when_intercept_hit(tmp_path: Path):
    tool = _DummyTool()
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("c1", "web_fetch", {"url": "https://www.bilibili.com/video/BV1"})
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    memory = _HintMemory(
        [[{"id": "p1", "summary": "B站链接必须改用 yt-dlp", "intercept": True}]]
    )
    loop = _make_loop(tmp_path, provider, tool, memory)

    asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

    assert tool.calls == []


def test_intercept_result_contains_not_executed_label(tmp_path: Path):
    tool = _DummyTool()
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("c1", "web_fetch", {"url": "https://www.bilibili.com/video/BV1"})
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    memory = _HintMemory(
        [[{"id": "p1", "summary": "B站链接必须改用 yt-dlp", "intercept": True}]]
    )
    loop = _make_loop(tmp_path, provider, tool, memory)

    asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

    tool_message = [m for m in provider.calls[1]["messages"] if m.get("role") == "tool"][0]
    assert "工具未执行" in tool_message["content"]
    assert "执行拦截" in tool_message["content"]


def test_non_intercept_procedure_still_executes(tmp_path: Path):
    tool = _DummyTool()
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("c1", "web_fetch", {"url": "https://www.bilibili.com/video/BV1"})
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    memory = _HintMemory([[{"id": "p1", "summary": "下载前先确认清晰度", "intercept": False}]])
    loop = _make_loop(tmp_path, provider, tool, memory)

    asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

    assert len(tool.calls) == 1


def test_intercept_ids_deduped_across_iterations(tmp_path: Path):
    tool = _DummyTool()
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("c1", "web_fetch", {"url": "https://www.bilibili.com/video/BV1"})
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("c2", "web_fetch", {"url": "https://www.bilibili.com/video/BV1"})
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    memory = _HintMemory(
        [
            [{"id": "p1", "summary": "B站链接必须改用 yt-dlp", "intercept": True}],
            [{"id": "p1", "summary": "B站链接必须改用 yt-dlp", "intercept": True}],
            [],
        ]
    )
    loop = _make_loop(tmp_path, provider, tool, memory)

    asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

    first_tool = [m for m in provider.calls[1]["messages"] if m.get("role") == "tool"][0]
    second_tool = [m for m in provider.calls[2]["messages"] if m.get("role") == "tool"][-1]
    assert "工具未执行" in first_tool["content"]
    assert second_tool["content"] == "fetched"
    assert len(tool.calls) == 1

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig
from agent.memory import MemoryStore
from agent.provider import LLMResponse, ToolCall
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from core.memory.port import DefaultMemoryPort


class _DummyTool(Tool):
    def __init__(self, name: str = "shell") -> None:
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
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def execute(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "tool output"


class _FakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _HintMemory:
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def keyword_match_procedures(self, action_tokens: list[str]) -> list[dict]:
        return list(self._items)


def _make_loop(
    tmp_path: Path,
    provider: _FakeProvider,
    tool: Tool,
    memory: _HintMemory | None = None,
) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(tool)
    return AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=cast(Any, provider),
            tools=tools,
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_port=cast(
                Any,
                memory or DefaultMemoryPort(MemoryStore(tmp_path)),
            ),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=5)),
    )


def test_tool_result_is_clean_after_5a(tmp_path: Path):
    tool = _DummyTool()
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "shell", {"command": "pacman -S jq"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    memory = _HintMemory([{"id": "p1", "summary": "pacman 调用时必须加 --noconfirm"}])
    loop = _make_loop(tmp_path, provider, tool, memory)

    asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

    tool_messages = [m for m in provider.calls[1]["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "tool output"
    assert "【操作规范提醒】" not in tool_messages[0]["content"]


def test_hint_appears_in_reflect_prompt(tmp_path: Path):
    tool = _DummyTool()
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "shell", {"command": "pacman -S jq"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    memory = _HintMemory([{"id": "p1", "summary": "pacman 调用时必须加 --noconfirm"}])
    loop = _make_loop(tmp_path, provider, tool, memory)

    asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

    reflect_messages = [m for m in provider.calls[1]["messages"] if m.get("role") == "system"]
    assert "【⚠️ 操作规范提醒 | 适用于本轮工具调用】" in reflect_messages[-1]["content"]
    assert "--noconfirm" in reflect_messages[-1]["content"]


def test_hint_header_is_distinct_from_tool_output(tmp_path: Path):
    tool = _DummyTool()
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "shell", {"command": "pacman -S jq"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    memory = _HintMemory([{"id": "p1", "summary": "pacman 调用时必须加 --noconfirm"}])
    loop = _make_loop(tmp_path, provider, tool, memory)

    asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

    tool_message = [m for m in provider.calls[1]["messages"] if m.get("role") == "tool"][0]
    reflect_message = [m for m in provider.calls[1]["messages"] if m.get("role") == "system"][-1]
    assert tool_message["content"] == "tool output"
    assert "【⚠️ 操作规范提醒 | 适用于本轮工具调用】" in reflect_message["content"]


def test_no_hint_when_no_matching_procedure(tmp_path: Path):
    tool = _DummyTool()
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "shell", {"command": "echo hi"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    loop = _make_loop(tmp_path, provider, tool, _HintMemory([]))

    asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

    reflect_message = [m for m in provider.calls[1]["messages"] if m.get("role") == "system"][-1]
    assert "【⚠️ 操作规范提醒 | 适用于本轮工具调用】" not in reflect_message["content"]

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from agent.prompting import is_context_frame

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


def _make_loop(
    tmp_path: Path,
    provider: _FakeProvider,
    tool: Tool,
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
            memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=5)),
    )


def test_reflect_prompt_no_longer_contains_procedure_hint(tmp_path: Path):
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
    loop = _make_loop(tmp_path, provider, tool)

    asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

    reflect_msgs = provider.calls[1]["messages"]
    all_content = " ".join(str(m.get("content", "")) for m in reflect_msgs)
    assert "【⚠️ 操作规范提醒 | 适用于本轮工具调用】" not in all_content
    # context frame 应作为 user 消息存在
    context_frame_msgs = [
        m for m in reflect_msgs
        if m.get("role") == "user"
        and is_context_frame(str(m.get("content", "")))
    ]
    assert len(context_frame_msgs) > 0, "expected at least one context frame user message"

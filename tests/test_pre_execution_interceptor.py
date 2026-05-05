import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig, MemoryServices
from agent.provider import LLMResponse, ToolCall
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from tests.memory_fakes import FakeMemoryEngine


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
            memory_services=MemoryServices(engine=FakeMemoryEngine(tmp_path)),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=5)),
    )

def test_tool_executes_without_procedure_interceptor(tmp_path: Path):
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
    loop = _make_loop(tmp_path, provider, tool)

    asyncio.run(loop._run_agent_loop([{"role": "user", "content": "test"}]))

    assert len(tool.calls) == 1

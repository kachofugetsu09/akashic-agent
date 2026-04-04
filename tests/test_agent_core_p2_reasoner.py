import asyncio
from typing import Any, cast

from agent.core.reasoner import DefaultReasoner
from agent.core.runtime_support import LLMServices, ToolDiscoveryState
from agent.looping.ports import LLMConfig
from agent.provider import LLMResponse, ToolCall
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool


class _DummyTool(Tool):
    def __init__(self, name: str = "dummy") -> None:
        self._name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"{self._name}-ok"


class _Provider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("provider.chat called more than expected")
        return self._responses.pop(0)


def test_default_reasoner_runs_tool_loop_and_returns_reasoner_result():
    provider = _Provider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider)),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        memory_port=cast(Any, type("_M", (), {"keyword_match_procedures": lambda self, _: []})()),
        tool_search_enabled=False,
    )

    result = asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    assert result.reply == "final"
    assert result.metadata["tools_used"] == ["dummy"]
    assert result.invocations[0].name == "dummy"
    assert result.metadata["visible_names"] is None


def test_default_reasoner_unlocks_tool_search_visibility():
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("s1", "tool_search", {"query": "hidden"})],
            ),
            LLMResponse(content="", tool_calls=[ToolCall("h1", "hidden_tool", {})]),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    hidden = _DummyTool("hidden_tool")
    tools.register(hidden)
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider)),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        memory_port=cast(Any, type("_M", (), {"keyword_match_procedures": lambda self, _: []})()),
        tool_search_enabled=True,
    )

    result = asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    assert result.reply == "done"
    assert "hidden_tool" in result.metadata["tools_used"]
    assert "hidden_tool" in result.metadata["visible_names"]
    assert len(hidden.calls) == 1

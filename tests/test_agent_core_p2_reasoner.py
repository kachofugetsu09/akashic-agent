import asyncio
from datetime import datetime
from types import SimpleNamespace
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
        memory_window=40,
    )

    result = asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    assert result.reply == "final"
    assert result.metadata["tools_used"] == ["dummy"]
    assert result.invocations[0].name == "dummy"
    assert result.metadata["visible_names"] is None
    first_messages = provider.calls[0]["messages"]
    assert any("本轮时间锚点" in str(m.get("content", "")) for m in first_messages)
    assert any(
        m.get("role") == "system" and "本轮时间锚点" in str(m.get("content", ""))
        for m in first_messages
    )


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
        memory_window=40,
    )

    result = asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    assert result.reply == "done"
    assert "hidden_tool" in result.metadata["tools_used"]
    assert "hidden_tool" in result.metadata["visible_names"]
    assert len(hidden.calls) == 1


def test_default_reasoner_preflight_includes_deferred_tool_names():
    provider = _Provider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    tools.register(
        _DummyTool("mcp_github__list_commits"),
        source_type="mcp",
        source_name="github",
    )
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider)),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        memory_port=cast(Any, type("_M", (), {"keyword_match_procedures": lambda self, _: []})()),
        tool_search_enabled=True,
        memory_window=40,
    )

    asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    first_messages = provider.calls[0]["messages"]
    preflight = next(
        str(m.get("content", ""))
        for m in first_messages
        if "本轮时间锚点" in str(m.get("content", ""))
    )
    assert "未加载工具目录" in preflight
    assert "mcp_github__list_commits" in preflight
    assert "dummy" not in preflight


def test_default_reasoner_deferred_tool_direct_call_requires_select():
    provider = _Provider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "schedule", {})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    tools.register(_DummyTool("schedule"))
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider)),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        memory_port=cast(Any, type("_M", (), {"keyword_match_procedures": lambda self, _: []})()),
        tool_search_enabled=True,
        memory_window=40,
    )

    result = asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    assert "schedule" not in result.metadata["tools_used"]
    assert result.reply == "final"
    tool_chain = list(result.metadata["tool_chain"])
    assert len(tool_chain) >= 1
    schedule_call = next((c for c in tool_chain[0]["calls"] if c["name"] == "schedule"), None)
    assert schedule_call is not None
    assert "select:" in schedule_call["result"]
    assert "tool_search" in schedule_call["result"]


def test_default_reasoner_preloaded_tool_not_in_deferred_list():
    provider = _Provider([LLMResponse(content="done", tool_calls=[])])
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    tools.register(_DummyTool("schedule"))
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider)),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        memory_port=cast(Any, type("_M", (), {"keyword_match_procedures": lambda self, _: []})()),
        tool_search_enabled=True,
        memory_window=40,
    )

    asyncio.run(
        reasoner.run(
            [{"role": "user", "content": "hi"}],
            preloaded_tools={"schedule"},
        )
    )

    first_messages = provider.calls[0]["messages"]
    preflight = next(
        str(m.get("content", ""))
        for m in first_messages
        if "本轮时间锚点" in str(m.get("content", ""))
    )
    assert "schedule" not in preflight


def test_default_reasoner_run_turn_uses_context_render():
    provider = _Provider([LLMResponse(content="done", tool_calls=[])])
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider)),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        memory_port=cast(Any, type("_M", (), {"keyword_match_procedures": lambda self, _: []})()),
        tool_search_enabled=False,
        memory_window=40,
        context=cast(Any, SimpleNamespace(
            render=lambda request: SimpleNamespace(
                messages=[{"role": "user", "content": request.current_message}],
            ),
            build_messages=lambda **_: (_ for _ in ()).throw(AssertionError("legacy build_messages should not be used")),
            build_runtime_guard_context=lambda **_: (_ for _ in ()).throw(AssertionError("legacy runtime_guard should not be used")),
        )),
        session_manager=cast(Any, SimpleNamespace(save_async=lambda *_args, **_kwargs: None)),
    )

    session = SimpleNamespace(
        key="cli:1",
        messages=[{"role": "assistant", "content": "old"}],
        get_history=lambda max_messages=40: [{"role": "assistant", "content": "old"}],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="hi",
        media=[],
        channel="cli",
        chat_id="1",
        timestamp=datetime(2026, 4, 5, 12, 0, 0),
    )

    result = asyncio.run(reasoner.run_turn(msg=msg, session=session))

    assert result.reply == "done"

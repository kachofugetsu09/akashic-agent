import asyncio
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import LLMServices, ToolDiscoveryState
from agent.looping.ports import LLMConfig
from agent.provider import LLMResponse, ToolCall
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool
from bus.event_bus import EventBus
from bus.events_lifecycle import ToolCallCompleted, ToolCallStarted


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


class _TimeoutProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        raise asyncio.TimeoutError


def test_default_reasoner_runs_tool_loop_and_returns_reasoner_result():
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "dummy", {})],
                cache_prompt_tokens=100,
                cache_hit_tokens=40,
            ),
            LLMResponse(
                content="final",
                tool_calls=[],
                cache_prompt_tokens=120,
                cache_hit_tokens=60,
            ),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = DefaultReasoner(
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=40,
    )

    result = asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    assert result.reply == "final"
    assert result.metadata["tools_used"] == ["dummy"]
    assert result.invocations[0].name == "dummy"
    assert result.metadata["visible_names"] is None
    react_stats = result.metadata["react_stats"]
    assert react_stats["iteration_count"] == 2
    assert react_stats["turn_input_sum_tokens"] >= react_stats["turn_input_peak_tokens"]
    assert react_stats["final_call_input_tokens"] == react_stats["turn_input_peak_tokens"]
    assert react_stats["cache_prompt_tokens"] == 220
    assert react_stats["cache_hit_tokens"] == 100
    first_messages = provider.calls[0]["messages"]
    assert not any("未加载工具目录" in str(m.get("content", "")) for m in first_messages)


def test_default_reasoner_observes_tool_lifecycle_events():
    provider = _Provider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {"x": 7})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tool = _DummyTool()
    tools.register(tool, always_on=True)
    event_bus = EventBus()
    order: list[str] = []
    started_events: list[ToolCallStarted] = []
    completed_events: list[ToolCallCompleted] = []
    event_bus.on(
        ToolCallStarted,
        lambda event: order.append("started") or started_events.append(event),
    )
    event_bus.on(
        ToolCallCompleted,
        lambda event: order.append("completed") or completed_events.append(event),
    )
    reasoner = DefaultReasoner(
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=40,
        context=cast(Any, SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                messages=[{"role": "user", "content": request.current_message}],
            ),
        )),
        session_manager=cast(Any, SimpleNamespace()),
        event_bus=event_bus,
    )
    session = SimpleNamespace(
        key="telegram:123",
        messages=[],
        get_history=lambda max_messages=40: [],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="hi",
        media=[],
        channel="telegram",
        chat_id="123",
        timestamp=datetime(2026, 4, 5, 12, 0, 0),
    )

    result = asyncio.run(reasoner.run_turn(msg=msg, session=cast(Any, session)))

    assert result.reply == "final"
    assert order == ["started", "completed"]
    assert started_events[0].session_key == "telegram:123"
    assert started_events[0].channel == "telegram"
    assert started_events[0].chat_id == "123"
    assert started_events[0].iteration == 1
    assert started_events[0].call_id == "c1"
    assert started_events[0].tool_name == "dummy"
    assert started_events[0].arguments == {"x": 7}
    assert completed_events[0].session_key == "telegram:123"
    assert completed_events[0].call_id == "c1"
    assert completed_events[0].tool_name == "dummy"
    assert completed_events[0].arguments == {"x": 7}
    assert completed_events[0].final_arguments == {"x": 7}
    assert completed_events[0].status == "success"
    assert completed_events[0].result_preview == "dummy-ok"


def test_default_reasoner_observes_blocked_tool_lifecycle_events():
    provider = _Provider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "hidden_tool", {"x": 1})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    hidden = _DummyTool("hidden_tool")
    tools.register(hidden)
    event_bus = EventBus()
    order: list[str] = []
    started_events: list[ToolCallStarted] = []
    completed_events: list[ToolCallCompleted] = []
    event_bus.on(
        ToolCallStarted,
        lambda event: order.append("started") or started_events.append(event),
    )
    event_bus.on(
        ToolCallCompleted,
        lambda event: order.append("completed") or completed_events.append(event),
    )
    reasoner = DefaultReasoner(
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
        memory_window=40,
        event_bus=event_bus,
    )

    result = asyncio.run(
        reasoner.run(
            [{"role": "user", "content": "hi"}],
            tool_event_session_key="telegram:123",
            tool_event_channel="telegram",
            tool_event_chat_id="123",
        )
    )

    assert result.reply == "final"
    assert hidden.calls == []
    assert order == ["started", "completed"]
    assert started_events[0].tool_name == "hidden_tool"
    assert started_events[0].arguments == {"x": 1}
    assert completed_events[0].tool_name == "hidden_tool"
    assert completed_events[0].arguments == {"x": 1}
    assert completed_events[0].final_arguments == {"x": 1}
    assert completed_events[0].status == "blocked"
    assert "select:hidden_tool" in completed_events[0].result_preview


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
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
        memory_window=40,
    )

    result = asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    assert result.reply == "done"
    assert "hidden_tool" in result.metadata["tools_used"]
    assert "hidden_tool" in result.metadata["visible_names"]
    assert len(hidden.calls) == 1


def test_default_reasoner_preflight_includes_deferred_tool_names():
    """调用方（如 _run_agent_loop）负责注入 deferred tools hint；run() 本身不再自动注入。"""
    from agent.core.passive_turn import build_turn_injection_prompt
    from agent.prompting import build_context_frame_content, build_context_frame_message
    from agent.prompting import PromptSectionRender

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
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
        memory_window=40,
    )

    # 调用方负责在调用 run() 前注入 hint。
    hint = build_turn_injection_prompt(
        tools=tools,
        tool_search_enabled=True,
        visible_names=tools.get_always_on_names(),
    )
    frame_content = build_context_frame_content(
        [PromptSectionRender(name="tool_hint", content=hint, is_static=False)]
    )
    initial_messages = [
        build_context_frame_message(frame_content),
        {"role": "user", "content": "hi"},
    ]
    asyncio.run(reasoner.run(initial_messages))

    first_messages = provider.calls[0]["messages"]
    preflight = next(
        str(m.get("content", ""))
        for m in first_messages
        if "未加载工具目录" in str(m.get("content", ""))
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
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
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
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
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
    assert not any("未加载工具目录" in str(m.get("content", "")) for m in first_messages)


def test_default_reasoner_run_turn_uses_context_render():
    provider = _Provider([LLMResponse(content="done", tool_calls=[])])
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = DefaultReasoner(
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=40,
        context=cast(Any, SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                messages=[{"role": "user", "content": request.current_message}],
            ),
            build_messages=lambda **_: (_ for _ in ()).throw(AssertionError("legacy build_messages should not be used")),
            build_turn_injection_context=lambda **_: (_ for _ in ()).throw(AssertionError("legacy turn_injection should not be used")),
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

    result = asyncio.run(reasoner.run_turn(msg=msg, session=cast(Any, session)))

    assert result.reply == "done"


def test_default_reasoner_run_turn_reports_llm_timeout():
    provider = _TimeoutProvider()
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = DefaultReasoner(
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=40,
        context=cast(Any, SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                messages=[{"role": "user", "content": request.current_message}],
            ),
        )),
        session_manager=cast(Any, SimpleNamespace(save_async=lambda *_args, **_kwargs: None)),
    )
    session = SimpleNamespace(
        key="cli:1",
        messages=[],
        get_history=lambda max_messages=40: [],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="hi",
        media=[],
        channel="cli",
        chat_id="1",
        timestamp=datetime(2026, 4, 5, 12, 0, 0),
    )

    result = asyncio.run(reasoner.run_turn(msg=msg, session=cast(Any, session)))

    assert result.reply == "模型流响应中断，请刷新对话重试。"
    assert len(provider.calls) == 1


def test_empty_content_with_thinking_triggers_retry_and_succeeds():
    provider = _Provider(
        [
            LLMResponse(content=None, tool_calls=[], thinking="长思考过程"),
            LLMResponse(content="正式回复", tool_calls=[], thinking="新思考"),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = DefaultReasoner(
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=40,
    )

    result = asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    assert result.reply == "正式回复"
    assert result.thinking == "新思考"
    retry_call = provider.calls[1]
    assert retry_call["tools"] == []
    assert len(provider.calls) == 2


def test_empty_content_with_thinking_retry_still_empty_falls_back():
    provider = _Provider(
        [
            LLMResponse(content=None, tool_calls=[], thinking="只有思考"),
            LLMResponse(content=None, tool_calls=[], thinking=None),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = DefaultReasoner(
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=40,
    )

    result = asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    assert result.reply == "（无响应）"
    assert result.thinking == "只有思考"
    assert len(provider.calls) == 2


def test_empty_content_without_thinking_no_retry():
    provider = _Provider(
        [
            LLMResponse(content=None, tool_calls=[], thinking=None),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = DefaultReasoner(
        llm=cast(Any, LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider))),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=40,
    )

    result = asyncio.run(reasoner.run([{"role": "user", "content": "hi"}]))

    assert result.reply == "（无响应）"
    assert len(provider.calls) == 1

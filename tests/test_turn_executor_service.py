import asyncio
from typing import Any, cast

from agent.looping.ports import LLMConfig, LLMServices
from agent.looping.tool_execution import ToolDiscoveryState, TurnExecutor
from agent.provider import LLMResponse, ToolCall
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry


class _DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "dummy"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        return "dummy-ok"


class _ScheduleTool(Tool):
    @property
    def name(self) -> str:
        return "schedule"

    @property
    def description(self) -> str:
        return "schedule a reminder"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        return "scheduled"


class _GithubTool(Tool):
    @property
    def name(self) -> str:
        return "mcp_github__list_commits"

    @property
    def description(self) -> str:
        return "github commits"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class _Provider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {})]),
            LLMResponse(content="final", tool_calls=[]),
        ]

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_turn_executor_runs_tool_then_returns_final():
    provider = _Provider()
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)

    executor = TurnExecutor(
        llm=LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider)),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        memory_port=cast(Any, type("_M", (), {"keyword_match_procedures": lambda self, _: []})()),
        tool_search_enabled=False,
    )

    content, tools_used, tool_chain, visible, thinking = asyncio.run(
        executor.execute([{"role": "user", "content": "hi"}])
    )

    assert content == "final"
    assert tools_used == ["dummy"]
    assert len(tool_chain) == 1
    assert visible is None
    assert thinking is None
    # 第一次调用时，preflight prompt 已追加到消息列表
    first_messages = provider.calls[0]["messages"]
    assert any("本轮时间锚点" in str(m.get("content", "")) for m in first_messages)


def test_turn_executor_preflight_includes_deferred_tool_names():
    provider = _Provider()
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    tools.register(
        _GithubTool(),
        source_type="mcp",
        source_name="github",
    )

    executor = TurnExecutor(
        llm=LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider)),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        memory_port=cast(Any, type("_M", (), {"keyword_match_procedures": lambda self, _: []})()),
        tool_search_enabled=True,
    )

    asyncio.run(executor.execute([{"role": "user", "content": "hi"}]))

    first_messages = provider.calls[0]["messages"]
    preflight = next(
        str(m.get("content", ""))
        for m in first_messages
        if "本轮时间锚点" in str(m.get("content", ""))
    )
    assert "未加载工具目录" in preflight
    assert "mcp_github__list_commits" in preflight
    assert "dummy" not in preflight


def test_deferred_tool_direct_call_requires_select_not_auto_unlock():
    """deferred 工具（非 always_on）被直接调用时，返回 select: 引导错误，不自动解锁执行。"""

    class _DirectCallProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._responses = [
                # 第一次：模型直接调用 schedule（deferred 工具，未通过 tool_search 加载）
                LLMResponse(content="", tool_calls=[ToolCall("c1", "schedule", {})]),
                # 第二次：收到错误后给出最终回复
                LLMResponse(content="final", tool_calls=[]),
            ]

        async def chat(self, **kwargs: Any):
            self.calls.append(kwargs)
            return self._responses.pop(0)

    provider = _DirectCallProvider()
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    tools.register(_ScheduleTool())  # 注册但非 always_on → deferred

    executor = TurnExecutor(
        llm=LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider)),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        memory_port=cast(Any, type("_M", (), {"keyword_match_procedures": lambda self, _: []})()),
        tool_search_enabled=True,
    )

    content, tools_used, tool_chain, visible, thinking = asyncio.run(
        executor.execute([{"role": "user", "content": "hi"}])
    )

    # schedule 不应被执行（未解锁）
    assert "schedule" not in tools_used
    assert content == "final"

    # 第一轮 tool_chain 应包含 schedule 调用及 select: 引导错误
    assert len(tool_chain) >= 1
    calls = tool_chain[0]["calls"]
    schedule_call = next((c for c in calls if c["name"] == "schedule"), None)
    assert schedule_call is not None
    assert "select:" in schedule_call["result"]
    assert "tool_search" in schedule_call["result"]


def test_deferred_tool_not_in_preflight_deferred_list_when_preloaded():
    """preloaded（LRU 上轮已用）工具不应出现在 preflight 的"未加载工具目录"中。"""

    class _PreloadedProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._responses = [LLMResponse(content="done", tool_calls=[])]

        async def chat(self, **kwargs: Any):
            self.calls.append(kwargs)
            return self._responses.pop(0)

    provider = _PreloadedProvider()
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    tools.register(_ScheduleTool())  # deferred

    executor = TurnExecutor(
        llm=LLMServices(provider=cast(Any, provider), light_provider=cast(Any, provider)),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        memory_port=cast(Any, type("_M", (), {"keyword_match_procedures": lambda self, _: []})()),
        tool_search_enabled=True,
    )

    # 以 schedule 作为 preloaded_tools（模拟上轮用过、LRU 预加载）
    asyncio.run(
        executor.execute(
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
    # schedule 已预加载，不应出现在"未加载工具目录"里
    assert "schedule" not in preflight

from typing import Any, cast

from agent.core import (
    ContextBundle,
    InboundMessage,
    LLMResponse,
    OutboundMessage,
    ReasonerResult,
    ToolCall,
    ToolDiscoveryState,
)
from agent.looping.ports import LLMServices


def test_agent_core_foundation_types_construct_cleanly():
    inbound = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="hello",
    )
    outbound = OutboundMessage(
        channel="cli",
        chat_id="1",
        content="ok",
    )
    bundle = ContextBundle(skill_mentions=["search"])
    response = LLMResponse(reply="done", tool_calls=[ToolCall(id="c1", name="dummy")])
    result = ReasonerResult(reply="done", tools_used=["dummy"])

    assert inbound.session_key == "cli:1"
    assert outbound.content == "ok"
    assert bundle.skill_mentions == ["search"]
    assert response.tool_calls[0].name == "dummy"
    assert result.tools_used == ["dummy"]


def test_agent_core_runtime_support_tool_discovery_lru():
    state = ToolDiscoveryState(capacity=2)
    state.update("cli:1", ["tool_a", "tool_b"], {"always"})
    assert state.get_preloaded("cli:1") == {"tool_a", "tool_b"}
    assert state.get_preloaded_ordered("cli:1") == ["tool_a", "tool_b"]

    state.update("cli:1", ["tool_a"], {"always"})
    state.update("cli:1", ["tool_c"], {"always"})

    assert state.get_preloaded("cli:1") == {"tool_a", "tool_c"}
    assert state.get_preloaded_ordered("cli:1") == ["tool_a", "tool_c"]
    assert "tool_b" not in state.get_preloaded("cli:1")


def test_agent_core_runtime_support_skips_always_on_and_tool_search():
    state = ToolDiscoveryState()
    state.update(
        "cli:1", ["always_tool", "tool_search", "hidden_tool"], {"always_tool"}
    )

    assert state.get_preloaded("cli:1") == {"hidden_tool"}


def test_agent_core_runtime_support_does_not_store_empty_session_cache():
    state = ToolDiscoveryState()

    state.update("cli:1", ["always_tool", "tool_search"], {"always_tool"})

    assert "cli:1" not in state._unlocked


def test_agent_core_runtime_support_bounds_session_cache():
    state = ToolDiscoveryState(session_capacity=2)
    state.update("cli:1", ["tool_a"], set())
    state.update("cli:2", ["tool_b"], set())

    assert state.get_preloaded("cli:1") == {"tool_a"}

    state.update("cli:3", ["tool_c"], set())

    assert "cli:2" not in state._unlocked
    assert state.get_preloaded("cli:1") == {"tool_a"}
    assert state.get_preloaded("cli:3") == {"tool_c"}


def test_agent_core_runtime_support_service_types_hold_objects():
    llm = LLMServices(
        provider=cast(Any, object()),
        light_provider=cast(Any, object()),
    )
    assert llm.provider is not None

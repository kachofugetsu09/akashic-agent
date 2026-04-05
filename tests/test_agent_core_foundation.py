from agent.core import (
    ChatMessage,
    ContextBundle,
    InboundMessage,
    LLMResponse,
    LLMServices,
    MemoryConfig,
    MemoryServices,
    OutboundMessage,
    ReasonerResult,
    ToolCall,
    ToolDiscoveryState,
    TurnRecord,
)


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
    bundle = ContextBundle(history=[ChatMessage(role="user", content="hi")])
    response = LLMResponse(reply="done", tool_calls=[ToolCall(id="c1", name="dummy")])
    result = ReasonerResult(reply="done", invocations=response.tool_calls)
    record = TurnRecord(msg=inbound, reply="done", invocations=response.tool_calls)

    assert inbound.session_key == "cli:1"
    assert outbound.content == "ok"
    assert bundle.history[0].content == "hi"
    assert response.tool_calls[0].name == "dummy"
    assert result.invocations[0].id == "c1"
    assert record.reply == "done"


def test_agent_core_runtime_support_tool_discovery_lru():
    state = ToolDiscoveryState(capacity=2)
    state.update("cli:1", ["tool_a", "tool_b"], {"always"})
    assert state.get_preloaded("cli:1") == {"tool_a", "tool_b"}

    state.update("cli:1", ["tool_a"], {"always"})
    state.update("cli:1", ["tool_c"], {"always"})

    assert state.get_preloaded("cli:1") == {"tool_a", "tool_c"}
    assert "tool_b" not in state.get_preloaded("cli:1")


def test_agent_core_runtime_support_skips_always_on_and_tool_search():
    state = ToolDiscoveryState()
    state.update("cli:1", ["always_tool", "tool_search", "hidden_tool"], {"always_tool"})

    assert state.get_preloaded("cli:1") == {"hidden_tool"}


def test_agent_core_runtime_support_service_types_hold_objects():
    llm = LLMServices(provider=object(), light_provider=object())
    memory = MemoryServices(port=object())
    config = MemoryConfig(window=12, top_k_history=3)

    assert llm.provider is not None
    assert memory.port is not None
    assert config.window == 12
    assert config.top_k_history == 3

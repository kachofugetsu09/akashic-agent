from agent.core.runtime_support import ToolDiscoveryState


def test_tool_discovery_state_keeps_most_recent_tools():
    state = ToolDiscoveryState(capacity=2)
    state.update("cli:1", ["tool_a", "tool_b"], {"always"})
    assert state.get_preloaded_ordered("cli:1") == ["tool_a", "tool_b"]

    state.update("cli:1", ["tool_a"], {"always"})
    state.update("cli:1", ["tool_c"], {"always"})

    assert state.get_preloaded_ordered("cli:1") == ["tool_a", "tool_c"]


def test_tool_discovery_state_skips_always_on_and_tool_search():
    state = ToolDiscoveryState()
    state.update(
        "cli:1", ["always_tool", "tool_search", "hidden_tool"], {"always_tool"}
    )

    assert state.get_preloaded_ordered("cli:1") == ["hidden_tool"]


def test_tool_discovery_state_does_not_store_empty_session_cache():
    state = ToolDiscoveryState()

    state.update("cli:1", ["always_tool", "tool_search"], {"always_tool"})

    assert "cli:1" not in state._unlocked


def test_tool_discovery_state_bounds_session_cache():
    state = ToolDiscoveryState(session_capacity=2)
    state.update("cli:1", ["tool_a"], set())
    state.update("cli:2", ["tool_b"], set())

    assert state.get_preloaded_ordered("cli:1") == ["tool_a"]

    state.update("cli:3", ["tool_c"], set())

    assert "cli:2" not in state._unlocked
    assert state.get_preloaded_ordered("cli:1") == ["tool_a"]
    assert state.get_preloaded_ordered("cli:3") == ["tool_c"]

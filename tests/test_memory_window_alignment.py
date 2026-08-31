from agent.looping.ports import AgentLoopConfig


def test_agent_loop_config_does_not_own_plugin_memory_policy() -> None:
    config = AgentLoopConfig()

    assert not hasattr(config, "context_compaction")
    assert not hasattr(config, "memory")

from agent.looping.ports import AgentLoopConfig


def test_agent_loop_config_owns_context_compaction_not_memory_window() -> None:
    config = AgentLoopConfig()

    assert config.context_compaction.keep_recent_tokens == 20_000
    assert not hasattr(config, "memory")

from agent.looping.ports import MemoryConfig


def test_memory_window_aligns_keep_count_to_context_frame_turns() -> None:
    assert MemoryConfig(window=2).keep_count == 3
    assert MemoryConfig(window=6).keep_count == 3
    assert MemoryConfig(window=24).keep_count == 12
    assert MemoryConfig(window=40).keep_count == 21
    assert MemoryConfig(window=43).keep_count == 24


def test_consolidation_threshold_uses_aligned_keep_count() -> None:
    config = MemoryConfig(window=40)

    assert config.keep_count == 21
    assert config.consolidation_min_new_messages == 10

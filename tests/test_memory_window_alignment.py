from agent.looping.ports import MemoryConfig


def test_memory_window_aligns_keep_count_to_context_frame_turns() -> None:
    assert MemoryConfig(window=2).keep_count == 2
    assert MemoryConfig(window=6).keep_count == 4
    assert MemoryConfig(window=24).keep_count == 12
    assert MemoryConfig(window=40).keep_count == 20
    assert MemoryConfig(window=43).keep_count == 22

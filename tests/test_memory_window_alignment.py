from agent.looping.ports import MemoryConfig


def test_memory_window_is_the_actual_even_message_limit() -> None:
    assert MemoryConfig().keep_count == 20
    assert MemoryConfig(window=2).keep_count == 2
    assert MemoryConfig(window=6).keep_count == 6
    assert MemoryConfig(window=24).keep_count == 24
    assert MemoryConfig(window=40).keep_count == 40
    assert MemoryConfig(window=43).keep_count == 44

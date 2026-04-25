import asyncio

from memory2.memorizer import Memorizer, _parse_history_entry_happened_at
from memory2.store import MemoryStore2


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_parse_history_entry_happened_at_from_prefix():
    assert (
        _parse_history_entry_happened_at("[2026-03-08 12:00] 用户确认信息")
        == "2026-03-08T12:00:00"
    )
    assert (
        _parse_history_entry_happened_at("[2026-03-08T12:01] 用户确认信息")
        == "2026-03-08T12:01:00"
    )
    assert (
        _parse_history_entry_happened_at("[2026-03-08] 用户确认信息")
        == "2026-03-08T00:00:00"
    )
    assert _parse_history_entry_happened_at("用户确认信息") is None


def test_upsert_consolidation_event_fills_missing_happened_at_on_duplicate(tmp_path):
    store = MemoryStore2(tmp_path / "memory2.db")

    store.upsert_consolidation_event(
        source_ref="session@1",
        summary="[2026-03-08 12:00] same",
        embedding=[0.1, 0.2, 0.3],
    )
    store.upsert_consolidation_event(
        source_ref="session@2",
        summary="[2026-03-08 12:00] same",
        embedding=[0.1, 0.2, 0.3],
        happened_at="2026-03-08T12:00:00",
    )

    items = store.list_by_type("event")
    assert len(items) == 1
    assert items[0]["happened_at"] == "2026-03-08T12:00:00"


def test_save_from_consolidation_writes_happened_at(tmp_path):
    store = MemoryStore2(tmp_path / "memory2.db")
    memorizer = Memorizer(store, _FakeEmbedder())

    async def _run() -> None:
        await memorizer.save_from_consolidation(
            history_entry="[2026-03-08 12:00] 用户确认信息",
            behavior_updates=[],
            source_ref="session@1-10",
            scope_channel="telegram",
            scope_chat_id="123",
        )

    asyncio.run(_run())

    items = store.list_by_type("event")
    assert len(items) == 1
    assert items[0]["happened_at"] == "2026-03-08T12:00:00"


def test_save_from_consolidation_skips_duplicate_source_ref(tmp_path):
    store = MemoryStore2(tmp_path / "memory2.db")
    memorizer = Memorizer(store, _FakeEmbedder())

    async def _run() -> None:
        await memorizer.save_from_consolidation(
            history_entry="[2026-03-08 12:00] second with different text",
            behavior_updates=[],
            source_ref="session@1-10",
            scope_channel="telegram",
            scope_chat_id="123",
        )
        await memorizer.save_from_consolidation(
            history_entry="[2026-03-08 12:00] first",
            behavior_updates=[],
            source_ref="session@1-10",
            scope_channel="telegram",
            scope_chat_id="123",
        )

    asyncio.run(_run())

    items = store.list_by_type("event")
    assert len(items) == 1
    assert items[0]["reinforcement"] == 1

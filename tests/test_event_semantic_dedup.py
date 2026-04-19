import asyncio
from datetime import datetime, timedelta, timezone

from memory2.memorizer import Memorizer
from memory2.store import MemoryStore2


class _FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        return list(self._mapping[text])


def test_near_duplicate_event_not_saved_again(tmp_path):
    store = MemoryStore2(tmp_path / "memory2.db")
    embedder = _FakeEmbedder(
        {
            "用户把仓库脱敏后公开发布": [1.0, 0.0],
            "用户公开了脱敏后的仓库": [0.99, 0.01],
        }
    )
    memorizer = Memorizer(store, embedder)

    async def _run() -> None:
        await memorizer.save_from_consolidation(
            history_entry="用户把仓库脱敏后公开发布",
            behavior_updates=[],
            source_ref="session@1-10#0",
            scope_channel="telegram",
            scope_chat_id="1",
        )
        await memorizer.save_from_consolidation(
            history_entry="用户公开了脱敏后的仓库",
            behavior_updates=[],
            source_ref="session@1-10#1",
            scope_channel="telegram",
            scope_chat_id="1",
        )

    asyncio.run(_run())

    items = store.list_by_type("event")
    assert len(items) == 1


def test_distinct_event_saves_normally(tmp_path):
    store = MemoryStore2(tmp_path / "memory2.db")
    embedder = _FakeEmbedder(
        {
            "用户把仓库脱敏后公开发布": [1.0, 0.0],
            "用户买了一个新键盘": [0.0, 1.0],
        }
    )
    memorizer = Memorizer(store, embedder)

    async def _run() -> None:
        await memorizer.save_from_consolidation(
            history_entry="用户把仓库脱敏后公开发布",
            behavior_updates=[],
            source_ref="session@1-10#0",
            scope_channel="telegram",
            scope_chat_id="1",
        )
        await memorizer.save_from_consolidation(
            history_entry="用户买了一个新键盘",
            behavior_updates=[],
            source_ref="session@1-10#1",
            scope_channel="telegram",
            scope_chat_id="1",
        )

    asyncio.run(_run())

    items = store.list_by_type("event")
    assert len(items) == 2


def test_reinforcement_incremented_on_dedup(tmp_path):
    store = MemoryStore2(tmp_path / "memory2.db")
    embedder = _FakeEmbedder(
        {
            "用户把仓库脱敏后公开发布": [1.0, 0.0],
            "用户公开了脱敏后的仓库": [0.99, 0.01],
        }
    )
    memorizer = Memorizer(store, embedder)

    async def _run() -> None:
        await memorizer.save_from_consolidation(
            history_entry="用户把仓库脱敏后公开发布",
            behavior_updates=[],
            source_ref="session@1-10#0",
            scope_channel="telegram",
            scope_chat_id="1",
        )
        await memorizer.save_from_consolidation(
            history_entry="用户公开了脱敏后的仓库",
            behavior_updates=[],
            source_ref="session@1-10#1",
            scope_channel="telegram",
            scope_chat_id="1",
        )

    asyncio.run(_run())

    items = store.list_by_type("event")
    assert items[0]["reinforcement"] == 2


def test_emotional_weight_merged_on_event_dedup(tmp_path):
    store = MemoryStore2(tmp_path / "memory2.db")
    embedder = _FakeEmbedder(
        {
            "用户把仓库脱敏后公开发布": [1.0, 0.0],
            "用户公开了脱敏后的仓库": [0.99, 0.01],
        }
    )
    memorizer = Memorizer(store, embedder)

    async def _run() -> None:
        await memorizer.save_from_consolidation(
            history_entry="用户把仓库脱敏后公开发布",
            behavior_updates=[],
            source_ref="session@1-10#0",
            scope_channel="telegram",
            scope_chat_id="1",
            emotional_weight=0,
        )
        await memorizer.save_from_consolidation(
            history_entry="用户公开了脱敏后的仓库",
            behavior_updates=[],
            source_ref="session@1-10#1",
            scope_channel="telegram",
            scope_chat_id="1",
            emotional_weight=8,
        )

    asyncio.run(_run())

    items = store.list_by_type("event")
    assert items[0]["emotional_weight"] == 8


def test_dedup_window_is_7_days(tmp_path):
    store = MemoryStore2(tmp_path / "memory2.db")
    embedder = _FakeEmbedder(
        {
            "用户把仓库脱敏后公开发布": [1.0, 0.0],
            "用户公开了脱敏后的仓库": [0.99, 0.01],
        }
    )
    memorizer = Memorizer(store, embedder)

    async def _run() -> None:
        await memorizer.save_from_consolidation(
            history_entry="用户把仓库脱敏后公开发布",
            behavior_updates=[],
            source_ref="session@1-10#0",
            scope_channel="telegram",
            scope_chat_id="1",
        )

    asyncio.run(_run())

    old_created_at = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    store._db.execute(
        "UPDATE memory_items SET created_at=?, updated_at=? WHERE memory_type='event'",
        (old_created_at, old_created_at),
    )
    store._db.commit()

    async def _run_again() -> None:
        await memorizer.save_from_consolidation(
            history_entry="用户公开了脱敏后的仓库",
            behavior_updates=[],
            source_ref="session@1-10#1",
            scope_channel="telegram",
            scope_chat_id="1",
        )

    asyncio.run(_run_again())

    items = store.list_by_type("event")
    assert len(items) == 2

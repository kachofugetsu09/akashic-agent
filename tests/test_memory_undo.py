from __future__ import annotations

import json
from pathlib import Path

from memory2.store import MemoryStore2
from plugins.default_memory.engine import DefaultMemoryEngine


def _item_id(result: str) -> str:
    return result.split(":", 1)[1]


def _result_list(result: dict[str, object], key: str) -> list[str]:
    value = result[key]
    assert isinstance(value, list)
    return [str(item) for item in value]


def _engine(store: MemoryStore2) -> DefaultMemoryEngine:
    engine = DefaultMemoryEngine.__new__(DefaultMemoryEngine)
    engine._v2_store = store
    return engine


def test_undo_marks_direct_source_memory_superseded(tmp_path: Path):
    store = MemoryStore2(tmp_path / "memory2.db")
    try:
        engine = _engine(store)
        item_id = _item_id(
            store.upsert_item(
                memory_type="preference",
                summary="用户喜欢简短回复",
                embedding=[0.1, 0.2],
                source_ref="cli:1:1",
            )
        )

        result = engine.undo_by_message_sources(["cli:1:1"])

        assert _result_list(result, "affected_ids") == [item_id]
        assert store.get_items_by_ids([item_id])[0]["status"] == "superseded"
    finally:
        store.close()


def test_undo_dry_run_does_not_change_memory_status(tmp_path: Path):
    store = MemoryStore2(tmp_path / "memory2.db")
    try:
        engine = _engine(store)
        item_id = _item_id(
            store.upsert_item(
                memory_type="preference",
                summary="用户喜欢简短回复",
                embedding=[0.1, 0.2],
                source_ref="cli:1:1",
            )
        )

        result = engine.undo_by_message_sources(["cli:1:1"], dry_run=True)

        assert _result_list(result, "affected_ids") == [item_id]
        assert store.get_items_by_ids([item_id])[0]["status"] == "active"
    finally:
        store.close()


def test_undo_marks_consolidation_window_memory_superseded(tmp_path: Path):
    store = MemoryStore2(tmp_path / "memory2.db")
    try:
        engine = _engine(store)
        base = json.dumps(["cli:1:0", "cli:1:1", "cli:1:2"], ensure_ascii=False)
        history_id = _item_id(
            store.upsert_item(
                memory_type="event",
                summary="用户完成了压缩前的一轮任务",
                embedding=[0.1, 0.2],
                source_ref=f"{base}#h:abc",
            )
        )
        profile_id = _item_id(
            store.upsert_item(
                memory_type="profile",
                summary="用户在测试 undo",
                embedding=[0.2, 0.3],
                source_ref=f"{base}#profile",
            )
        )

        result = engine.undo_by_message_sources(["cli:1:1"])

        assert set(_result_list(result, "affected_ids")) == {history_id, profile_id}
        assert _result_list(result, "rollback_source_ids") == ["cli:1:0", "cli:1:1", "cli:1:2"]
        rows = store.get_items_by_ids([history_id, profile_id])
        assert [row["status"] for row in rows] == ["superseded", "superseded"]
    finally:
        store.close()


def test_undo_restores_old_memory_replaced_by_affected_new_memory(tmp_path: Path):
    store = MemoryStore2(tmp_path / "memory2.db")
    try:
        engine = _engine(store)
        old_id = _item_id(
            store.upsert_item(
                memory_type="preference",
                summary="旧偏好",
                embedding=[0.1, 0.2],
                source_ref="cli:1:old",
            )
        )
        new_id = _item_id(
            store.upsert_item(
                memory_type="preference",
                summary="新偏好",
                embedding=[0.2, 0.3],
                source_ref="cli:1:4",
            )
        )
        old_item = store.get_items_by_ids([old_id])[0]
        new_item = store.get_items_by_ids([new_id])[0]
        store.mark_superseded_batch([old_id])
        store.record_replacements(old_items=[old_item], new_item=new_item, source_ref="cli:1:4")

        result = engine.undo_by_message_sources(["cli:1:4"])

        assert _result_list(result, "affected_ids") == [new_id]
        assert _result_list(result, "restored_ids") == [old_id]
        old_row, new_row = store.get_items_by_ids([old_id, new_id])
        assert old_row["status"] == "active"
        assert new_row["status"] == "superseded"
    finally:
        store.close()

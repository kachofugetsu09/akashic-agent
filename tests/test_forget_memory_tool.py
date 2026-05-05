import json
from pathlib import Path

import pytest

from agent.tools.forget_memory import ForgetMemoryTool
from core.memory.engine import ForgetRequest, ForgetResult
from memory2.store import MemoryStore2


class _MemoryWriter:
    def __init__(self, store: MemoryStore2) -> None:
        self._store = store

    async def remember(self, request):
        raise NotImplementedError

    async def forget(self, request: ForgetRequest) -> ForgetResult:
        items = self._store.get_items_by_ids(request.ids)
        found_ids = [str(item["id"]) for item in items]
        if found_ids:
            self._store.mark_superseded_batch(found_ids)
        return ForgetResult(
            superseded_ids=found_ids,
            missing_ids=[item_id for item_id in request.ids if item_id not in found_ids],
            items=list(items),
        )

    def reinforce_items_batch(self, ids: list[str]) -> None:
        return None


@pytest.mark.asyncio
async def test_forget_memory_marks_existing_items_superseded(tmp_path: Path):
    store = MemoryStore2(tmp_path / "memory2.db")
    try:
        result = store.upsert_item(
            memory_type="event",
            summary="用户采访了汪远哲",
            embedding=[0.1, 0.2],
            source_ref="tg:1:1",
        )
        item_id = result.split(":", 1)[1]
        tool = ForgetMemoryTool(_MemoryWriter(store))

        raw = await tool.execute(ids=[item_id])
        payload = json.loads(raw)

        assert payload["superseded_ids"] == [item_id]
        assert payload["missing_ids"] == []
        item = store.get_items_by_ids([item_id])[0]
        assert item["status"] == "superseded"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_forget_memory_ignores_duplicates_and_reports_missing(tmp_path: Path):
    store = MemoryStore2(tmp_path / "memory2.db")
    try:
        result = store.upsert_item(
            memory_type="event",
            summary="用户做过支付排障",
            embedding=[0.1, 0.2],
            source_ref="tg:1:2",
        )
        item_id = result.split(":", 1)[1]
        tool = ForgetMemoryTool(_MemoryWriter(store))

        raw = await tool.execute(ids=[item_id, "missing", item_id])
        payload = json.loads(raw)

        assert payload["requested_ids"] == [item_id, "missing"]
        assert payload["superseded_ids"] == [item_id]
        assert payload["missing_ids"] == ["missing"]
    finally:
        store.close()

from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace
from typing import cast

import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.mobile_ui import PluginMobileUiProvider
from agent.plugins.generation import PluginGeneration
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotStore,
    lease_runtime_snapshot,
)


@pytest.mark.asyncio
async def test_mobile_ui_query_keeps_the_callers_snapshot_after_promotion() -> None:
    """A child query must retain the request snapshot across a generation switch."""

    store = RuntimeSnapshotStore()
    first_generation = SimpleNamespace(
        plugin_id="fixture",
        generation_id="fixture-s1",
        source_revision="rev-1",
        lease_count=0,
    )
    first = RuntimeSnapshot(
        "s1",
        {"fixture": cast(PluginGeneration, first_generation)},
        composition_active_plugin_ids=frozenset({"fixture"}),
    )
    store.install(first)
    provider = PluginMobileUiProvider(
        cast(PluginManager, SimpleNamespace(snapshot_store=store))
    )
    started = Event()
    release = Event()
    selected: list[str] = []

    def binding(snapshot: RuntimeSnapshot, _generation: object) -> object:
        selected.append(snapshot.snapshot_id)

        def query(
            _method: str,
            _payload: dict[str, object],
            *,
            session_id: str | None,
            turn_id: str | None,
        ) -> dict[str, object]:
            _ = session_id, turn_id
            started.set()
            if not release.wait(5):
                raise AssertionError("fixture query 未收到释放信号")
            return {"snapshot_id": snapshot.snapshot_id}

        return SimpleNamespace(query=query)

    provider._mobile_ui_binding = binding  # type: ignore[method-assign]

    async def promote() -> None:
        if not await asyncio.to_thread(started.wait, 5):
            raise AssertionError("fixture query 未启动")
        second_generation = SimpleNamespace(
            plugin_id="fixture",
            generation_id="fixture-s2",
            source_revision="rev-2",
            lease_count=0,
        )
        second = RuntimeSnapshot(
            "s2",
            {"fixture": cast(PluginGeneration, second_generation)},
            composition_active_plugin_ids=frozenset({"fixture"}),
        )
        transaction = store.begin_publish(second)
        await store.commit(transaction)
        release.set()

    try:
        async with lease_runtime_snapshot(store):
            promotion = asyncio.create_task(promote())
            result = await provider.query(
                "fixture",
                "rev-1",
                "fixture.query",
                {},
                session_id=None,
                turn_id=None,
            )
            await promotion
        assert selected == ["s1"]
        assert result == {"snapshot_id": "s1"}
        assert store.current is not None
        assert store.current.snapshot_id == "s2"
    finally:
        release.set()
        await provider.aclose()

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from infra.mobile_realtime.plugin_ui import PluginUiQuery, PluginUiQueryScheduler


class _BlockingProvider:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []

    async def query(
        self,
        plugin_id: str,
        plugin_revision: str,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        self.calls.append(plugin_id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            await self.release.wait()
            return {"plugin_id": plugin_id}
        finally:
            self.active -= 1


def _query(index: int, *, plugin_id: str, slot: str, owner: str = "owner") -> PluginUiQuery:
    return PluginUiQuery(
        request_id=f"request-{index}",
        owner_id=owner,
        plugin_id=plugin_id,
        plugin_revision="revision-1",
        method="read.current",
        payload={},
        slot=slot,
        session_id=None,
        turn_id=None,
    )


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("等待调度状态超时")


@pytest.mark.asyncio
async def test_dashboard_query_uses_reserved_fourth_device_slot() -> None:
    provider = _BlockingProvider()
    scheduler = PluginUiQueryScheduler(provider)
    regular = [
        asyncio.create_task(
            scheduler.execute(
                "device",
                _query(index, plugin_id=f"plugin-{index}", slot="drawer.panel"),
            )
        )
        for index in range(3)
    ]
    await _wait_until(lambda: provider.active == 3)

    dashboard = asyncio.create_task(
        scheduler.execute(
            "device",
            _query(3, plugin_id="dashboard", slot="dashboard.main"),
        )
    )
    queued = asyncio.create_task(
        scheduler.execute(
            "device",
            _query(4, plugin_id="queued", slot="dashboard.main"),
        )
    )
    await _wait_until(lambda: provider.active == 4)
    await asyncio.sleep(0.02)
    assert provider.active == 4
    assert not queued.done()

    provider.release.set()
    await asyncio.gather(*regular, dashboard, queued)
    assert provider.max_active == 4


@pytest.mark.asyncio
async def test_plugin_gate_caps_same_plugin_at_two_queries() -> None:
    provider = _BlockingProvider()
    scheduler = PluginUiQueryScheduler(provider)
    tasks = [
        asyncio.create_task(
            scheduler.execute(
                "device",
                _query(index, plugin_id="same", slot="dashboard.main"),
            )
        )
        for index in range(3)
    ]

    await _wait_until(lambda: provider.active == 2)
    await asyncio.sleep(0.02)
    assert provider.active == 2
    assert len(provider.calls) == 2

    provider.release.set()
    await asyncio.gather(*tasks)
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_cancel_owner_cancels_only_owned_queries() -> None:
    provider = _BlockingProvider()
    scheduler = PluginUiQueryScheduler(provider)
    cancelled = asyncio.create_task(
        scheduler.execute(
            "device",
            _query(1, plugin_id="first", slot="dashboard.main", owner="old"),
        )
    )
    survivor = asyncio.create_task(
        scheduler.execute(
            "device",
            _query(2, plugin_id="second", slot="dashboard.main", owner="new"),
        )
    )
    await _wait_until(lambda: provider.active == 2)

    assert await scheduler.cancel_owner("device", "old") == 1
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert not survivor.done()

    provider.release.set()
    assert await survivor == {"plugin_id": "second"}

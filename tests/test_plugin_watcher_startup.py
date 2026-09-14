"""首次扫描不是更新授权；旧候选只在新变化或明确唤醒后处理。"""
import asyncio
from typing import cast

import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.watcher import PluginWatcher


@pytest.mark.asyncio
@pytest.mark.parametrize("manual", [False, True])
async def test_startup_scan_does_not_resume_candidate_without_new_input(manual):
    loop = asyncio.get_running_loop()
    scanned, reconciled = asyncio.Event(), asyncio.Event()
    calls = []

    class Manager:
        revision = "unpromoted-before-restart"

        def watch_revision(self):
            value = self.revision
            loop.call_soon_threadsafe(scanned.set)
            return value

        async def reconcile_changed(self):
            calls.append(self.revision)
            reconciled.set()
            return []

    manager = Manager()
    watcher = PluginWatcher(cast(PluginManager, manager), interval_seconds=3600)
    task = asyncio.create_task(watcher.run())
    try:
        if manual:
            watcher.wake()
        else:
            watcher._wake.set()  # 模拟一次轮询，不携带手动更新授权。
        await scanned.wait()
        if not manual:
            assert calls == []
            manager.revision = "new-change-after-restart"
            watcher._wake.set()
        await reconciled.wait()
        assert calls == [manager.revision]
    finally:
        watcher.stop()
        await task


@pytest.mark.asyncio
async def test_invalid_live_input_does_not_reconcile_restored_stable():
    loop = asyncio.get_running_loop()
    scanned = asyncio.Event()

    class Manager:
        def watch_revision(self):
            loop.call_soon_threadsafe(scanned.set)
            raise ValueError("invalid live plugin identity")

        async def reconcile_changed(self):
            raise AssertionError("invalid live inputs cannot replace stable")

    watcher = PluginWatcher(cast(PluginManager, Manager()), interval_seconds=3600)
    task = asyncio.create_task(watcher.run())
    try:
        watcher._wake.set()
        await scanned.wait()
    finally:
        watcher.stop()
        await task

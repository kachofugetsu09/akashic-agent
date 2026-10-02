"""从真实 Drift 来源能力验证慢提交、重复取消和 changed 事件。"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import threading
import time
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent.plugin_composition import CompositionRoot, PluginRuntime
from plugins.drift import plugin
from plugins.drift.store import DriftStore


async def check(directory: Path, cancel: bool):
    root = CompositionRoot("drift-proposal-io")
    fiber = await root.mount(plugin.apply, name="drift", runtime=PluginRuntime(
        "drift", "io", directory, directory, directory, {}))
    ctx, now = fiber.context, datetime.now(UTC)
    store = DriftStore(directory / "drift.sqlite3")
    store.propose("old", "one", {"text": "original"}, now)
    with sqlite3.connect(store.path) as db:
        original = db.execute("SELECT * FROM proposals WHERE proposal_id='old'").fetchall()
    loop, main_thread = asyncio.get_running_loop(), threading.get_ident()
    reached, release = asyncio.Event(), threading.Event()
    stamps, events = [], []
    connect = sqlite3.connect
    class Connection(sqlite3.Connection):
        def commit(self):
            if not stamps:
                stamps.append(time.perf_counter())
                loop.call_soon_threadsafe(reached.set)
                release.wait(1)
            super().commit()
    def connection(*args, **kwargs):
        if str(args[0]) == str(store.path):
            kwargs["factory"] = Connection
        return connect(*args, **kwargs)
    def changed(_):
        assert threading.get_ident() == main_thread
        events.append("changed")
    job = None
    try:
        async with ctx.runtime_scope():
            effect = await ctx.on(plugin.DRIFT_CHANGED, changed)
            source = ctx.require(plugin.DRIFT_PROPOSALS_V2)
            payload = {"nested": {"text": "captured"}}
            try:
                with patch.object(sqlite3, "connect", connection):
                    job = asyncio.create_task(source.propose("new", "one", payload, now))
                    await asyncio.wait_for(reached.wait(), 3)
                    lag = time.perf_counter() - stamps[0]
                    assert lag < .2, lag
                    payload["nested"]["text"] = "changed by caller"
                    if cancel:
                        job.cancel()
                        await asyncio.sleep(0)
                        job.cancel()
                        await asyncio.sleep(0)
                        assert not job.done()
                    release.set()
                    try:
                        await job
                    except asyncio.CancelledError:
                        assert cancel
                assert events == ["changed"]
                replay = await source.propose("new", "one", {"nested": {"text": "captured"}}, now)
                assert not replay["inserted"]
                assert events == ["changed"]
                proposals = store.snapshot(now)["proposals"]
                assert next(p for p in proposals if p["ref"]["proposal_id"] == "new")["payload"] == {"nested": {"text": "captured"}}
                with connect(store.path) as db:
                    assert db.execute("SELECT * FROM proposals WHERE proposal_id='old'").fetchall() == original
                    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                    assert not db.execute("PRAGMA foreign_key_check").fetchall()
                return {"cancel": cancel, "loop_delay_seconds": lag, "changed": len(events)}
            finally:
                release.set()
                if job is not None:
                    await asyncio.gather(job, return_exceptions=True)
                await effect.aclose()
    finally:
        await root.dispose()


async def main():
    rows = []
    with TemporaryDirectory() as temporary:
        for cancel in (False, True):
            path = Path(temporary) / str(cancel)
            path.mkdir()
            rows.append(await check(path, cancel))
    print(json.dumps({"cases": rows, "cleanup": "passed"}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

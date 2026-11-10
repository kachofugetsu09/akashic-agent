"""真实 EventMail 发布、SQLite 与换代排空；慢 commit 保留原 SQL 效果。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import threading
import time
from datetime import UTC, datetime
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.model import FiberState, PluginRuntime
from plugins.eventmail import plugin
from plugins.eventmail.store import EventMailStore


async def check(directory: Path, kind: str, cancel: bool) -> dict:
    """从公开来源入口提交，取消时保留已接纳事务和 loop 上的 changed 事件。"""
    root = CompositionRoot("eventmail-io")
    runtime = PluginRuntime("eventmail", "io", directory, directory, directory, {})
    fiber = await root.mount(plugin.apply, name="eventmail", runtime=runtime)
    assert fiber.state is FiberState.ACTIVE, fiber.error
    ctx = fiber.context
    store = EventMailStore(directory / "eventmail.sqlite3")
    now = datetime.now(UTC)
    store.submit("original", "batch", [{"item_id": "one", "revision": "r1", "payload": {"text": "original"}}])
    with sqlite3.connect(store.path) as db:
        original = db.execute("SELECT * FROM mail_envelopes ORDER BY seq").fetchall()
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
            return super().commit()

    def connection(*args, **kwargs):
        if str(args[0]) == str(store.path):
            kwargs["factory"] = Connection
        return connect(*args, **kwargs)

    def changed(_):
        assert threading.get_ident() == main_thread
        events.append("changed")

    try:
        async with ctx.runtime_scope():
            effect = await ctx.on(plugin.EVENTMAIL_CHANGED, changed)
            service = ctx.require({"content": plugin.EVENTMAIL_CONTENT_SOURCE,
                                   "alert": plugin.EVENTMAIL_ALERT_SOURCE,
                                   "context": plugin.EVENTMAIL_CONTEXT_SOURCE}[kind])
            source = service.bind("source")
            payload = {"text": "captured"}
            async def submit():
                if kind == "content":
                    return await source.submit("batch", [{"item_id": "two", "revision": "r1", "payload": payload}])
                return await source.report(event_id="two", payload=payload, observed_at=now)
            job = None
            try:
                with patch.object(sqlite3, "connect", connection):
                    job = asyncio.create_task(submit())
                    await asyncio.wait_for(reached.wait(), 3)
                    delay = time.perf_counter() - stamps[0]
                    assert delay < 0.2, delay
                    payload["text"] = "caller changed"
                    if cancel:
                        job.cancel()
                        job.cancel()
                        await loop.run_in_executor(None, lambda: None)
                        assert not job.done()
                    release.set()
                    try:
                        await job
                    except asyncio.CancelledError:
                        assert cancel
                assert events == ["changed"]
                with connect(store.path) as db:
                    rows = db.execute("SELECT * FROM mail_envelopes ORDER BY seq").fetchall()
                    assert rows[:len(original)] == original
                    assert len(rows) == len(original) + 1
                    payload_json = db.execute("SELECT payload_json FROM mail_envelopes ORDER BY seq DESC LIMIT 1").fetchone()[0]
                    assert json.loads(payload_json)["text"] == "captured"
                    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                    assert not db.execute("PRAGMA foreign_key_check").fetchall()
                source.close()
                rebound = service.bind("source")
                if kind == "alert":
                    assert await rebound.status(event_id="two") == "pending"
                rebound.close()
                return {"kind": kind, "cancel": cancel, "loop_delay_seconds": delay, "changed": len(events)}
            finally:
                release.set()
                if job is not None and not job.done():
                    job.cancel()
                    await asyncio.gather(job, return_exceptions=True)
                source.close()
                await effect.aclose()
    finally:
        await root.dispose()


async def main(directory):
    results = []
    for kind in ("content", "alert", "context"):
        for cancel in (False, True):
            target = directory / f"{kind}-{cancel}"
            target.mkdir()
            results.append(await check(target, kind, cancel))
    return results


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-eventmail-io-") as directory:
        results = asyncio.run(main(Path(directory)))
    print(json.dumps({"cases": results, "cleanup": "passed"}, indent=2))

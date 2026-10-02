"""真实 Session 与 Wake 来源在慢提交、重复取消后的消息和恢复指针。"""
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
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION,
    MessageWriters, OwnerState, SessionAdmission,
)
from plugins.wake.api import DeliveryTarget
from plugins.wake.request import Request, TOOLS
from plugins.wake.source import Source
from plugins.wake.state import WakeState
from session.log import MessageLog, SessionAttributes
from session.message import Message, Output


async def check(directory: Path, phase: str, cancel: bool):
    """暂停真实事务提交；恢复后从原 owner 重读，不用返回值替代持久事实。"""
    log, root = MessageLog(directory / "sessions.db"), CompositionRoot("source-write-io")
    contexts = []
    async def storage(ctx):
        await ctx.provide(MESSAGE_CATALOG, log.catalog())
        await ctx.provide(MESSAGE_WRITERS, MessageWriters(log))
        await ctx.provide(OWNER_STATE, OwnerState(log))
        await ctx.provide(SESSION_ADMISSION, SessionAdmission(log))
    async def consumer(ctx):
        contexts.append(ctx)
    await root.mount(storage, name="storage")
    await root.mount(consumer, name="wake", inject=(MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION),
                     runtime=PluginRuntime("wake", "io", directory, directory, directory, {}))
    ctx = contexts[0]
    request = Request(flow_id="a" * 32, owner="drift", now=datetime.now(UTC), timezone="UTC",
        target=DeliveryTarget(channel="unused", recipient="unused", session_id="target"),
        sink={"name": "unused", "binding_id": "unused", "address": "unused"}, program_binding="unused",
        tools={name: "unused" for name in TOOLS["drift"]}, snapshot_seq=0, rules="", history="")
    source = Source(ctx, WakeState(directory / "wake.db"))
    reached, release = asyncio.Event(), threading.Event()
    loop, stamps = asyncio.get_running_loop(), []
    original_write = log._write
    def write(callback):
        def commit():
            value = callback()
            selected = (isinstance(value, SessionAttributes) if phase == "session" else
                        isinstance(value, Message) if phase == "quiet" else value is None)
            if selected and not stamps:
                stamps.append(time.perf_counter())
                loop.call_soon_threadsafe(reached.set)
                release.wait(1)
            return value
        return original_write(commit)
    job = None
    try:
        async with ctx.runtime_scope():
            if phase in {"quiet", "settle"}:
                await source.accept(request)
            reader = log.reader(request.session_id)
            before = reader.snapshot()
            with patch.object(log, "_write", write):
                async def operation():
                    async with ctx.runtime_scope():
                        if phase in {"session", "accept"}:
                            await source.accept(request)
                        else:
                            await source._settled(request, reader)
                job = asyncio.create_task(operation())
                await asyncio.wait_for(reached.wait(), 3)
                lag = time.perf_counter() - stamps[0]
                assert lag < 0.2, lag
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
            rows = reader.snapshot()
            assert rows[:len(before)] == before
            if phase == "session" and cancel:
                assert not rows and source.read(request.flow_id) is None
                assert reader.attributes == SessionAttributes(visibility="internal", learning="excluded")
            else:
                found = source.read(request.flow_id)
                assert found is not None
                if phase == "quiet" and cancel:
                    assert found[0].value["settled"] is False
                    assert isinstance(rows[-1].body, Output)
                if phase in {"quiet", "settle"}:
                    await source._settled(request, reader)
                    assert not source.pending() and len(reader.snapshot()) == 2
                else:
                    assert len(rows) == 1
            # 相同原请求重试不覆盖已提交消息或恢复指针。
            await source.accept(request)
            stable = reader.snapshot()
            await source.accept(request)
            assert reader.snapshot() == stable
        with sqlite3.connect(directory / "sessions.db") as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert not db.execute("PRAGMA foreign_key_check").fetchall()
        return {"phase": phase, "cancel": cancel, "loop_delay_seconds": lag, "messages": len(stable)}
    finally:
        release.set()
        if job is not None:
            if not job.done():
                job.cancel()
            await asyncio.gather(job, return_exceptions=True)
        await root.dispose()
        log.close()


async def main(directory):
    results = []
    for phase in ("session", "accept", "quiet", "settle"):
        for cancel in (False, True):
            target = directory / f"{phase}-{cancel}"
            target.mkdir()
            results.append(await check(target, phase, cancel))
    return results


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-source-write-") as directory:
        results = asyncio.run(main(Path(directory)))
    print(json.dumps({"cases": results, "cleanup": "passed"}, indent=2))

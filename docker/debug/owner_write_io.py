"""真实摘要和 Computer owner 在慢事务、重复取消后的存储及本地效果。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from agent.plugin_composition import CompositionRoot, MCP_SERVERS, PluginRuntime
from plugins.ledger.contract import OWNER_STATE, OwnerState
from plugins.compaction.records import SummaryRecord, SummaryRecords
from plugins.computer.control import endpoint_name
from plugins.computer.plugin import ComputerControl, _fail_group
from plugins.context.api import summary_range
from plugins.ledger.log import MessageConflict, MessageLog, OwnerTransaction
from plugins.ledger.contract import Input


async def check(directory: Path, phase: str, cancel: bool):
    """SQLite 提交屏障只延迟真实工作，Computer 通过本地 socket 写实际文件。"""
    log, root = MessageLog(directory / "sessions.db"), CompositionRoot("owner-writes")
    contexts, effects = [], []
    generation = "local-driver"
    effect = directory / "effect.txt"
    async def serve(reader, writer):
        payload = json.loads(await reader.readline())
        effect.write_text(payload["code"])
        effects.append(payload)
        writer.write(json.dumps({"path": str(effect)}).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    server = await asyncio.start_unix_server(serve, path="\0" + endpoint_name(directory, generation))
    class LocalMcp:
        @asynccontextmanager
        async def open(self, _ctx, _name):
            yield SimpleNamespace(generation_id=generation)
    async def storage(ctx):
        await ctx.provide(OWNER_STATE, OwnerState(log))
        await ctx.provide(MCP_SERVERS, LocalMcp())
    async def consumer(ctx):
        contexts.append(ctx)
    await root.mount(storage, name="storage")
    await root.mount(consumer, name="consumer", inject=(OWNER_STATE, MCP_SERVERS),
        runtime=PluginRuntime("consumer", "io", directory, directory, directory, {}))
    ctx = contexts[0]
    writer = log.writer("s", author="user", source="web", body_types=(Input,), content={})
    writer.append("one", Input(()))
    original = log.reader("s").snapshot()
    async with ctx.runtime_scope():
        state = ctx.require(OWNER_STATE).open(ctx)
    summary = SummaryRecord(reference="summary", session_id="s", generation=1, parent=None,
        source_message_ids=("one",), summary_message_ids=("one",), content="original facts",
        model_call_ids=("model",), trigger="soft_limit", context_window=1000,
        max_output_tokens=100, keep_recent_tokens=100, tokens_before=800, tokens_after=300)
    summaries = SummaryRecords(state)
    saved = state.transact(lambda tx: tx.save("computer-use:old", {"phase": "started"}, expected_version=None))
    reached, release = asyncio.Event(), threading.Event()
    loop, stamps = asyncio.get_running_loop(), []
    original_save = OwnerTransaction.save
    def save(tx, key, value, **kwargs):
        row = original_save(tx, key, value, **kwargs)
        selected = (key == "head:s" if phase == "summary" else
                    value.get("phase") == ("failed" if phase == "failed" else "started"))
        if selected and not stamps:
            stamps.append(time.perf_counter())
            loop.call_soon_threadsafe(reached.set)
            release.wait(1)
        return row
    job = None
    try:
        async def execute():
            async with ctx.runtime_scope():
                if phase == "summary":
                    result = summaries.publish(summary, log.reader("s"), parent=None, summary_range=summary_range)
                elif phase == "failed":
                    result = _fail_group(ctx, [("computer-use:old", saved)], "invalid persisted identity")
                else:
                    result = ComputerControl(ctx).run("new", "binding", {"code": "actual local effect",
                        "_computer": {"session_id": "s", "source": "web", "turn_input_id": "one"}})
                if inspect.isawaitable(result):
                    return await result
                return result
        with patch.object(OwnerTransaction, "save", save):
            job = asyncio.create_task(execute())
            await asyncio.wait_for(reached.wait(), 4)
            lag = time.perf_counter() - stamps[0]
            if os.getenv("BASELINE") != "1":
                assert lag < .2, lag
                assert not job.done()
                assert await log.reader("peer").snapshot_async(through_seq=-1) == ()
                assert not effects
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
        assert log.reader("s").snapshot() == original
        if phase == "summary":
            assert summaries.head("s") == summary
            replay = summaries.publish(summary, log.reader("s"), parent=None, summary_range=summary_range)
            if inspect.isawaitable(replay):
                await replay
            competing = summary.model_copy(update={"reference": "competing"})
            try:
                result = summaries.publish(competing, log.reader("s"), parent=None, summary_range=summary_range)
                if inspect.isawaitable(result):
                    await result
            except MessageConflict:
                pass
            else:
                raise AssertionError("stale summary parent was accepted")
            assert summaries.read("competing") is None
        elif phase == "failed":
            assert state.read("computer-use:old").value["phase"] == "failed"
        else:
            assert state.read("computer-use:new").value["phase"] == "started"
            assert len(effects) == (0 if cancel and os.getenv("BASELINE") != "1" else 1)
            if effects:
                assert effect.read_text() == "actual local effect"
        with sqlite3.connect(directory / "sessions.db") as database:
            assert database.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert database.execute("PRAGMA foreign_key_check").fetchall() == []
        return {"phase": phase, "cancel": cancel, "loop_delay_seconds": lag, "effects": len(effects)}
    finally:
        release.set()
        if job is not None:
            await asyncio.gather(job, return_exceptions=True)
        server.close()
        await server.wait_closed()
        await root.dispose()
        writer.expire()
        log.close()


async def main():
    results = []
    with TemporaryDirectory() as temporary:
        for phase in ("summary", "started", "failed"):
            for cancel in (False, True):
                path = Path(temporary) / f"{phase}-{cancel}"
                path.mkdir()
                results.append(await check(path, phase, cancel))
    print(json.dumps({"cases": results, "cleanup": "passed"}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

"""真实 Subagent 准入额度、消息和 Task 在慢写入及取消中的顺序。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import threading
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION,
    MessageWriters, OwnerState, SessionAdmission,
)
from agent.plugin_composition.tasks import TASKS, PluginTasks
from plugins.content.plugin import check_text
from plugins.subagent.inputs import CONTENT, CHECK_ORIGIN
from plugins.subagent.request import PROFILE_TOOLS, Request
from plugins.subagent.runtime import Subagents, SubagentBusy, drain
from session.log import MessageLog, OwnerTransaction
from session.message import ContentReferences, Control, Message


def request(number):
    return Request(job_id=f"{number:032x}", label="local", profile="research", background=False, retry_count=0,
        parent_session_id="parent", parent_message_id="parent-input", parent_part_index=0, origin=None,
        sink=None, program_binding="binding", tools={key: "binding" for key in PROFILE_TOOLS["research"]})


async def check(directory: Path, phase: str, cancel: bool):
    """两个独立门面共享实际准入 owner；暂停只延迟真实事务。"""
    log, tasks, root = MessageLog(directory / "sessions.db"), PluginTasks(), CompositionRoot("subagent-write")
    contexts = []
    async def storage(ctx):
        for key, value in ((MESSAGE_CATALOG, log.catalog()), (MESSAGE_WRITERS, MessageWriters(log)),
                           (OWNER_STATE, OwnerState(log)), (SESSION_ADMISSION, SessionAdmission(log)),
                           (TASKS, tasks), (CONTENT, SimpleNamespace(check_text=check_text)),
                           (CHECK_ORIGIN, lambda _: ContentReferences())):
            await ctx.provide(key, value)
    async def consumer(ctx):
        contexts.append(ctx)
    await root.mount(storage, name="storage")
    await root.mount(consumer, name="subagent", inject=(MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE,
        SESSION_ADMISSION, TASKS, CONTENT, CHECK_ORIGIN), runtime=PluginRuntime("subagent", "io", directory,
        directory, directory, {}, workspace_roots=("subagent-runs",), workspace_files=("memory/spawn_trace.jsonl",)))
    ctx = contexts[0]
    log.save_binding("binding", {"scenario": True})
    first, second = Subagents(ctx), Subagents(ctx)
    reached, release = asyncio.Event(), threading.Event()
    loop, stamps = asyncio.get_running_loop(), []
    original_save, original_write = OwnerTransaction.save, log._write
    def hold():
        if not stamps:
            stamps.append(time.perf_counter())
            loop.call_soon_threadsafe(reached.set)
            release.wait(1)
    def save(tx, key, value, **kwargs):
        row = original_save(tx, key, value, **kwargs)
        if phase == "capacity" and key == "3" or phase == "settle" and value.get("settled") is True:
            hold()
        return row
    def write(callback):
        def commit():
            result = callback()
            if phase == "pause" and isinstance(result, tuple) and isinstance(result[0], Message) and isinstance(result[0].body, Control):
                hold()
            return result
        return original_write(commit)
    job = competitor = running = None
    try:
        async with ctx.runtime_scope():
            await first.accept("1", request(1), "one")
            original = log.reader(request(1).session_id).snapshot()
            if phase == "capacity":
                await first.accept("2", request(2), "two")
            if phase == "pause":
                entered = asyncio.Event()
                async def work(_task):
                    entered.set()
                    await asyncio.Event().wait()
                running = await ctx.require(TASKS).open(ctx).admit(("job", "1"), lambda slot: slot.start(work))
                await entered.wait()
            async def operation():
                async with ctx.runtime_scope():
                    if phase == "capacity":
                        await first.accept("3", request(3), "three")
                    elif phase == "pause":
                        await first.cancel(request(1).job_id)
                    else:
                        await first._settle("1")
            async def compete():
                async with ctx.runtime_scope():
                    await second.accept("4", request(4), "four")
            with patch.object(OwnerTransaction, "save", save), patch.object(log, "_write", write):
                job = asyncio.create_task(operation())
                await asyncio.wait_for(reached.wait(), 3)
                lag = time.perf_counter() - stamps[0]
                assert lag < 0.2, lag
                if phase == "capacity":
                    competitor = asyncio.create_task(compete())
                    await loop.run_in_executor(None, lambda: None)
                    assert not competitor.done()
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
                if competitor is not None:
                    try:
                        await competitor
                    except SubagentBusy:
                        pass
                    else:
                        raise AssertionError("第四个并发子任务越过容量限制")
            assert log.reader(request(1).session_id).snapshot()[:len(original)] == original
            if phase == "capacity":
                assert len(first.jobs()) == 3
                assert request(4).session_id not in log.catalog().snapshot_attributes()
                await second.accept("3", request(3), "three")
                assert len(first.jobs()) == 3
            elif phase == "pause":
                assert running is not None and running.done and not running.active
                assert (await first.outcome(log.reader(request(1).session_id)))[0] == "cancelled"
                recovery = await first.start("1")
                if recovery is not None:
                    await drain(recovery)
                assert not first.jobs()
            else:
                assert first.read("1")[0].value["settled"] is True
        traces = [json.loads(line) for line in (directory / "memory/spawn_trace.jsonl").read_text().splitlines()]
        assert len([row for row in traces if row["phase"] == "started"]) == (3 if phase == "capacity" else 1)
        with sqlite3.connect(directory / "sessions.db") as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert not db.execute("PRAGMA foreign_key_check").fetchall()
        return {"phase": phase, "cancel": cancel, "loop_delay_seconds": lag, "trace_records": len(traces)}
    finally:
        release.set()
        for pending in (job, competitor):
            if pending is not None:
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        await tasks.close()
        await root.dispose()
        log.close()


async def main(directory):
    results = []
    for phase in ("capacity", "pause", "settle"):
        for cancel in (False, True):
            target = directory / f"{phase}-{cancel}"
            target.mkdir()
            results.append(await check(target, phase, cancel))
    return results


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-subagent-write-") as directory:
        results = asyncio.run(main(Path(directory)))
    print(json.dumps({"cases": results, "cleanup": "passed"}, indent=2))

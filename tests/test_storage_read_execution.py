"""O/C4: pinned read work does not own the writer or change committed prefixes."""
import asyncio
from contextlib import closing
import sqlite3
import threading

import pytest

from session.log import MessageLog
from session.message import Input
from tests.test_message_log import writer
from tests.test_akasha_execution import WorkerGate


@pytest.mark.asyncio
async def test_read_snapshot_does_not_hold_unrelated_writer(tmp_path):
    log = MessageLog(tmp_path / "sessions.db")
    writer(log).append("first", Input(()))
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    committed = threading.Event()

    def read():
        def snapshot():
            before = log.reader("s").head()
            loop.call_soon_threadsafe(entered.set)
            assert committed.wait(2), "read snapshot blocked the host writer"
            assert log.reader("s").head() == before
        log.owner("audit").snapshot(snapshot)

    job = asyncio.create_task(asyncio.to_thread(read))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        writer(log).append("second", Input(()))
        committed.set()
        await job
        assert log.reader("s").head() == 1
    finally:
        committed.set()
        await asyncio.gather(job, return_exceptions=True)
        log.close()


@pytest.mark.asyncio
async def test_async_prefix_is_fixed_and_external_changes_invalidate_cache(tmp_path, monkeypatch):
    import session.log as storage

    log = MessageLog(tmp_path / "sessions.db")
    writer(log).append("first", Input(()))
    reader = log.reader("s").incremental()
    gate = WorkerGate()
    original = storage._message

    def decode(row):
        gate.stop()
        return original(row)

    monkeypatch.setattr(storage, "_message", decode)
    job = asyncio.create_task(reader.snapshot_async(through_seq=0))
    try:
        await gate.wait(job)
        writer(log).append("second", Input(()))
        gate.release.set()
        assert [m.message_id for m in await job] == ["first"]
        monkeypatch.setattr(storage, "_message", original)
        assert [m.message_id for m in reader.snapshot()] == ["first", "second"]
        with closing(sqlite3.connect(tmp_path / "sessions.db")) as external, external:
            external.execute("DELETE FROM messages WHERE id='first'")
        assert [m.message_id for m in reader.snapshot()] == ["second"]
    finally:
        gate.release.set()
        await asyncio.gather(job, return_exceptions=True)
        log.close()


def test_close_inside_read_scope_closes_the_actual_writer(tmp_path):
    log = MessageLog(tmp_path / "sessions.db")
    writer(log).append("first", Input(()))
    owner = log.owner("audit")
    owner.snapshot(log.close)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        log.reader("s").head()
    with pytest.raises(RuntimeError, match="closed"):
        owner.snapshot(lambda: None)
    log.close()


@pytest.mark.asyncio
async def test_external_edit_during_async_read_does_not_restart_snapshot(tmp_path, monkeypatch):
    """O/C4: one pinned read finishes; an external edit only invalidates its cache."""
    import session.log as storage

    log = MessageLog(tmp_path / "sessions.db")
    writer(log).append("first", Input(()))
    writer(log).append("second", Input(()))
    reader = log.reader("s").incremental()
    gate = WorkerGate()
    original = storage._message

    def decode(row):
        gate.stop()
        return original(row)

    monkeypatch.setattr(storage, "_message", decode)
    job = asyncio.create_task(reader.snapshot_async(through_seq=1))
    try:
        await gate.wait(job)
        with closing(sqlite3.connect(tmp_path / "sessions.db")) as external, external:
            external.execute("DELETE FROM messages WHERE id='first'")
        gate.release.set()
        assert [m.message_id for m in await job] == ["first", "second"]
        monkeypatch.setattr(storage, "_message", original)
        assert [m.message_id for m in reader.snapshot()] == ["second"]
    finally:
        gate.release.set()
        await asyncio.gather(job, return_exceptions=True)
        log.close()


@pytest.mark.asyncio
async def test_subagent_outcome_read_keeps_control_responsive_and_prefix_fixed(tmp_path, monkeypatch):
    """O/C3/C4: 子任务终态查询不阻塞控制，也不把后来控制塞进旧快照。"""
    import session.log as storage
    from plugins.subagent.runtime import Subagents
    from session.message import Control

    log = MessageLog(tmp_path / "sessions.db")
    writer(log, source="subagent").append("job-input", Input(()))
    reader = log.reader("s")
    gate = WorkerGate()
    original = storage._message

    def decode(row):
        if row["id"] == "job-input":
            gate.stop()
        return original(row)

    monkeypatch.setattr(storage, "_message", decode)

    async def query():
        return await Subagents.outcome(reader)

    job = asyncio.create_task(query())
    try:
        await gate.wait(job)
        control = writer(log, source="subagent", bodies=(Control,))
        control.append("pause", Control("pause", 0, "user stop"))
        writer(log, source="peer").append("peer", Input(()))
        gate.release.set()
        assert await job is None
        monkeypatch.setattr(storage, "_message", original)
        assert await Subagents.outcome(reader) == ("cancelled", "子任务已按请求取消。")
        assert [item.message_id for item in reader.snapshot()] == ["job-input", "pause", "peer"]
    finally:
        gate.release.set()
        await asyncio.gather(job, return_exceptions=True)
        log.close()


@pytest.mark.asyncio
async def test_subagent_cancel_rechecks_completion_committed_during_read(tmp_path, monkeypatch):
    """C3: 真实来源在异步读取期间完成，旧取消检查不能改写其终态。"""
    from agent.plugin_composition import CompositionRoot, PluginRuntime
    from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, MessageWriters, OwnerState
    from plugins.subagent.request import PROFILE_TOOLS, Request
    from plugins.subagent.runtime import Subagents
    from session.log import MessageReader
    from session.message import ContentPart, ContentReferences, Output

    log = MessageLog(tmp_path / "sessions.db")
    root = CompositionRoot("subagent-cancel-read")
    contexts = []

    async def storage(ctx):
        await ctx.provide(MESSAGE_CATALOG, log.catalog())
        await ctx.provide(MESSAGE_WRITERS, MessageWriters(log))
        await ctx.provide(OWNER_STATE, OwnerState(log))

    async def consumer(ctx):
        contexts.append(ctx)

    await root.mount(storage, name="storage")
    await root.mount(consumer, name="jobs", inject=(MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE),
                     runtime=PluginRuntime("jobs", "jobs", tmp_path, tmp_path, tmp_path, {}))
    ctx = contexts[0]
    request = Request(job_id="a" * 32, label="job", profile="research", background=False,
                      retry_count=0, parent_session_id="parent", parent_message_id="parent-input",
                      parent_part_index=0, origin=None, sink=None, program_binding="program",
                      tools={name: name for name in PROFILE_TOOLS["research"]})
    messages = log.writer(request.session_id, author="subagent", source="subagent",
                          body_types=(Input, Output), content={"subagent.request": lambda _: ContentReferences()})
    messages.append(request.input_id, Input((ContentPart("subagent.request", request.model_dump()),)))
    async with ctx.runtime_scope():
        state = ctx.require(OWNER_STATE).open(ctx)
        state.transact(lambda tx: tx.save("job", {"session_id": request.session_id,
            "input_id": request.input_id, "settled": False}, expected_version=None))
    gate = WorkerGate()
    original = MessageReader.snapshot

    def snapshot(reader, **kwargs):
        gate.stop()
        return original(reader, **kwargs)

    monkeypatch.setattr(MessageReader, "snapshot", snapshot)

    async def cancel():
        async with ctx.runtime_scope():
            return await Subagents(ctx).cancel(request.job_id)

    job = asyncio.create_task(cancel())
    try:
        await gate.wait(job)
        messages.append("completed", Output((), "complete"))
        gate.release.set()
        assert await job is False
        monkeypatch.setattr(MessageReader, "snapshot", original)
        assert [item.message_id for item in log.reader(request.session_id).snapshot()] == [request.input_id, "completed"]
    finally:
        gate.release.set()
        await asyncio.gather(job, return_exceptions=True)
        await root.dispose()
        log.close()

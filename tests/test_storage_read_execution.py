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

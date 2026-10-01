from session.message import ContentReferences
import asyncio
from collections.abc import Awaitable
from typing import cast
from contextlib import asynccontextmanager
import pytest
from agent.plugin_composition.tasks import Tasks
from plugins.sources.session import SourceSession as Conversation
from session.log import MessageLog, WriterExpired
from session.message import ContentPart, Control, Input, Output

@asynccontextmanager
async def source(tmp_path, run, *, changed=None):
    log = MessageLog(tmp_path / "sessions.db")
    tasks = Tasks()
    def writer(body, *, source="conversation", author="app", call_ref=None):
        return log.writer("s", author=author, source=source, body_types=(body,),
                          content={"text": lambda part: ContentReferences()}, call_ref=call_ref,
                          check_call=lambda call: None)
    async def program(task, reader, source):
        output = writer(Output, source=source)
        task.on_close(output.expire)
        return await run(task, reader, output)
    conversation = Conversation(
        reader=log.reader("s"), inputs=writer(Input), controls=writer(Control),
        tasks=tasks, changed=changed,
    )
    try:
        yield conversation, log, writer, program
    finally:
        await tasks.close()
        log.close()

@pytest.mark.asyncio
async def test_interrupt_inputs_survive_and_old_output_cannot_commit(tmp_path):
    entered = asyncio.Event()
    drain = asyncio.Event()
    cancelled = asyncio.Event()
    writers = []
    async def run(task, reader, writer):
        writers.append(writer)
        if len(writers) == 1:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
                await drain.wait()
        snapshot = reader.snapshot()
        return writer.append("answer", Output((ContentPart("text", "answer"),), "complete"),
                             expected_source_head=reader.head(source="conversation"))
    async with source(tmp_path, run) as (conversation, log, writer, program):
        first = await conversation.accept("u1", Input(()))
        task = await conversation.start(program)
        await entered.wait()
        await conversation.accept("u2", Input(()))
        await cancelled.wait()
        replacement = asyncio.create_task(conversation.start(program))
        await conversation.accept("u3", Input(()))
        writer(Output, source="wake").append("proactive", Output((), "complete"))
        with pytest.raises(WriterExpired):
            writers[0].append("stale", Output((), "complete"))
        assert len(writers) == 1
        drain.set()
        latest = await replacement
        await latest.join()
        assert [m.message_id for m in log.reader("s").read()] == ["u1", "u2", "u3", "proactive", "answer"]
        assert await conversation.start(program) is None
        assert await conversation.accept("u1", Input(())) == first
        assert await conversation.start(program) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", [False, True])
async def test_source_commit_drains_before_cancel_and_rejects_late_start(tmp_path, monkeypatch, closing):
    """C3/C4/C5：取消不丢提交收据，新输入先提交时旧 intent 不能启动效果。"""
    import sqlite3
    from contextlib import closing as close_connection
    from agent.plugin_composition.tasks import TaskServiceClosed
    from plugins.sources.session import check_source
    from tests.test_akasha_execution import WorkerGate, loop_turn

    changed = []
    started = asyncio.Event()
    outputs = []

    def record(reader, name, pending):
        asyncio.get_running_loop()
        changed.append((name, reader.head()))

    async def run(task, reader, writer):
        outputs.append(writer)
        started.set()
        await asyncio.Event().wait()

    async with source(tmp_path, run, changed=record) as (conversation, log, make_writer, program):
        original = await conversation.accept("first", Input(()))
        task = await conversation.start(program)
        assert task is not None
        await started.wait()
        gate = WorkerGate()
        notify = log._notify
        stopped = False

        def committed():
            nonlocal stopped
            notify()
            if not stopped and log.reader("s").get("second") is not None:
                stopped = True
                gate.stop()

        monkeypatch.setattr(log, "_notify", committed)
        job = asyncio.create_task(conversation.accept("second", Input(())))
        state = log.owner("first-effect")
        close_job = None
        queued = None
        effect = None
        try:
            # 1. SQL 已提交，loop 通知还没运行；旧 Task 此时仍有 active 权限。
            await gate.wait(job)
            assert task.active
            with log.reader("s").read_snapshot():
                assert log.reader("s").get("second") is not None
            await loop_turn()
            job.cancel()

            def claim(tx):
                check_source(task, log.reader("s"), "conversation", original.seq, transaction=tx)
                return tx.save("started", {"phase": "started"}, expected_version=None)

            effect = asyncio.create_task(state.transact_async(claim))
            if closing:
                queued = asyncio.create_task(conversation.start(program))
                await loop_turn()
                close_job = asyncio.create_task(conversation._tasks.close())
                await loop_turn()
                assert not close_job.done(), "关闭必须等待已接纳磁盘提交"
            # 2. 取消仍排空 worker，通知一次并撤权；晚到的首次 intent 整体回滚。
            gate.release.set()
            with pytest.raises(asyncio.CancelledError):
                await job
            with pytest.raises(asyncio.CancelledError):
                await effect
            if close_job is not None:
                await close_job
            if queued is not None:
                with pytest.raises(TaskServiceClosed):
                    await queued
            with pytest.raises(asyncio.CancelledError):
                await task.join()
            assert changed == [("conversation", 0), ("conversation", 1)]
            assert state.read("started") is None
            assert log.reader("s").get("first") == original
            assert [m.message_id for m in log.reader("s").snapshot()] == ["first", "second"]
            with pytest.raises(WriterExpired):
                outputs[0].append("old-output", Output((), "complete"))
            assert await conversation._inputs.append_async(
                "first", Input(()), on_commit=lambda _message, created: pytest.fail("replay was newly created") if created else None,
            ) == original
            with close_connection(sqlite3.connect(tmp_path / "sessions.db")) as raw:
                assert raw.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert not raw.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            gate.release.set()
            await asyncio.gather(*(cast(Awaitable[object], item) for item in (job, effect, queued, close_job) if item is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("automatic", [False, True])
async def test_stop_waiting_for_storage_preserves_reply_and_explicit_head(tmp_path, monkeypatch, automatic):
    """C3/C4：内部停止重选前缀，显式 head 不得被自动放宽，原回复保持完整。"""
    from session.log import MessageWriter, SourceHeadConflict
    waiting = asyncio.Event()
    release = asyncio.Event()
    answer_ready = asyncio.Event()
    finish = asyncio.Event()
    original = MessageWriter.append_async

    async def append(writer, message_id, body, **kwargs):
        if message_id == "stop":
            waiting.set()
            await release.wait()
        return await original(writer, message_id, body, **kwargs)

    async def run(task, reader, writer):
        answer_ready.set()
        await finish.wait()
        return writer.append("answer", Output((), "complete"),
                             expected_source_head=reader.head(source="conversation"))

    monkeypatch.setattr(MessageWriter, "append_async", append)
    async with source(tmp_path, run) as (conversation, log, _writer, program):
        first = await conversation.accept("input", Input(()))
        task = await conversation.start(program)
        assert task is not None
        await answer_ready.wait()
        operation = conversation.pause("stop") if automatic else conversation.control(
            "stop", Control("pause", first.seq), expected_head=first.seq, handle=task.handle,
        )
        job = asyncio.create_task(operation)
        try:
            await waiting.wait()
            finish.set()
            answer = await task.join()
            release.set()
            if automatic:
                stop = await job
                assert isinstance(stop.body, Control) and stop.body.through_seq == answer.seq
                assert [m.message_id for m in log.reader("s").snapshot()] == ["input", "answer", "stop"]
                assert await conversation.pause("stop") == stop
            else:
                with pytest.raises(SourceHeadConflict):
                    await job
                assert [m.message_id for m in log.reader("s").snapshot()] == ["input", "answer"]
            assert log.reader("s").get("input") == first
            assert log.reader("s").get("answer") == answer
        finally:
            finish.set()
            release.set()
            await asyncio.gather(job, return_exceptions=True)

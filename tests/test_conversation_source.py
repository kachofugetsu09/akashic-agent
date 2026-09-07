from session.message import ContentReferences
import asyncio
from contextlib import asynccontextmanager

import pytest

from agent.plugin_composition.tasks import Tasks, StaleTask
from plugins.conversation.source import Conversation, needs_reply
from session.log import MessageConflict, MessageLog, WriterExpired
from session.message import ContentPart, Control, Input, Output, ToolCall, ToolResult, CallRef


@asynccontextmanager
async def source(tmp_path, run):
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
        tasks=tasks,
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
async def test_pause_survives_results_and_restart_new_input_wakes(tmp_path):
    entered = asyncio.Event()
    async def run(task, reader, writer):
        entered.set()
        await asyncio.Event().wait()
    async with source(tmp_path, run) as (conversation, log, writer, program):
        await conversation.accept("u1", Input(()))
        task = await conversation.start(program)
        await entered.wait()
        with pytest.raises(StaleTask):
            await conversation.control("stale", Control("pause", 0), expected_head=0, handle="old")
        log.save_binding("tool", {"version": 1})
        call = writer(Output).append("call", Output((ToolCall("tool", {}),), "continue"))
        paused = await conversation.control("stop", Control("pause", call.seq), expected_head=call.seq, handle=task.handle)
        assert await conversation.control("stop", Control("pause", call.seq), expected_head=call.seq, handle=task.handle) == paused
        writer(ToolResult, call_ref=CallRef("call", 0)).append("result", ToolResult(CallRef("call", 0), "success", ()))
        assert await conversation.start(program) is None
        assert not needs_reply(log.reader("s").snapshot(), "conversation")
        # 新对象没有持久 active/attempt；同一日志足以恢复暂停。
        restarted = Conversation(reader=log.reader("s"), inputs=writer(Input), controls=writer(Control),
                                 tasks=Tasks())
        assert await restarted.start(program) is None
        await conversation.accept("u2", Input(()))
        assert needs_reply(log.reader("s").snapshot(), "conversation")


@pytest.mark.asyncio
async def test_control_cannot_target_other_source_and_abandon_does_not_eat_new_input(tmp_path):
    async def run(task, reader, writer):
        return None
    async with source(tmp_path, run) as (conversation, log, writer, program):
        await conversation.accept("u1", Input(()))
        writer(Output, source="timer").append("timer", Output((), "complete"))
        with pytest.raises(MessageConflict, match="同来源"):
            await conversation.control("wrong", Control("pause", 1), expected_head=0, handle=None)
        await conversation.accept("u2", Input(()))
        await conversation.control("abandon", Control("abandon", 0), expected_head=2, handle=None)
        assert needs_reply(log.reader("s").snapshot(), "conversation")
        with pytest.raises(MessageConflict, match="关闭"):
            await conversation.control("again", Control("abandon", 0), expected_head=3, handle=None)
        await conversation.control("stop", Control("pause", 2), expected_head=3, handle=None)
        assert await conversation.start(program) is None
        await conversation.control("resume", Control("resume", 2), expected_head=4, handle=None)
        assert needs_reply(log.reader("s").snapshot(), "conversation")


@pytest.mark.asyncio
async def test_explicit_retry_reuses_input_and_replays_resume_after_later_messages(tmp_path):
    attempts = 0
    async def run(task, reader, writer):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("model request failed")
        return writer.append("a", Output((ContentPart("text", "answer"),), "complete"))
    async with source(tmp_path, run) as (conversation, log, writer, program):
        await conversation.accept("u1", Input((ContentPart("text", "question"),)))
        with pytest.raises(ValueError, match="model request"):
            await (await conversation.start(program)).join()
        resumed = await conversation.resume("retry", "u1")
        assert isinstance(resumed.body, Control) and resumed.body.action == "resume"
        assert needs_reply(log.reader("s").snapshot(), "conversation")
        await (await conversation.start(program)).join()
        await conversation.accept("u2", Input((ContentPart("text", "next"),)))
        assert await conversation.resume("retry", "u1") == resumed
        with pytest.raises(MessageConflict, match="另一条输入"):
            await conversation.resume("retry", "u2")
        assert [m.message_id for m in log.reader("s").snapshot() if isinstance(m.body, Input)] == ["u1", "u2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", ["complete", "abandon", "running", "newer"])
async def test_retry_rejects_closed_active_or_superseded_inputs(tmp_path, closed):
    entered, release = asyncio.Event(), asyncio.Event()
    async def run(task, reader, writer):
        entered.set()
        await release.wait()
    async with source(tmp_path, run) as (conversation, log, writer, program):
        await conversation.accept("u1", Input(()))
        writer(Control).append("failed", Control("failure", 0, "failure"))
        if closed == "complete":
            writer(Output).append("a", Output((), "quiet"))
        elif closed == "abandon":
            await conversation.control("abandon", Control("abandon", 0), expected_head=1, handle=None)
        elif closed == "newer":
            await conversation.accept("u2", Input(()))
        else:
            await conversation.resume("first-retry", "u1")
            await conversation.start(program)
            await entered.wait()
        try:
            with pytest.raises(MessageConflict):
                await conversation.resume("retry", "u1")
        finally:
            release.set()


@pytest.mark.asyncio
async def test_stop_and_duplicate_stop_wait_for_cleanup_without_cancelling_it_twice(tmp_path):
    entered, draining, release = (asyncio.Event() for _ in range(3))
    held = []
    async def run(task, reader, writer):
        held.append(writer)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            draining.set()
            await release.wait()
    async with source(tmp_path, run) as (conversation, log, writer, program):
        await conversation.accept("u1", Input(()))
        task = await conversation.start(program)
        await entered.wait()
        stopping = asyncio.create_task(conversation.control(
            "stop", Control("pause", 0), expected_head=0, handle=task.handle))
        await draining.wait()
        assert not stopping.done()
        with pytest.raises(WriterExpired):
            held[0].append("stale", Output((), "quiet"))
        # 等待者退出不会向已经结算中的旧工作再送一次取消。
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert not task.done
        repeated = asyncio.create_task(conversation.control(
            "stop", Control("pause", 0), expected_head=0, handle=task.handle))
        release.set()
        assert (await repeated).message_id == "stop"
        assert task.done
        assert [m.message_id for m in log.reader("s").snapshot()] == ["u1", "stop"]


@pytest.mark.asyncio
async def test_completion_waits_for_human_reply_and_cancellation_does_not_cancel_human(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    async def human(task, reader, writer):
        entered.set()
        await release.wait()
        return writer.append("human-answer", Output((), "complete"))
    async with source(tmp_path, human) as (conversation, log, writer, program):
        await conversation.accept("human", Input(()))
        task = await conversation.start(program)
        await entered.wait()
        waiting = asyncio.Event()
        async def report(task, reader):
            pytest.fail("completion cannot start while human reply is active")
        async def complete():
            waiting.set()
            return await conversation.complete(report)
        pending = asyncio.create_task(complete())
        await waiting.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert task.active
        release.set()
        await task.join()
        assert log.reader("s").get("human-answer") is not None


@pytest.mark.asyncio
async def test_new_human_input_revokes_completion_then_result_resumes_after_reply(tmp_path):
    entered, revoked, drain = asyncio.Event(), asyncio.Event(), asyncio.Event()
    attempts = []
    async def human(task, reader, writer):
        return writer.append("human-answer", Output((), "complete"))
    async with source(tmp_path, human) as (conversation, log, writer, program):
        writer(Output).append("previous-answer", Output((), "complete"))
        async def report(task, reader):
            output = writer(Output, source="background:one")
            task.on_close(output.expire)
            attempts.append(output)
            if len(attempts) == 1:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    revoked.set()
                    await drain.wait()
            assert reader.get("human-answer") is not None
            return output.append("report", Output((ContentPart("text", "result"),), "complete"))
        pending = asyncio.create_task(conversation.complete(report))
        await entered.wait()
        await conversation.accept("human", Input(()))
        await revoked.wait()
        with pytest.raises(WriterExpired):
            attempts[0].append("late", Output((), "complete"))
        start = asyncio.create_task(conversation.start(program))
        drain.set()
        await (await start).join()
        result = await asyncio.wait_for(pending, 3)
        assert result.message_id == "report"
        assert [m.message_id for m in log.reader("s").snapshot()] == ["previous-answer", "human", "human-answer", "report"]
        assert len(attempts) == 2

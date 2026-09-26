from session.message import ContentReferences
import asyncio
from contextlib import asynccontextmanager
import pytest
from agent.plugin_composition.tasks import Tasks
from plugins.sources.session import SourceSession as Conversation
from session.log import MessageLog, WriterExpired
from session.message import ContentPart, Control, Input, Output

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

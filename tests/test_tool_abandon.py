import asyncio
from contextlib import asynccontextmanager

import pytest

from agent.restart import RestartGate
from plugins.content.plugin import check_text
from plugins.tools.abandon import abandon_call, follow_abandon, reject_start
from plugins.tools.api import MessageReply, Result, durable_call_key, result_message_id
from plugins.tools.execution import ToolExecution, _fingerprint
from session.log import OwnerTransaction
from session.message import CallRef, ContentPart, Control, Input, Output, ToolCall, ToolResult
from tests.test_tool_execution_receipts import dialogue, environment


def abandon(log, reply, identity="abandon"):
    reader = reply.reader
    writer = log.writer(reader.session_id, author="user", source=reply.writer.source,
                        body_types=(Control,), content={})
    return writer.append(identity, Control("abandon", reader.head(source=reply.writer.source)))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prepare", "invoke", "cleanup"])
async def test_abandon_releases_waiter_but_keeps_real_owner_and_one_result(environment, phase):
    log, state, tasks, probe, _, execution = environment
    reply = dialogue(log)
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    gate = RestartGate(boot_id="test", supervised=True, commit=lambda _: None)
    root = gate.acquire()
    execution._child_permit = root.child

    async def held():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    if phase == "prepare":
        async def prepare(arguments, source=None):
            await held()
            return arguments
        probe.prepare = prepare
    elif phase == "invoke":
        async def invoke(key, arguments):
            probe.calls.append((key, arguments))
            await held()
            return Result("success", (ContentPart("text", "late actual result"),))
        probe.invoke = invoke
    else:
        @asynccontextmanager
        async def open_tool(binding):
            yield probe
            await held()
        execution._open_tool = open_tool

    running = asyncio.create_task(execution.execute_call(reply))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        abandon(log, reply)
        assert (await abandon_call(state, tasks, reply, task_key="tools")).outcome == {
            "prepare": "denied", "invoke": "interrupted", "cleanup": "success",
        }[phase]
        result = await asyncio.wait_for(running, 1)
        await asyncio.wait_for(cancelled.wait(), 1)
        root.release()
        assert gate.permit_count == 1
        effect = await tasks.admit(("tools", durable_call_key(reply.call_ref)), lambda slot: slot.current)
        assert effect is not None and not effect.done
        # 重复等待不会重放，且关闭旧内容 writer 后迟到正常返回仍不能写第二次。
        assert await asyncio.wait_for(execution.execute_call(reply), 1) == result
        reply.writer.expire()
        release.set()
        try:
            await effect.join()
        except asyncio.CancelledError:
            pass
        assert gate.permit_count == 0
        results = [m.body for m in reply.reader.snapshot() if isinstance(m.body, ToolResult)]
        assert results == [ToolResult(reply.call_ref, result.outcome, result.parts)]
        assert len(probe.calls) == (0 if phase == "prepare" else 1)
    finally:
        release.set()
        root.release()
        await tasks.close()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_startup_consumer_settles_old_calls_after_new_turn_completed(environment):
    log, state, tasks, probe, _, execution = environment
    first = dialogue(log)
    outputs = log.writer("s", author="agent", source="conversation", body_types=(Output,),
                         content={"text": check_text}, check_call=lambda call: None)
    outputs.append("more", Output((ToolCall("fixed-A", {}), ToolCall("fixed-A", {})), "continue"))
    # 一个 started 回执模拟进程中断；另两个调用连 Task 都没有创建。
    state.transact(lambda tx: tx.save(durable_call_key(first.call_ref), {
        "version": 1, "request": _fingerprint("fixed-A", {}, first), "binding": "fixed-A",
        "reply_id": first.message_id, "phase": "started", "arguments": {}, "permission": {},
    }, expected_version=None))
    abandon(log, first)
    log.writer("s", author="user", source="conversation", body_types=(Input,), content={"text": check_text}).append(
        "new-input", Input((ContentPart("text", "new work"),)))
    outputs.append("new-done", Output((ContentPart("text", "new answer"),), "complete"))
    other = log.writer("s", author="agent", source="other", body_types=(Output,), content={}, check_call=lambda call: None)
    other.append("other-call", Output((ToolCall("fixed-A", {}),), "continue"))

    async def reply(reader, source, ref):
        writer = log.writer(reader.session_id, author="tool", source=source,
                            body_types=(ToolResult,), content={"text": check_text}, call_ref=ref)
        return MessageReply(result_message_id(ref), ref, reader, writer, reject_start)

    watcher = asyncio.create_task(follow_abandon(log.catalog(), state, tasks, reply, task_key="tools"))
    try:
        async def settled():
            async for _ in first.reader.follow():
                results = [m.body for m in first.reader.snapshot() if isinstance(m.body, ToolResult)]
                if len(results) == 3:
                    return results
            raise AssertionError("工具结果订阅提前结束")
        results = await asyncio.wait_for(settled(), 1)
        assert [r.outcome for r in results] == ["interrupted", "denied", "denied"]
        assert results[0].call_ref == first.call_ref
        assert not probe.calls and probe.query_count == 0
        assert (await execution.execute_call(first)).outcome == "interrupted"
        assert first.reader.get("new-done").body.parts[0].value == "new answer"
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        await tasks.close()


@pytest.mark.asyncio
async def test_failed_abandon_transaction_does_not_cancel_or_partly_publish(environment, monkeypatch):
    log, state, tasks, probe, _, execution = environment
    reply = dialogue(log)
    probe.release.clear()
    running = asyncio.create_task(execution.execute_call(reply))
    await probe.started.wait()
    abandon(log, reply)
    save = OwnerTransaction.save

    def fail(self, key, value, **kwargs):
        if value["phase"] == "done":
            raise OSError("interrupt commit failed")
        return save(self, key, value, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(OwnerTransaction, "save", fail)
            with pytest.raises(OSError, match="interrupt commit failed"):
                await abandon_call(state, tasks, reply, task_key="tools")
        assert reply.reader.get(reply.message_id) is None
        assert state.read(durable_call_key(reply.call_ref)).value["phase"] == "started"
        active = await tasks.admit(("tools", durable_call_key(reply.call_ref)), lambda slot: slot.current)
        assert active.active
        probe.release.set()
        assert (await running).outcome == "success"
    finally:
        probe.release.set()
        await tasks.close()
        await asyncio.gather(running, return_exceptions=True)

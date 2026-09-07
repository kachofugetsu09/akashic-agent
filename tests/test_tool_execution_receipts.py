import asyncio
from dataclasses import replace
from contextlib import asynccontextmanager

import pytest

from agent.plugin_composition.tasks import Tasks
from plugins.tools.execution import Denied, MessageReply, Result, ToolExecution
from session.log import MessageLog, OwnerTransaction
from session.message import CallRef, ContentPart, Output, ToolCall, ToolResult


class Probe:
    idempotent = False

    def __init__(self, state):
        self.state = state
        self.calls = []
        self.query_count = 0
        self.query_result = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def prepare(self, arguments):
        return {**arguments, "prepared": True}

    async def invoke(self, key, arguments):
        assert self.state.read(key).value["phase"] == "started"
        self.calls.append((key, arguments))
        self.started.set()
        await self.release.wait()
        return Result("success", (ContentPart("text", "actual output"),))

    async def query(self, key):
        self.query_count += 1
        return self.query_result


@pytest.fixture
def environment(tmp_path):
    log = MessageLog(tmp_path / "state.db")
    state = log.owner("tool-execution")
    tasks = Tasks()
    probe = Probe(state)
    permissions = []

    async def authorize(binding, arguments):
        permissions.append((binding, arguments))
        return {"policy": "test-policy", "revision": 1}

    @asynccontextmanager
    async def open_tool(binding):
        assert binding == "fixed-A"
        yield probe

    execution = ToolExecution(state, tasks, open_tool, authorize, task_key="tools")
    yield log, state, tasks, probe, permissions, execution
    log.close()


@pytest.mark.asyncio
async def test_execute_and_repeat_use_final_arguments_and_one_durable_result(
    environment,
):
    log, state, tasks, probe, permissions, execution = environment
    try:
        first = await execution.execute("request", "fixed-A", {"value": 1})
        assert await execution.execute("request", "fixed-A", {"value": 1}) == first
        assert probe.calls == [("program:request", {"value": 1, "prepared": True})]
        assert permissions == [("fixed-A", {"value": 1, "prepared": True})]
        assert state.read("program:request").value["phase"] == "done"
        with pytest.raises(ValueError, match="不一致"):
            await execution.execute("request", "fixed-A", {"value": 2})
        assert len(probe.calls) == 1
        assert log.reader("unused-session").read() == ()
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_denied_final_arguments_never_enter_tool(environment):
    _, state, tasks, probe, _, execution = environment

    async def deny(binding, arguments):
        assert arguments["prepared"] is True
        raise Denied("permission revoked")

    execution._authorize = deny
    try:
        result = await execution.execute("request", "fixed-A", {})
        assert result.outcome == "denied"
        assert not probe.calls
        assert state.read("program:request").value["result"]["outcome"] == "denied"
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_duplicate_waiter_cancel_does_not_cancel_the_effect_owner(environment):
    _, state, tasks, probe, _, execution = environment
    probe.release.clear()
    first = asyncio.create_task(execution.execute("request", "fixed-A", {}))
    await probe.started.wait()
    joined = asyncio.Event()
    original_admit = tasks.admit

    async def observe_join(key, callback):
        result = await original_admit(key, callback)
        joined.set()
        return result

    tasks.admit = observe_join
    second = asyncio.create_task(execution.execute("request", "fixed-A", {}))
    try:
        await joined.wait()
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert not first.done()
        assert state.read("program:request").value["phase"] == "started"
        probe.release.set()
        assert (await first).outcome == "success"
        assert len(probe.calls) == 1
    finally:
        probe.release.set()
        await tasks.close()


@pytest.mark.asyncio
async def test_owner_cancel_records_unknown_and_never_blindly_retries(environment):
    _, state, tasks, probe, _, execution = environment
    probe.release.clear()
    running = asyncio.create_task(execution.execute("request", "fixed-A", {}))
    await probe.started.wait()
    try:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert state.read("program:request").value["result"]["outcome"] == "unknown"
        assert (await execution.execute("request", "fixed-A", {})).outcome == "unknown"
        assert len(probe.calls) == 1
    finally:
        probe.release.set()
        await tasks.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [False, True])
async def test_restart_started_call_queries_before_deciding_unknown(environment, query):
    _, state, tasks, probe, _, execution = environment
    await execution.execute("request", "fixed-A", {})
    completed = state.read("program:request")
    # Crash fixture: durable start survived, the result transaction did not.
    state.transact(
        lambda tx: tx.save(
            "program:request",
            {
                key: value
                for key, value in {**completed.value, "phase": "started"}.items()
                if key != "result"
            },
            expected_version=completed.version,
        )
    )
    if query:
        probe.query_result = Result("success", (ContentPart("text", "queried output"),))
    try:
        result = await execution.execute("request", "fixed-A", {})
        assert result.outcome == ("success" if query else "unknown")
        assert probe.query_count == 1
        assert len(probe.calls) == 1
    finally:
        await tasks.close()


def dialogue(log, *, session_id="s", arguments=None):
    """建立真实的已提交调用和只允许该调用结果的写入能力。"""
    log.save_binding("fixed-A", {"target": "immutable-A"})
    outputs = log.writer(
        session_id,
        author="agent",
        source="conversation",
        body_types=(Output,),
        content={},
        check_call=lambda call: None,
    )
    outputs.append("call", Output((ToolCall("fixed-A", arguments or {}),), "continue"))
    ref = CallRef("call", 0)
    writer = log.writer(
        session_id,
        author="tool",
        source="conversation",
        body_types=(ToolResult,),
        content={"text": lambda part: ()},
        call_ref=ref,
    )
    return MessageReply("result", ref, log.reader(session_id), writer)


@pytest.mark.asyncio
async def test_dialogue_receipt_points_to_one_message_body(environment):
    log, state, tasks, probe, _, execution = environment
    reply = dialogue(log)
    try:
        result = await execution.execute_call(reply)
        stored = log.reader("s").get("result")
        assert stored.body == ToolResult(reply.call_ref, result.outcome, result.parts)
        assert state.read(probe.calls[0][0]).value["result"] == {
            "message_id": "result",
            "seq": stored.seq,
        }
        assert await execution.execute_call(reply) == result
        assert len(log.reader("s").read()) == 2
        assert len(probe.calls) == 1
        assert log.reader("other-session").get("result") is None
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_dialogue_derives_request_and_prevents_a_second_result_destination(
    environment,
):
    log, state, tasks, probe, _, execution = environment
    reply = dialogue(log, arguments={"value": 1})
    try:
        first, second = await asyncio.gather(
            execution.execute_call(reply), execution.execute_call(reply)
        )
        assert first == second
        assert len(probe.calls) == 1
        assert probe.calls[0][1] == {"value": 1, "prepared": True}
        with pytest.raises(ValueError, match="不一致"):
            await execution.execute_call(replace(reply, message_id="another-result"))
        assert len(probe.calls) == 1
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_concurrent_call_destinations_cannot_duplicate_effect(environment):
    log, _, tasks, probe, _, execution = environment
    reply = dialogue(log)
    probe.release.clear()
    first = asyncio.create_task(execution.execute_call(reply))
    await probe.started.wait()
    joined = asyncio.Event()
    admit = tasks.admit

    async def observe_join(key, callback):
        slot = await admit(key, callback)
        joined.set()
        return slot

    tasks.admit = observe_join
    second = asyncio.create_task(
        execution.execute_call(replace(reply, message_id="another-result"))
    )
    try:
        await joined.wait()
        probe.release.set()
        await first
        with pytest.raises(ValueError, match="不一致"):
            await second
        assert len(probe.calls) == 1
        assert log.reader("s").get("another-result") is None
    finally:
        probe.release.set()
        await tasks.close()


@pytest.mark.asyncio
async def test_receipt_write_failure_rolls_back_message_and_recovers_by_query(
    environment,
    monkeypatch,
):
    log, state, tasks, probe, _, execution = environment
    reply = dialogue(log)
    save = OwnerTransaction.save

    def fail_after_message(self, key, value, **kwargs):
        if value["phase"] == "done":
            assert log.reader("s").get("result") is not None
            raise OSError("receipt disk failure")
        return save(self, key, value, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(OwnerTransaction, "save", fail_after_message)
            with pytest.raises(OSError, match="receipt disk failure"):
                await execution.execute_call(reply)
        assert state.read(probe.calls[0][0]).value["phase"] == "started"
        assert log.reader("s").get("result") is None
        assert log.reader("s").head() == 0
        probe.query_result = Result("success", (ContentPart("text", "actual output"),))
        assert await execution.execute_call(reply) == probe.query_result
        assert probe.query_count == 1
        assert len(probe.calls) == 1
        assert log.reader("s").get("result").seq == 1
        assert len(log.reader("s").read()) == 2
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_idempotent_recovery_rechecks_permission_after_query(environment):
    _, state, tasks, probe, _, execution = environment
    await execution.execute("request", "fixed-A", {})
    completed = state.read("program:request")
    state.transact(
        lambda tx: tx.save(
            "program:request",
            {
                key: value
                for key, value in {
                    **completed.value,
                    "phase": "started",
                }.items()
                if key != "result"
            },
            expected_version=completed.version,
        )
    )
    probe.idempotent = True

    async def authorize(binding, arguments):
        if probe.query_count:
            raise Denied("revoked during query")
        return {"policy": "allowed before query"}

    execution._authorize = authorize
    try:
        result = await execution.execute("request", "fixed-A", {})
        assert result.outcome == "unknown"
        assert probe.query_count == 1
        assert len(probe.calls) == 1
    finally:
        await tasks.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch", ["reader-authority", "writer-session", "writer-call", "expired"]
)
async def test_dialogue_rejects_mismatched_capabilities_before_effect(
    environment, tmp_path, mismatch
):
    log, _, tasks, probe, _, execution = environment
    reply = dialogue(log, arguments={"value": 1})
    other = MessageLog(tmp_path / "other.db")
    try:
        if mismatch == "reader-authority":
            dialogue(other, arguments={"value": 2})
            reply = replace(reply, reader=other.reader("s"))
        elif mismatch == "writer-session":
            reply = replace(
                reply,
                writer=log.writer(
                    "other-session",
                    author="tool",
                    source="conversation",
                    body_types=(ToolResult,),
                    content={},
                    call_ref=reply.call_ref,
                ),
            )
        elif mismatch == "writer-call":
            reply = replace(
                reply,
                writer=log.writer(
                    "s",
                    author="tool",
                    source="conversation",
                    body_types=(ToolResult,),
                    content={},
                    call_ref=CallRef("wrong-call", 0),
                ),
            )
        else:
            reply.writer.expire()
        with pytest.raises((ValueError, PermissionError, RuntimeError)):
            await execution.execute_call(reply)
        assert not probe.calls
        assert log.reader("s").head() == 0
    finally:
        other.close()
        await tasks.close()


def restore_started(state):
    completed = state.read("program:request")
    state.transact(
        lambda tx: tx.save(
            "program:request",
            {
                key: value
                for key, value in {
                    **completed.value,
                    "phase": "started",
                }.items()
                if key != "result"
            },
            expected_version=completed.version,
        )
    )


@pytest.mark.asyncio
async def test_revoked_execution_permission_does_not_discard_queried_success(
    environment,
):
    _, state, tasks, probe, _, execution = environment
    await execution.execute("request", "fixed-A", {})
    restore_started(state)
    probe.query_result = Result("success", (ContentPart("text", "known effect"),))

    async def deny(binding, arguments):
        raise AssertionError("historical query must not need a new execution grant")

    execution._authorize = deny
    try:
        assert await execution.execute("request", "fixed-A", {}) == probe.query_result
        assert probe.query_count == 1
        assert len(probe.calls) == 1
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_cancel_during_recovery_prevents_later_idempotent_reexecution(
    environment,
):
    _, state, tasks, probe, _, execution = environment
    await execution.execute("request", "fixed-A", {})
    restore_started(state)
    probe.idempotent = True
    entered = asyncio.Event()
    release = asyncio.Event()

    async def query(key):
        entered.set()
        await release.wait()
        return None

    probe.query = query
    running = asyncio.create_task(execution.execute("request", "fixed-A", {}))
    try:
        await entered.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert state.read("program:request").value["phase"] == "done"
        assert (await execution.execute("request", "fixed-A", {})).outcome == "unknown"
        assert len(probe.calls) == 1
    finally:
        release.set()
        await tasks.close()

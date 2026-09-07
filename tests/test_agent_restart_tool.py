from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugin_composition.tasks import Tasks
from agent.restart import RESTART_GATE, RestartRejectedError
from plugins.agent_restart.plugin import (
    PendingRestart,
    RestartRuntime,
    RestartTool,
)
from plugins.delivery.api import FINAL_OUTPUT_DELIVERY
from plugins.tools.api import CallSource, ContentPart, Denied, MessageReply, Result, durable_call_key
from plugins.tools.execution import ToolExecution
from plugins.turn_projection.plugin import TURN_PROJECTION, TurnProjection
from session.log import MessageLog
from session.message import CallRef, ContentReferences, Message, Output, ToolCall, ToolResult, freeze_json


def _source(message_id: str = "call-a", reason: str = "reload") -> CallSource:
    call_ref = CallRef(message_id, 0)
    message = Message(
        message_id,
        "session-a",
        0,
        datetime.now(timezone.utc),
        "agent",
        "conversation",
        Output((ToolCall("binding-a", {"reason": reason}),), "continue"),
    )
    return CallSource(call_ref, (message,))


def _pending(source: CallSource, request_id: str = "restart-a") -> PendingRestart:
    message = source.messages[0]
    arguments = freeze_json({"reason": "reload"})
    assert isinstance(arguments, Mapping)
    return PendingRestart(
        source.call_ref,
        request_id,
        message.session_id,
        message.source,
        durable_call_key(source.call_ref),
        cast(Mapping[str, object], arguments),
    )


class _Reader:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.error = error

    async def follow(self):
        self.started.set()
        if self.error is not None:
            raise self.error
        await self.release.wait()
        if False:
            yield None


class _Catalog:
    def __init__(self, reader: _Reader) -> None:
        self.reader_value = reader

    def reader(self, _session_id: str) -> _Reader:
        return self.reader_value


class _Gate:
    def __init__(self) -> None:
        self.aborted: list[str] = []
        self.committed: list[str] = []

    def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

    async def commit(self, request_id: str) -> None:
        self.committed.append(request_id)


def _context(gate: _Gate, reader: _Reader) -> SimpleNamespace:
    catalog = _Catalog(reader)
    services = {
        RESTART_GATE: gate,
        MESSAGE_CATALOG: catalog,
        TURN_PROJECTION: object(),
        FINAL_OUTPUT_DELIVERY: object(),
    }
    return SimpleNamespace(require=services.__getitem__)


@pytest.mark.asyncio
async def test_restart_tool_binds_invoke_to_prepared_durable_call() -> None:
    runtime = RestartRuntime(cast(Any, object()))
    tool = RestartTool(runtime)
    source = _source()
    with pytest.raises(ValueError, match="只能包含 reason"):
        await tool.prepare({"reason": "reload", "extra": True}, source)
    prepared = await tool.prepare({"reason": " reload "}, source)
    pending = tool._prepared
    assert pending is not None
    assert prepared == {"reason": "reload"}
    assert pending.effect_key == durable_call_key(source.call_ref)
    assert pending.arguments == prepared
    assert runtime.pending is None

    started: list[PendingRestart] = []

    async def start(value: PendingRestart) -> None:
        started.append(value)

    runtime.start = start  # type: ignore[method-assign]
    result = await tool.invoke(pending.effect_key, prepared)
    assert result.outcome == "success"
    assert started == [pending]

    with pytest.raises(RestartRejectedError, match="durable key"):
        await tool.invoke("message:[\"other\",0]", prepared)
    with pytest.raises(RestartRejectedError, match="参数"):
        await tool.invoke(pending.effect_key, {"reason": "other"})


@pytest.mark.asyncio
async def test_restart_prepare_is_idempotent_only_for_same_call_and_arguments() -> None:
    runtime = RestartRuntime(cast(Any, object()))
    tool = RestartTool(runtime)
    source = _source()
    await tool.prepare({"reason": "reload"}, source)
    first = tool._prepared
    assert first is not None
    assert runtime.pending is None

    await tool.prepare({"reason": "reload"}, source)
    assert tool._prepared is first

    with pytest.raises(RestartRejectedError, match="参数不一致"):
        await tool.prepare({"reason": "different"}, source)
    with pytest.raises(RestartRejectedError, match="多个 restart"):
        await tool.prepare({"reason": "reload"}, _source("call-b"))


@pytest.mark.asyncio
async def test_restart_recovery_is_explicitly_unknown_without_boot_pending() -> None:
    runtime = RestartRuntime(cast(Any, object()))
    tool = RestartTool(runtime)
    assert tool.idempotent is False
    with pytest.raises(RestartRejectedError, match="当前 boot"):
        await tool.invoke("message:[\"call-a\",0]", {"reason": "reload"})


@pytest.mark.asyncio
async def test_restart_wait_failure_clears_only_its_pending_owner() -> None:
    source = _source()
    pending = _pending(source)
    gate = _Gate()
    reader = _Reader(error=RestartRejectedError("turn failed"))
    runtime = RestartRuntime(_context(gate, reader))
    runtime.pending = pending

    with pytest.raises(RestartRejectedError, match="turn failed"):
        await runtime._wait_for_final_output(pending)

    assert runtime.pending is None
    assert gate.aborted == [pending.request_id]

    replacement = _pending(_source("call-b"), "restart-b")
    runtime.pending = replacement
    runtime._clear_pending(pending)
    assert runtime.pending is replacement


@pytest.mark.asyncio
async def test_restart_wait_cancellation_reopens_gate_and_allows_next_prepare() -> None:
    source = _source()
    pending = _pending(source)
    gate = _Gate()
    reader = _Reader()
    runtime = RestartRuntime(_context(gate, reader))
    runtime.pending = pending

    waiting = asyncio.create_task(runtime._wait_for_final_output(pending))
    await reader.started.wait()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert runtime.pending is None
    assert gate.aborted == [pending.request_id]

    replacement = _pending(_source("call-b"), "restart-b")
    runtime.prepare(replacement)
    assert runtime.pending is replacement


@pytest.mark.asyncio
async def test_restart_waits_for_complete_turn_delivery_before_gate_commit(tmp_path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    gate = _Gate()
    delivered: list[str] = []
    try:
        log.save_binding("binding-a", {})
        writer = log.writer(
            "session-a", author="agent", source="conversation",
            body_types=(Output,), content={}, check_call=lambda _call: None,
        )
        call = writer.append(
            "call-a", Output((ToolCall("binding-a", {"reason": "reload"}),), "continue")
        )
        source = CallSource(CallRef(call.message_id, 0), (call,))
        pending = _pending(source)

        class _Delivery:
            async def wait(self, _reader, turn) -> None:
                delivered.append(turn.ending_message_id or "")

        catalog = SimpleNamespace(reader=log.reader)
        context = SimpleNamespace(require={
            RESTART_GATE: gate,
            MESSAGE_CATALOG: catalog,
            TURN_PROJECTION: TurnProjection(),
            FINAL_OUTPUT_DELIVERY: _Delivery(),
        }.__getitem__)
        runtime = RestartRuntime(context)
        runtime.pending = pending
        waiting = asyncio.create_task(runtime._wait_for_final_output(pending))
        await asyncio.sleep(0)
        writer.append("final-a", Output((), "complete"))
        await waiting

        assert delivered == ["final-a"]
        assert gate.committed == [pending.request_id]
        assert gate.aborted == []
        assert runtime.pending is None
    finally:
        log.close()


@pytest.mark.asyncio
async def test_denied_prepare_does_not_block_next_restart_call(tmp_path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    state = log.owner("tool-execution")
    tasks = Tasks()
    runtime = RestartRuntime(cast(Any, object()))
    started: list[PendingRestart] = []

    async def start(pending: PendingRestart) -> None:
        started.append(pending)

    runtime.start = start  # type: ignore[method-assign]
    calls = 0

    async def authorize(_binding: str, _arguments: Mapping[str, object]) -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Denied("policy denied")
        return {"allowed": True}

    @asynccontextmanager
    async def open_tool(_binding: str):
        yield RestartTool(runtime)

    execution = ToolExecution(state, tasks, open_tool, authorize, task_key="tools")
    log.save_binding("binding-a", {})
    output = log.writer(
        "session-a", author="agent", source="conversation", body_types=(Output,),
        content={}, check_call=lambda _call: None,
    )

    def reply(message_id: str, call_id: str, reason: str) -> MessageReply:
        call = output.append(call_id, Output((ToolCall("binding-a", {"reason": reason}),), "continue"))
        ref = CallRef(call.message_id, 0)
        result_writer = log.writer(
            "session-a", author="tool", source="conversation", body_types=(ToolResult,),
            content={"text": lambda _part: ContentReferences()}, call_ref=ref,
        )
        return MessageReply(message_id, ref, log.reader("session-a"), result_writer, lambda: None)

    try:
        denied = await execution.execute_call(reply("result-a", "call-a", "first"))
        assert denied.outcome == "denied"
        assert runtime.pending is None

        succeeded = await execution.execute_call(reply("result-b", "call-b", "second"))
        assert succeeded.outcome == "success"
        assert [item.call_ref.message_id for item in started] == ["call-b"]
    finally:
        await tasks.close()
        log.close()

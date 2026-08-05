from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from agent.control.errors import ThreadBusyError
from agent.control.events import TurnEvent
from agent.control.models import (
    TurnItem,
    TurnItemKind,
    TurnRecord,
    TurnRequest,
    TurnResult,
    TurnStatus,
    TurnUsage,
)
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from session.store import SessionStore


def _assert_single_terminal(runtime: ConversationRuntime, turn_id: str) -> None:
    terminal = [
        event
        for event in runtime._history[turn_id]
        if event.method == "turn/completed"
    ]
    assert len(terminal) == 1


@pytest.mark.asyncio
async def test_runtime_persists_events_and_terminal_result(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest) -> str:
        return f"reply:{request.input}"

    runtime = ConversationRuntime(store, execute)
    handle = await runtime.start_turn(TurnRequest("programmatic:test", "hello"))
    events = [event async for event in handle.events()]
    result = await handle.result()

    assert [event.method for event in events if event.method.startswith("turn/")] == [
        "turn/queued",
        "turn/started",
        "turn/completed",
    ]
    assert result.status is TurnStatus.COMPLETED
    assert result.final_response == "reply:hello"
    assert store.read_turn(handle.id) is not None
    _assert_single_terminal(runtime, handle.id)
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_runtime_rejects_same_thread_and_interrupts_exact_turn(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    reached = asyncio.Event()

    async def execute(_request: TurnRequest) -> str:
        reached.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    runtime = ConversationRuntime(store, execute)
    first = await runtime.start_turn(TurnRequest("programmatic:test", "held"))
    await reached.wait()
    with pytest.raises(ThreadBusyError):
        _ = await runtime.start_turn(TurnRequest("programmatic:test", "conflict"))
    record = await first.interrupt()
    assert record.status is TurnStatus.INTERRUPTED
    _assert_single_terminal(runtime, first.id)
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_queued_interrupt_becomes_cancelled(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(_request: TurnRequest) -> str:
        return "done"

    runtime = ConversationRuntime(store, execute)
    queued = await runtime.start_turn(TurnRequest("programmatic:two", "queued"))
    record = await queued.interrupt()
    assert record.status is TurnStatus.CANCELLED
    _assert_single_terminal(runtime, queued.id)
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_runtime_executes_different_threads_concurrently(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    both_started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    max_active = 0

    async def execute(_request: TurnRequest) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            both_started.set()
        try:
            await release.wait()
            return "done"
        finally:
            active -= 1

    runtime = ConversationRuntime(store, execute)
    first = await runtime.start_turn(TurnRequest("programmatic:one", "first"))
    second = await runtime.start_turn(TurnRequest("programmatic:two", "second"))
    await asyncio.wait_for(both_started.wait(), timeout=1)

    assert max_active == 2
    release.set()
    first_result, second_result = await asyncio.gather(first.result(), second.result())
    assert first_result.status is TurnStatus.COMPLETED
    assert second_result.status is TurnStatus.COMPLETED
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_runtime_persists_structured_tool_items_and_usage(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    tool_item = TurnItem(
        TurnItemKind.TOOL_CALL,
        "item_tool_1",
        {"callId": "call-1", "name": "lookup", "status": "completed"},
    )
    usage = TurnUsage(
        input_tokens=8,
        output_tokens=3,
        request_count=1,
        covered_request_count=1,
        coverage="exact",
    )

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        return ControlExecutionResult("answer", [tool_item], ["ans", "wer"], usage)

    runtime = ConversationRuntime(store, execute)
    handle = await runtime.start_turn(TurnRequest("programmatic:structured", "hello"))
    events = [event async for event in handle.events()]
    result = await handle.result()

    assert [item.kind for item in result.items] == [
        TurnItemKind.USER_MESSAGE,
        TurnItemKind.TOOL_CALL,
        TurnItemKind.ASSISTANT_MESSAGE,
    ]
    assert result.usage == usage
    assert [
        event.data["delta"]
        for event in events
        if event.method == "item/assistantMessage/delta"
    ] == ["ans", "wer"]
    tool_events = [event for event in events if event.data.get("item") == tool_item.to_dict()]
    assert [event.method for event in tool_events] == ["item/started", "item/completed"]
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_runtime_replays_large_delta_stream_without_detaching_subscriber(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    deltas = [f"chunk-{sequence}" for sequence in range(300)]
    response = "".join(deltas)

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        return ControlExecutionResult(response=response, deltas=deltas)

    runtime = ConversationRuntime(store, execute)
    handle = await runtime.start_turn(
        TurnRequest("programmatic:large-delta-stream", "hello")
    )
    events: list[TurnEvent] = []
    async for event in handle.events():
        if event.method == "turn/completed":
            persisted = store.read_turn(handle.id)
            assert persisted is not None
            assert persisted.status is TurnStatus.COMPLETED
        events.append(event)
    result = await handle.result()

    delta_events = [
        event
        for event in events
        if event.method == "item/assistantMessage/delta"
    ]
    assistant_item_id = cast(str, delta_events[0].data["itemId"])
    assistant_events = [
        event
        for event in events
        if event.data.get("itemId") == assistant_item_id
        or (
            isinstance(event.data.get("item"), dict)
            and event.data["item"].get("id") == assistant_item_id
        )
    ]
    assert [event.data["delta"] for event in delta_events] == deltas
    assert [event.data["sequence"] for event in delta_events] == list(range(300))
    assert [event.method for event in assistant_events] == [
        "item/started",
        *(["item/assistantMessage/delta"] * 300),
        "item/completed",
    ]
    assert events[-1].method == "turn/completed"
    assert result.status is TurnStatus.COMPLETED
    assert result.final_response == response
    assert [event.method for event in events].count("turn/completed") == 1
    _assert_single_terminal(runtime, handle.id)
    await runtime.shutdown()
    store.close()


async def _executor_with_open_tool(
    request: TurnRequest,
    started: asyncio.Event,
    *,
    fail: bool = False,
) -> str:
    emit = request.metadata["_controlItemEvent"]
    assert callable(emit)
    emit(
        "item/started",
        TurnItem(
            TurnItemKind.TOOL_CALL,
            "tool-x",
            {
                "callId": "call-x",
                "name": "lookup",
                "arguments": {"query": "x"},
                "status": "in_progress",
            },
        ),
    )
    started.set()
    if fail:
        raise RuntimeError("executor failed")
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


def _assert_closed_tool_lifecycle(
    result: TurnRecord | TurnResult,
    events: Sequence[TurnEvent],
    expected_status: TurnStatus,
) -> None:
    tool = next(item for item in result.items if item.id == "tool-x")
    assert tool.data == {
        "callId": "call-x",
        "name": "lookup",
        "arguments": {"query": "x"},
        "status": expected_status.value,
    }
    tool_events = [
        event
        for event in events
        if isinstance(event.data.get("item"), dict)
        and event.data["item"].get("id") == "tool-x"
    ]
    assert [event.method for event in tool_events] == ["item/started", "item/completed"]
    assert tool_events[-1].data["item"]["data"]["status"] == expected_status.value


@pytest.mark.asyncio
async def test_interrupt_closes_and_persists_open_tool_item(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    started = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        return await _executor_with_open_tool(request, started)

    runtime = ConversationRuntime(store, execute)
    handle = await runtime.start_turn(TurnRequest("programmatic:interrupt-item", "hello"))
    await started.wait()
    result = await handle.interrupt()
    events = [event async for event in handle.events()]

    assert result.status is TurnStatus.INTERRUPTED
    _assert_closed_tool_lifecycle(result, events, TurnStatus.INTERRUPTED)
    assert store.read_turn(handle.id) == result
    _assert_single_terminal(runtime, handle.id)
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_shutdown_cancel_closes_and_persists_open_tool_item(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    started = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        return await _executor_with_open_tool(request, started)

    runtime = ConversationRuntime(store, execute)
    handle = await runtime.start_turn(TurnRequest("programmatic:cancel-item", "hello"))
    await started.wait()
    await runtime.shutdown()
    result = await handle.result()
    events = [event async for event in handle.events()]

    assert result.status is TurnStatus.CANCELLED
    _assert_closed_tool_lifecycle(result, events, TurnStatus.CANCELLED)
    assert store.read_turn(handle.id) is not None
    _assert_single_terminal(runtime, handle.id)
    store.close()


@pytest.mark.asyncio
async def test_exception_closes_and_persists_open_tool_item(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    started = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        return await _executor_with_open_tool(request, started, fail=True)

    runtime = ConversationRuntime(store, execute)
    handle = await runtime.start_turn(TurnRequest("programmatic:failed-item", "hello"))
    await started.wait()
    result = await handle.result()
    events = [event async for event in handle.events()]

    assert result.status is TurnStatus.FAILED
    _assert_closed_tool_lifecycle(result, events, TurnStatus.FAILED)
    _assert_single_terminal(runtime, handle.id)
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_fatal_runtime_failure_is_delivered_to_subscriber_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(_request: TurnRequest) -> str:
        return "unreachable"

    runtime = ConversationRuntime(store, execute)
    handle = await runtime.start_turn(TurnRequest("programmatic:fatal", "hello"))
    queue = asyncio.Queue()
    runtime._subscribers[handle.id].add(queue)
    original_transition = cast(Any, store.transition_turn)

    def fail_start(*args: object, **kwargs: object):
        if kwargs.get("status") is TurnStatus.IN_PROGRESS:
            raise RuntimeError("transition failed")
        return original_transition(*args, **kwargs)

    monkeypatch.setattr(store, "transition_turn", fail_start)
    calls = 0
    original_fail_streams = runtime._fail_streams

    def count_failure(turn_id: str, error: BaseException) -> None:
        nonlocal calls
        calls += 1
        original_fail_streams(turn_id, error)

    monkeypatch.setattr(runtime, "_fail_streams", count_failure)
    with pytest.raises(RuntimeError, match="transition failed"):
        _ = await handle.result()
    assert calls == 1
    assert queue.qsize() == 1
    assert isinstance(queue.get_nowait(), RuntimeError)
    await runtime.shutdown()
    store.close()

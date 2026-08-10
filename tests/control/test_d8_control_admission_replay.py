from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent.control.errors import ControlAdmissionError
from agent.control.models import TurnItem, TurnItemKind, TurnRequest, TurnStatus
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.control.protocol.errors import SERVER_OVERLOADED
from agent.control.protocol.router import ConnectionRouter
from agent.control.service import ControlService
from session.store import SessionStore
from session.manager import SessionManager


@pytest.mark.asyncio
async def test_control_admission_rejects_only_current_start_and_releases_after_terminal(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        started.set()
        await release.wait()
        return request.input

    runtime = ConversationRuntime(store, execute, max_active_turns=1)
    first = await runtime.start_turn(TurnRequest("programmatic:one", "one"))
    await started.wait()

    with pytest.raises(ControlAdmissionError, match="resource-exhausted"):
        await runtime.start_turn(TurnRequest("programmatic:two", "two"))
    assert store.list_turns("programmatic:two", limit=10) == []
    assert runtime.admission_snapshot()["turns"] == 1

    release.set()
    assert (await first.result()).status is TurnStatus.COMPLETED
    assert runtime.admission_snapshot()["turns"] == 0
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_immediate_queued_interrupt_releases_control_admission(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest) -> str:
        return request.input

    runtime = ConversationRuntime(store, execute, max_active_turns=1)
    handle = await runtime.start_turn(TurnRequest("programmatic:queued", "hello"))
    record = await handle.interrupt()
    assert record.status is TurnStatus.CANCELLED
    assert runtime.admission_snapshot()["turns"] == 0
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_continued_attempt_admission_counts_only_current_input(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    first_started = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        if request.metadata["attemptOrdinal"] == 0:
            emit = request.metadata["_controlItemEvent"]
            item = TurnItem(
                TurnItemKind.TOOL_CALL,
                "large-tool",
                {
                    "callId": "large-call",
                    "name": "lookup",
                    "arguments": {},
                    "status": "success",
                    "resultPreview": "x" * 8192,
                },
            )
            emit("item/started", item)
            emit("item/completed", item)
            first_started.set()
            await asyncio.Event().wait()
        return "continued"

    runtime = ConversationRuntime(store, execute, max_active_bytes=2048)
    first = await runtime.start_turn(TurnRequest("programmatic:bounded", "u1"))
    await first_started.wait()
    assert (await first.interrupt()).status is TurnStatus.INTERRUPTED

    second = await runtime.start_turn(TurnRequest("programmatic:bounded", "u2"))

    assert (await second.result()).status is TurnStatus.COMPLETED
    assert runtime.admission_snapshot()["bytes"] == 0
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_router_maps_control_capacity_to_existing_overloaded_error(
    tmp_path: Path,
) -> None:
    sessions = SessionManager(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(request: TurnRequest) -> str:
        started.set()
        await release.wait()
        return request.input

    runtime = ConversationRuntime(sessions.control_store, execute, max_active_turns=1)
    service = ControlService(runtime, sessions, tmp_path)
    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    router = ConnectionRouter(service, send)
    await router.handle_line(
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":'
        b'{"protocolVersion":"1.0","clientInfo":{"name":"test","version":"1"}}}\n'
    )
    await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized","params":{}}\n')
    first_thread = service.start_thread({})["id"]
    second_thread = service.start_thread({})["id"]
    await router.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "turn/start",
                "params": {"threadId": first_thread, "input": "first", "metadata": {}},
            }
        ).encode()
        + b"\n"
    )
    await started.wait()
    await router.handle_line(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "turn/start",
                "params": {"threadId": second_thread, "input": "second", "metadata": {}},
            }
        ).encode()
        + b"\n"
    )
    response = next(item for item in sent if item.get("id") == 3)
    error = response["error"]
    assert isinstance(error, dict)
    assert error["code"] == SERVER_OVERLOADED
    assert "resource-exhausted" in error["message"]
    assert error["data"] == {"retryable": True, "failure": "operation_rejected"}
    assert sessions.control_store.list_turns(str(second_thread), limit=10) == []
    release.set()
    await router.close()
    await runtime.shutdown()
    sessions.close()


@pytest.mark.asyncio
async def test_replay_eviction_reports_snapshot_without_dropping_live_events(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    deltas = [f"d{index}" for index in range(8)]

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        return ControlExecutionResult("".join(deltas), deltas=deltas)

    runtime = ConversationRuntime(
        store,
        execute,
        replay_events_per_turn=4,
        replay_bytes_per_turn=4096,
        replay_bytes_global=4096,
    )
    handle = await runtime.start_turn(TurnRequest("programmatic:replay", "hello"))
    live_events = [event async for event in handle.events()]
    assert len([event for event in live_events if event.method == "item/assistantMessage/delta"]) == 8
    assert live_events[-1].method == "turn/completed"

    replay_events = [event async for event in handle.events()]
    assert replay_events[0].method == "replay/truncated"
    assert replay_events[0].data["replay_status"] == "replay_truncated"
    assert replay_events[0].data["snapshot"]["status"] == "completed"
    assert replay_events[-1].method == "turn/completed"
    assert len(runtime._history[handle.id]) <= 4
    assert store.read_turn(handle.id) is not None
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_global_replay_index_has_no_stale_nodes_after_multi_turn_eviction(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(_request: TurnRequest) -> ControlExecutionResult:
        return ControlExecutionResult("x" * 64, deltas=["x" * 8] * 10)

    runtime = ConversationRuntime(
        store,
        execute,
        replay_events_per_turn=4,
        replay_bytes_per_turn=4096,
        replay_bytes_global=1024,
    )
    handles = []
    for index in range(8):
        handle = await runtime.start_turn(TurnRequest(f"programmatic:multi-{index}", "hello"))
        handles.append(handle)
        await handle.result()

    retained_events = sum(len(history) for history in runtime._history.values())
    assert len(runtime._replay_order) == retained_events
    assert runtime.replay_bytes <= 1024
    assert all(
        key in runtime._replay_order
        for turn_id, sequences in runtime._history_sequences.items()
        for key in ((turn_id, sequence) for sequence in sequences)
    )
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_terminal_replay_expiry_reads_authoritative_store_snapshot(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(_request: TurnRequest) -> str:
        return "done"

    runtime = ConversationRuntime(store, execute, terminal_replay_ttl_seconds=0.001)
    handle = await runtime.start_turn(TurnRequest("programmatic:expired", "hello"))
    assert (await handle.result()).status is TurnStatus.COMPLETED
    await asyncio.sleep(0.01)

    replay_events = [event async for event in handle.events()]
    assert replay_events[0].method == "replay/expired"
    assert replay_events[0].data["replay_status"] == "replay_expired"
    assert replay_events[0].data["snapshot"]["status"] == "completed"
    assert runtime._history.get(handle.id) is None
    assert store.read_turn(handle.id) is not None
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_terminal_replay_reaper_evicts_without_followup_activity(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(_request: TurnRequest) -> str:
        return "done"

    runtime = ConversationRuntime(store, execute, terminal_replay_ttl_seconds=0.01)
    handle = await runtime.start_turn(TurnRequest("programmatic:reaper", "hello"))
    assert (await handle.result()).status is TurnStatus.COMPLETED
    reaper = runtime._replay_reaper_task
    assert reaper is not None

    for _ in range(100):
        if handle.id not in runtime._history:
            break
        await asyncio.sleep(0.005)

    assert handle.id not in runtime._history
    assert handle.id not in runtime._results
    assert handle.id not in runtime._subscribers
    assert handle.id not in runtime._terminal_replay_expiry
    assert store.read_turn(handle.id) is not None
    await runtime.shutdown()
    assert reaper.done()
    store.close()


@pytest.mark.asyncio
async def test_terminal_replay_reaper_surfaces_index_corruption(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(_request: TurnRequest) -> str:
        return "done"

    runtime = ConversationRuntime(store, execute, terminal_replay_ttl_seconds=0.01)
    handle = await runtime.start_turn(TurnRequest("programmatic:corrupt-replay", "hello"))
    assert (await handle.result()).status is TurnStatus.COMPLETED
    reaper = runtime._replay_reaper_task
    assert reaper is not None
    runtime._history_sequences[handle.id].clear()

    with caplog.at_level("CRITICAL", logger="agent.control.runtime"):
        with pytest.raises(RuntimeError, match="control replay index corrupted"):
            await asyncio.wait_for(asyncio.shield(reaper), timeout=1)
    assert "event=runtime_fatal owner=control.replay_reaper" in caplog.text
    assert runtime._replay_reaper_error is not None
    with pytest.raises(RuntimeError, match="control replay reaper failed"):
        await runtime.shutdown()
    assert store.read_turn(handle.id) is not None
    store.close()

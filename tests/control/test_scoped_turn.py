from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.control.errors import ThreadBusyError, TurnNotFoundError
from agent.control.models import TurnError, TurnRecord, TurnRequest, TurnStatus
from agent.control.runtime import ConversationRuntime
from agent.control.scoped_turn import ScopedTurnPort, TurnAcceptedReceipt
from agent.control.turn_scope import ToolGrant, TurnExecutionScope
from agent.plugin_composition.scoped_turns import PluginScopedTurns
from session.store import SessionStore


class _Scope:
    def __init__(self, counter: list[int]) -> None:
        self._counter = counter
        self._active = True
        counter[0] += 1

    @property
    def active(self) -> bool:
        return self._active

    def fork(self) -> _Scope:
        if not self._active:
            raise RuntimeError("scope closed")
        return _Scope(self._counter)

    async def release(self) -> None:
        if self._active:
            self._active = False
            self._counter[0] -= 1


def _queued_turn(turn_id: str, session_id: str) -> TurnRecord:
    return TurnRecord(
        id=turn_id,
        thread_id=session_id,
        status=TurnStatus.QUEUED,
        input=turn_id,
        created_at=datetime.now(UTC),
    )


def test_turn_scope_preload_must_be_exact_unique_and_authorized() -> None:
    with pytest.raises(ValueError, match="无首尾空白"):
        TurnExecutionScope(preloaded_tools=(" share_content",))
    with pytest.raises(ValueError, match="不得重复"):
        TurnExecutionScope(preloaded_tools=("share_content", "share_content"))
    with pytest.raises(ValueError, match="Tool grant 授权"):
        TurnExecutionScope(
            preloaded_tools=("share_content",),
            tool_grant=ToolGrant.only(("skip_content",)),
        )
    with pytest.raises(ValueError, match="terminal Tool 必须已 preload"):
        TurnExecutionScope(
            preloaded_tools=("share_content",),
            terminal_tools=("skip_content",),
            tool_grant=ToolGrant.only(("share_content", "skip_content")),
        )


@pytest.mark.asyncio
async def test_scoped_turn_handle_binds_acceptance_terminal_and_cleanup(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest) -> str:
        return f"reply:{request.input}"

    runtime = ConversationRuntime(store, execute)
    leases = [0]
    owner = _Scope(leases)
    handle = await ScopedTurnPort(runtime, owner).start(
        TurnRequest("programmatic:child", "hello")
    )

    assert leases == [2]
    assert handle.accepted.session_id == "programmatic:child"
    assert handle.accepted.turn_id == handle.id
    result = await handle.result()
    await handle.cleanup()

    assert result.status is TurnStatus.COMPLETED
    assert result.final_response == "reply:hello"
    assert leases == [1]
    await owner.release()
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_waiter_cancellation_does_not_cancel_scoped_turn(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    started = asyncio.Event()

    async def execute(_request: TurnRequest) -> str:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    runtime = ConversationRuntime(store, execute)
    leases = [0]
    owner = _Scope(leases)
    handle = await ScopedTurnPort(runtime, owner).start(
        TurnRequest("programmatic:cancel", "wait")
    )
    await started.wait()
    waiter = asyncio.create_task(handle.result())
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert leases == [2]
    result = await handle.interrupt()
    assert result.status is TurnStatus.INTERRUPTED
    assert leases == [1]
    await owner.release()
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_pre_admission_failure_releases_child_scope() -> None:
    class RejectingRuntime:
        async def start_turn(self, request: TurnRequest, **_: object) -> object:
            raise RuntimeError(f"rejected:{request.thread_id}")

    leases = [0]
    owner = _Scope(leases)
    port = ScopedTurnPort(RejectingRuntime(), owner)

    with pytest.raises(RuntimeError, match="rejected:programmatic:reject"):
        await port.start(TurnRequest("programmatic:reject", "no"))

    assert leases == [1]
    await owner.release()


@pytest.mark.asyncio
async def test_plugin_background_turn_acquires_and_releases_exact_scope(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(request: TurnRequest) -> str:
        return f"reply:{request.input}"

    runtime = ConversationRuntime(store, execute)
    session_id = "programmatic:background"
    store.create_session(key=session_id, metadata={"source": "scheduler"})
    leases = [0]

    async def acquire_scope() -> _Scope:
        return _Scope(leases)

    turns = PluginScopedTurns(
        runtime,
        store.create_session,
        store.get_session_meta,
        acquire_scope,
    )
    handle = await turns.start(
        session_id,
        "wake",
        scope=TurnExecutionScope(tool_source="scheduler"),
    )

    assert leases == [1]
    result = await handle.result()
    assert result.status is TurnStatus.COMPLETED
    assert result.final_response == "reply:wake"
    assert leases == [0]
    await runtime.shutdown()
    store.close()


def test_plugin_scoped_turns_reads_immutable_durable_statuses(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")

    async def execute(_request: TurnRequest) -> str:
        return "unused"

    runtime = ConversationRuntime(store, execute)
    service = PluginScopedTurns(runtime, store.create_session)
    session_id = "programmatic:read"
    expected = {
        "queued": TurnStatus.QUEUED,
        "active": TurnStatus.IN_PROGRESS,
        "completed": TurnStatus.COMPLETED,
        "failed": TurnStatus.FAILED,
        "cancelled": TurnStatus.CANCELLED,
        "interrupted": TurnStatus.INTERRUPTED,
    }
    for suffix, status in expected.items():
        turn_id = f"turn:{suffix}"
        store.create_turn(_queued_turn(turn_id, session_id))
        if status is TurnStatus.QUEUED:
            continue
        if status is TurnStatus.CANCELLED:
            store.transition_turn(
                turn_id,
                expected_status=TurnStatus.QUEUED,
                status=status,
            )
            continue
        store.transition_turn(
            turn_id,
            expected_status=TurnStatus.QUEUED,
            status=TurnStatus.IN_PROGRESS,
        )
        if status is TurnStatus.IN_PROGRESS:
            continue
        store.transition_turn(
            turn_id,
            expected_status=TurnStatus.IN_PROGRESS,
            status=status,
            error=(
                TurnError("fixture", "failed", True)
                if status is TurnStatus.FAILED
                else None
            ),
            final_response="done" if status is TurnStatus.COMPLETED else None,
        )

    for suffix, status in expected.items():
        view = service.read(TurnAcceptedReceipt(session_id, f"turn:{suffix}"))
        assert view.status is status
        with pytest.raises(FrozenInstanceError):
            view.status = TurnStatus.CANCELLED  # type: ignore[misc]

    with pytest.raises(TurnNotFoundError):
        service.read(TurnAcceptedReceipt(session_id, "turn:missing"))
    with pytest.raises(RuntimeError, match="candidate 验证期"):
        PluginScopedTurns.candidate_validation().read(
            TurnAcceptedReceipt(session_id, "turn:queued")
        )
    store.close()


def test_scoped_read_observes_restart_recovery_terminals(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SessionStore(path)
    session_id = "programmatic:restart"
    store.create_turn(_queued_turn("turn:queued-restart", session_id))
    store.create_turn(_queued_turn("turn:active-restart", session_id))
    store.transition_turn(
        "turn:active-restart",
        expected_status=TurnStatus.QUEUED,
        status=TurnStatus.IN_PROGRESS,
    )
    store.close()

    reopened = SessionStore(path)

    async def execute(_request: TurnRequest) -> str:
        return "unused"

    runtime = ConversationRuntime(reopened, execute)
    service = PluginScopedTurns(runtime, reopened.create_session)
    assert (
        service.read(TurnAcceptedReceipt(session_id, "turn:queued-restart")).status
        is TurnStatus.CANCELLED
    )
    assert (
        service.read(TurnAcceptedReceipt(session_id, "turn:active-restart")).status
        is TurnStatus.INTERRUPTED
    )
    reopened.close()


@pytest.mark.asyncio
async def test_scoped_fresh_turn_supersedes_failed_interaction_across_restart(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    session_id = "programmatic:fresh"

    async def fail(_request: TurnRequest) -> str:
        raise RuntimeError("first failed")

    runtime = ConversationRuntime(store, fail)
    first = await runtime.start_turn(TurnRequest(session_id, "old"))
    assert (await first.result()).status is TurnStatus.FAILED
    first_record = runtime.read_turn(session_id, first.id)
    await runtime.shutdown()

    observed_fresh: list[TurnRequest] = []

    async def fail_fresh(request: TurnRequest) -> str:
        observed_fresh.append(request)
        raise RuntimeError("fresh failed")

    runtime = ConversationRuntime(store, fail_fresh)
    owner = _Scope([0])
    fresh = await ScopedTurnPort(runtime, owner).start(TurnRequest(session_id, "new"))
    assert (await fresh.result()).status is TurnStatus.FAILED
    fresh_record = runtime.read_turn(session_id, fresh.id)
    assert fresh_record.metadata["interactionId"] == fresh.id
    assert fresh_record.metadata["attemptOrdinal"] == 0
    assert fresh_record.metadata["supersedesInteractionId"] == first_record.id
    assert "continuedFromTurnId" not in fresh_record.metadata
    assert observed_fresh[0].metadata["priorInputCount"] == 0
    await owner.release()
    await runtime.shutdown()

    observed_after_restart: list[TurnRequest] = []

    async def complete(request: TurnRequest) -> str:
        observed_after_restart.append(request)
        return "ok"

    runtime = ConversationRuntime(store, complete)
    next_turn = await runtime.start_turn(TurnRequest(session_id, "after"))
    assert (await next_turn.result()).status is TurnStatus.COMPLETED
    assert observed_after_restart[0].metadata["attemptOrdinal"] == 0
    assert observed_after_restart[0].metadata["priorInputCount"] == 0
    assert "continuedFromTurnId" not in observed_after_restart[0].metadata
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_passive_failed_attempt_still_continues_normally(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    session_id = "channel:passive"
    calls = 0
    observed: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> str:
        nonlocal calls
        calls += 1
        observed.append(request)
        if calls == 1:
            raise RuntimeError("retry")
        return "ok"

    runtime = ConversationRuntime(store, execute)
    first = await runtime.start_turn(TurnRequest(session_id, "one"))
    assert (await first.result()).status is TurnStatus.FAILED
    second = await runtime.start_turn(TurnRequest(session_id, "two"))
    assert (await second.result()).status is TurnStatus.COMPLETED

    assert observed[1].metadata["interactionId"] == first.id
    assert observed[1].metadata["attemptOrdinal"] == 1
    assert observed[1].metadata["continuedFromTurnId"] == first.id
    await runtime.shutdown()
    store.close()


@pytest.mark.asyncio
async def test_fixed_session_accepts_at_most_one_concurrent_scoped_turn(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(_request: TurnRequest) -> str:
        started.set()
        await release.wait()
        return "ok"

    runtime = ConversationRuntime(store, execute)
    owner = _Scope([0])
    port = ScopedTurnPort(runtime, owner)
    first = await port.start(TurnRequest("programmatic:fixed", "first"))
    await started.wait()
    with pytest.raises(ThreadBusyError):
        await port.start(TurnRequest("programmatic:fixed", "second"))
    assert len(store.list_turns("programmatic:fixed")) == 1

    release.set()
    assert (await first.result()).status is TurnStatus.COMPLETED
    await owner.release()
    await runtime.shutdown()
    store.close()

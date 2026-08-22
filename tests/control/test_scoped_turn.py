from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.control.models import TurnRequest, TurnStatus
from agent.control.runtime import ConversationRuntime
from agent.control.scoped_turn import ScopedTurnPort
from agent.control.turn_scope import TurnExecutionScope
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

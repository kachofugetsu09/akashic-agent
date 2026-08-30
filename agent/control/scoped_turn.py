from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

from agent.control.models import (
    TurnItem,
    TurnRecord,
    TurnRequest,
    TurnResult,
    TurnStatus,
)
from agent.control.turn_scope import TurnExecutionScope

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshotLease


class TurnAdmissionRetiredError(RuntimeError):
    """Report that an unaccepted child Turn must hand off to a newer Root."""


class RuntimeTurnHandle(Protocol):
    thread_id: str
    id: str

    async def result(self) -> TurnResult: ...

    async def interrupt(self) -> TurnRecord: ...


class ScopedTurnRuntime(Protocol):
    """Admit a Turn while retaining the exact caller-owned scope lease."""

    async def start_turn(
        self,
        request: TurnRequest,
        *,
        runtime_snapshot_lease: RuntimeSnapshotLease,
        execution_scope: TurnExecutionScope | None,
        fresh_interaction: bool,
    ) -> RuntimeTurnHandle: ...


@dataclass(frozen=True, slots=True)
class TurnAcceptedReceipt:
    """Identify the Turn after Core accepts custody."""

    session_id: str
    turn_id: str


@dataclass(frozen=True, slots=True)
class DurableTurnView:
    """Expose immutable durable Turn state without the SessionStore write surface."""

    session_id: str
    turn_id: str
    status: TurnStatus
    final_response: str | None
    error_type: str | None
    error_message: str | None
    error_retryable: bool | None
    items: tuple[TurnItem, ...] = ()

    @classmethod
    def from_record(cls, record: TurnRecord) -> DurableTurnView:
        error = record.error
        return cls(
            session_id=record.thread_id,
            turn_id=record.id,
            status=record.status,
            final_response=record.final_response,
            error_type=error.type if error is not None else None,
            error_message=error.message if error is not None else None,
            error_retryable=error.retryable if error is not None else None,
            items=tuple(record.items),
        )


class ScopedTurnHandle:
    """Expose one accepted Turn while Core settles terminal state and scope cleanup."""

    def __init__(self, inner: RuntimeTurnHandle, lease: RuntimeSnapshotLease) -> None:
        self._inner = inner
        self._lease = lease
        self._settlement = asyncio.create_task(
            self._settle(),
            name=f"scoped_turn_settlement:{inner.id}",
        )

    @property
    def accepted(self) -> TurnAcceptedReceipt:
        return TurnAcceptedReceipt(self._inner.thread_id, self._inner.id)

    @property
    def id(self) -> str:
        return self._inner.id

    @property
    def thread_id(self) -> str:
        return self._inner.thread_id

    async def result(self) -> TurnResult:
        """Wait for the typed terminal result without transferring cancellation."""

        return await asyncio.shield(self._settlement)

    async def interrupt(self) -> TurnResult:
        """Request interruption, then wait for the same terminal and cleanup owner."""

        _ = await self._inner.interrupt()
        return await self.result()

    async def cleanup(self) -> None:
        """Wait until terminal settlement has released the exact execution scope."""

        _ = await self.result()

    async def _settle(self) -> TurnResult:
        try:
            return await self._inner.result()
        finally:
            await self._lease.release()


class ScopedTurnPort:
    """Start Turns inside one exact scope and return Core-owned handles."""

    def __init__(
        self,
        runtime: ScopedTurnRuntime,
        scope: RuntimeSnapshotLease,
        *,
        execution_scope: TurnExecutionScope | None = None,
    ) -> None:
        self._runtime = runtime
        self._scope = scope
        self._execution_scope = execution_scope

    async def start(self, request: TurnRequest) -> ScopedTurnHandle:
        """Fork the exact scope, admit one Turn, and bind cleanup to its handle."""

        # 1. A closed owner scope cannot mint a later child Turn.
        if not self._scope.active:
            raise RuntimeError("scoped Turn owner scope 已释放")
        lease = self._scope.fork()

        # 2. Before acceptance this port still owns and releases the forked scope.
        try:
            handle = await self._runtime.start_turn(
                request,
                runtime_snapshot_lease=lease,
                execution_scope=self._execution_scope,
                fresh_interaction=True,
            )
        except BaseException:
            await lease.release()
            raise

        # 3. After acceptance the handle owns terminal settlement and scope cleanup.
        return ScopedTurnHandle(handle, lease)

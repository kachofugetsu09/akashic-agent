from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol, cast

from agent.control.models import TurnRecord, TurnRequest, TurnResult
from agent.control.turn_scope import TurnExecutionScope


class TurnAdmissionRetiredError(RuntimeError):
    """Report that an unaccepted child Turn must hand off to a newer Root."""


class TurnScopeLease(Protocol):
    """Keep one immutable execution scope alive until Turn cleanup."""

    @property
    def active(self) -> bool: ...

    def fork(self) -> TurnScopeLease: ...

    async def release(self) -> None: ...


class RuntimeTurnHandle(Protocol):
    thread_id: str
    id: str

    async def result(self) -> TurnResult: ...

    async def interrupt(self) -> TurnRecord: ...


@dataclass(frozen=True, slots=True)
class TurnAcceptedReceipt:
    """Identify the Turn after Core accepts custody."""

    session_id: str
    turn_id: str


class ScopedTurnHandle:
    """Expose one accepted Turn while Core settles terminal state and scope cleanup."""

    def __init__(self, inner: RuntimeTurnHandle, lease: TurnScopeLease) -> None:
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
        runtime: object,
        scope: TurnScopeLease,
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
            start_turn = getattr(self._runtime, "start_turn", None)
            if not callable(start_turn):
                raise RuntimeError("scoped Turn runtime 缺少 start_turn")
            kwargs: dict[str, object] = {"runtime_snapshot_lease": lease}
            if self._execution_scope is not None:
                kwargs["execution_scope"] = self._execution_scope
            pending = start_turn(request, **kwargs)
            if not inspect.isawaitable(pending):
                raise TypeError("scoped Turn runtime start_turn 必须返回 awaitable")
            handle = await cast(Awaitable[RuntimeTurnHandle], pending)
        except BaseException:
            await lease.release()
            raise

        # 3. After acceptance the handle owns terminal settlement and scope cleanup.
        return ScopedTurnHandle(handle, lease)

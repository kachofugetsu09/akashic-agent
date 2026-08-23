from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from agent.control.models import TurnStatus
from agent.control.scoped_turn import DurableTurnView, TurnAcceptedReceipt
from agent.control.timer import TimerHandle, TimerStatus
from agent.lifecycle.composition import CONTEXT_PREPARED_EVENT
from agent.lifecycle.types import BeforeTurnCtx
from agent.plugin_composition import (
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    SCOPED_TURNS,
    TIMERS,
    Context,
    EmitEventKey,
    PluginScopedTurns,
    PluginTimers,
    ServiceKey,
    TurnExecutionScope,
)
from plugins.wake.selection import DutyProposal, propose_content, propose_drift

api_version = 3
name = "wake"
version = "3.0.0"
desc = "Timer-driven Content and Drift scoped react"
author = "Akashic Core"
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


class ContentWakeServices(Protocol):
    def snapshot(self, now: datetime) -> Mapping[str, object]: ...

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def select(
        self,
        item_ref: Mapping[str, object],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]: ...

    def transition(
        self,
        selection_token: str,
        action: str,
        *,
        not_before: datetime | None = None,
    ) -> Mapping[str, object]: ...


class DriftWakeServices(Protocol):
    def snapshot(self, now: datetime) -> Mapping[str, object]: ...

    def select(
        self,
        ref: Mapping[str, object],
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]: ...

    def transition(self, token: str, action: str) -> Mapping[str, object]: ...

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...


CONTENT_WAKE = ServiceKey[ContentWakeServices]("content.wake.v1")
DRIFT_WAKE = ServiceKey[DriftWakeServices]("drift.wake.v1")
CONTENT_CHANGED = EmitEventKey[None]("content.changed")
inject = (TIMERS, SCOPED_TURNS, CONTENT_WAKE, DRIFT_WAKE)


class WakeRuntime:
    """Drive one durable due loop and settle its scoped Turn selections."""

    def __init__(
        self,
        timers: PluginTimers,
        turns: PluginScopedTurns,
        content: ContentWakeServices,
        drift: DriftWakeServices,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._timers = timers
        self._turns = turns
        self._content = content
        self._drift = drift
        self._now = now
        self._dirty = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._handle: TimerHandle | None = None
        self._closed = False

    async def start(self) -> None:
        """Reconcile accepted Turns, then recover the earliest durable deadline."""

        if self._closed:
            raise RuntimeError("Wake runtime 已关闭")
        if self._runner is not None:
            return
        await self._reconcile_selected()
        self._runner = asyncio.create_task(self._run(), name="wake:due-loop")

    async def close(self) -> None:
        """Cancel and await the owned wait/task without changing domain facts."""

        self._closed = True
        self._dirty.set()
        handle = self._handle
        runner = self._runner
        self._handle = None
        self._runner = None
        if handle is not None:
            _ = await handle.cancel()
        if runner is not None and runner is not asyncio.current_task():
            _ = await asyncio.gather(runner, return_exceptions=True)
        if handle is not None:
            await handle.cleanup()

    def content_changed(self) -> None:
        """Request a durable reread without treating the hint as authority."""

        self._dirty.set()

    async def prepare(self, ctx: BeforeTurnCtx) -> None:
        """Select Content then Drift for only one Wake scoped react."""

        if ctx.channel != "wake":
            return
        accepted = self._accepted(ctx)
        now = self._aware_now(ctx.timestamp)

        # 1. Content always owns the first proposal and CAS opportunity.
        content_snapshot = self._content.snapshot(now)
        content_items = _sequence(content_snapshot.get("items"), "Content items")
        content_proposal = propose_content(content_items, now=now)
        if content_proposal is not None:
            selected = self._content.select(
                content_proposal.ref,
                _integer(content_snapshot.get("snapshot_seq"), "snapshot_seq"),
                accepted,
                now,
            )
            if selected.get("selected") is not True:
                self._quiet(ctx)
                return
            token = _string(selected.get("selection_token"), "Content selection")
            if content_proposal.decision == "select":
                ctx.extra_hints.append(_hint("content", content_proposal))
                return
            self._content.transition(token, "await_change")

        # 2. Drift runs only after Content has no winning duty.
        drift_snapshot = self._drift.snapshot(now)
        drift_proposals = _sequence(drift_snapshot.get("proposals"), "Drift proposals")
        drift_proposal = propose_drift(drift_proposals)
        if drift_proposal is not None:
            selected = self._drift.select(drift_proposal.ref, accepted, now)
            if selected.get("selected") is not True:
                self._quiet(ctx)
                return
            token = _string(selected.get("selection_token"), "Drift selection")
            if drift_proposal.decision == "select":
                ctx.extra_hints.append(_hint("drift", drift_proposal))
                return
            action = "defer" if _proposal_next_due(drift_proposals, drift_proposal) else "await_change"
            self._drift.transition(token, action)
        self._quiet(ctx)

    async def _run(self) -> None:
        """Wait for deadline or hint, then admit at most one scoped Turn."""

        while not self._closed:
            self._dirty.clear()
            deadline = self._earliest_deadline()
            if deadline is None:
                await self._dirty.wait()
                continue
            handle = self._timers.schedule(deadline)
            self._handle = handle
            wait_timer = asyncio.create_task(handle.result())
            wait_dirty = asyncio.create_task(self._dirty.wait())
            done, pending = await asyncio.wait(
                {wait_timer, wait_dirty}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in pending:
                _ = await asyncio.gather(task, return_exceptions=True)
            if wait_dirty in done and wait_dirty.result():
                _ = await handle.cancel()
                await handle.cleanup()
                self._handle = None
                continue
            receipt = wait_timer.result()
            await handle.cleanup()
            self._handle = None
            if receipt.status is TimerStatus.CANCELLED or self._closed:
                continue
            if not self._has_due():
                continue
            await self._start_turn()

    async def _start_turn(self) -> None:
        """Admit one memoryless Wake Turn and settle its selected duty."""

        session = await self._turns.ensure_session(
            "wake:default",
            metadata={"programmatic": True, "wake": True},
        )
        handle = await self._turns.start(
            session,
            "Check durable Wake duties.",
            scope=TurnExecutionScope(
                tool_source="wake",
                stateless=True,
                memory_read=False,
                memory_write=False,
            ),
            channel="wake",
            chat_id="wake:default",
            sender="wake",
        )
        try:
            result = await handle.result()
            await self._settle_accepted(handle.accepted, _view_from_result(result))
        finally:
            await handle.cleanup()

    async def _reconcile_selected(self) -> None:
        """Forward-complete terminal selections or expose recovery order violations."""

        for owner, read_batch in (
            ("content", self._content.selected),
            ("drift", self._drift.selected),
        ):
            while receipts := read_batch(100):
                for receipt in receipts:
                    accepted = _accepted_receipt(receipt)
                    view = self._turns.read(accepted)
                    if view.status in {TurnStatus.QUEUED, TurnStatus.IN_PROGRESS}:
                        raise RuntimeError(
                            "Wake runtime.started 早于 Core Turn recovery/handoff: "
                            f"owner={owner}, accepted={accepted!r}, "
                            f"receipt={dict(receipt)!r}"
                        )
                    self._settle(owner, receipt, view)

    async def _settle_accepted(
        self, accepted: TurnAcceptedReceipt, view: DurableTurnView
    ) -> None:
        content = self._content.selection(
            {"session_id": accepted.session_id, "turn_id": accepted.turn_id}
        )
        if content is not None and content.get("status") == "selected":
            self._settle("content", content, view)
            return
        drift = self._drift.selection(
            {"session_id": accepted.session_id, "turn_id": accepted.turn_id}
        )
        if drift is not None and drift.get("status") == "selected":
            self._settle("drift", drift, view)

    def _settle(
        self,
        owner: str,
        receipt: Mapping[str, object],
        view: DurableTurnView,
    ) -> None:
        token = _string(receipt.get("selection_token"), "selection_token")
        if view.status is TurnStatus.COMPLETED:
            action = "ready_for_delivery"
        elif view.status is TurnStatus.FAILED and view.error_retryable is False:
            action = "invalidated"
        elif view.status in {
            TurnStatus.FAILED,
            TurnStatus.CANCELLED,
            TurnStatus.INTERRUPTED,
        }:
            action = "defer"
        else:
            raise RuntimeError(f"Wake 无法 settle 非终态 Turn: {view.status.value}")
        if owner == "content":
            deadline = self._aware_now() + timedelta(minutes=5) if action == "defer" else None
            transition = self._content.transition(token, action, not_before=deadline)
        else:
            if action == "defer" and receipt.get("next_due") is None:
                action = "await_change"
            transition = self._drift.transition(token, action)
        if transition.get("changed") is not True:
            raise RuntimeError(
                f"Wake selected transition 未提交: owner={owner}, "
                f"token={token}, result={dict(transition)!r}"
            )

    def _earliest_deadline(self) -> datetime | None:
        now = self._aware_now()
        content = self._content.snapshot(now).get("earliest_not_before")
        drift = self._drift.snapshot(now).get("next_due")
        deadlines = [_datetime(value) for value in (content, drift) if value is not None]
        return min(deadlines) if deadlines else None

    def _has_due(self) -> bool:
        now = self._aware_now()
        content = self._content.snapshot(now)
        if any(item.get("due") is True for item in _sequence(content.get("items"), "Content items")):
            return True
        drift = self._drift.snapshot(now)
        return any(item.get("due") is True for item in _sequence(drift.get("proposals"), "Drift proposals"))

    def _aware_now(self, value: datetime | None = None) -> datetime:
        instant = value or self._now()
        if instant.tzinfo is None:
            raise ValueError("Wake clock 必须带时区")
        return instant.astimezone(UTC)

    @staticmethod
    def _accepted(ctx: BeforeTurnCtx) -> dict[str, str]:
        if ctx.turn_id is None:
            raise RuntimeError("Wake lifecycle 缺少 accepted turn_id")
        return {"session_id": ctx.session_key, "turn_id": ctx.turn_id}

    @staticmethod
    def _quiet(ctx: BeforeTurnCtx) -> None:
        ctx.abort = True
        ctx.abort_reply = ""


async def apply(ctx: Context, config: object) -> None:
    """Compose Wake from Timer, scoped Turn, Content, Drift, and lifecycle."""

    _ = config
    runtime = WakeRuntime(
        ctx.require(TIMERS),
        ctx.require(SCOPED_TURNS),
        ctx.require(CONTENT_WAKE),
        ctx.require(DRIFT_WAKE),
    )

    def setup() -> object:
        return runtime.close

    _ = await ctx.effect(setup, label="wake-runtime")
    _ = await ctx.on(CONTENT_CHANGED, lambda _: runtime.content_changed())
    _ = await ctx.on(CONTEXT_PREPARED_EVENT, runtime.prepare)
    _ = await ctx.on(RUNTIME_STARTED, lambda _: runtime.start())
    _ = await ctx.on(RUNTIME_STOPPING, lambda _: runtime.close())


def _hint(owner: str, proposal: DutyProposal) -> str:
    return "Wake duty:\n" + json.dumps(
        {"owner": owner, "payload": dict(proposal.payload)},
        sort_keys=True,
        separators=(",", ":"),
    )


def _accepted_receipt(receipt: Mapping[str, object]) -> TurnAcceptedReceipt:
    accepted = receipt.get("accepted_turn")
    if not isinstance(accepted, Mapping):
        raise RuntimeError("Wake selected receipt 缺少 accepted_turn")
    return TurnAcceptedReceipt(
        _string(accepted.get("session_id"), "session_id"),
        _string(accepted.get("turn_id"), "turn_id"),
    )


def _view_from_result(result: object) -> DurableTurnView:
    status = getattr(result, "status", None)
    if not isinstance(status, TurnStatus):
        raise TypeError("Wake scoped Turn result 缺少 typed status")
    error = getattr(result, "error", None)
    return DurableTurnView(
        session_id=_string(getattr(result, "thread_id", None), "thread_id"),
        turn_id=_string(getattr(result, "id", None), "turn_id"),
        status=status,
        final_response=getattr(result, "final_response", None),
        error_type=getattr(error, "type", None),
        error_message=getattr(error, "message", None),
        error_retryable=getattr(error, "retryable", None),
    )


def _proposal_next_due(
    proposals: Sequence[Mapping[str, object]], proposal: DutyProposal
) -> bool:
    return any(
        item.get("ref") == proposal.ref and item.get("next_due") is not None
        for item in proposals
    )


def _sequence(value: object, field: str) -> Sequence[Mapping[str, object]]:
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise ValueError(f"{field} 必须是 Mapping sequence")
    return cast(Sequence[Mapping[str, object]], value)


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} 必须是整数")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} 必须是非空字符串")
    return value


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value)
    else:
        raise ValueError("Wake deadline 必须是 datetime 或 ISO 字符串")
    if result.tzinfo is None:
        raise ValueError("Wake deadline 必须带时区")
    return result.astimezone(UTC)

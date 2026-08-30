from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, cast
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, field_validator

from agent.plugin_composition import (
    BeforeTurnCtx,
    CONTEXT_PREPARED_EVENT,
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    SCOPED_TURNS,
    TIMERS,
    TOOL_CATALOG,
    DURABLE_DELIVERIES,
    Context,
    DurableDeliveryRequest,
    DurableDeliveryView,
    DurableTurnView,
    EmitEventKey,
    PluginScopedTurns,
    PluginDurableDeliveries,
    PluginTimers,
    PluginToolDefinition,
    PluginTools,
    PostCommitEffect,
    ScopedTurnHandle,
    ServiceKey,
    TimerHandle,
    TimerStatus,
    ToolExecutionContext,
    TurnAcceptedReceipt,
    TurnExecutionScope,
    TurnItem,
    TurnItemKind,
    TurnStatus,
    TurnStorage,
    ToolGrant,
    CONVERSATION_SEMANTIC_INTEREST,
    ConversationSemanticInterest,
)
from .pool import PoolResult, build_initial_score
from .legacy_rules import read_archived_rules
from .selection import DutyProposal, propose_content, propose_drift
from .state import ContentScore, WakeState

logger = logging.getLogger(__name__)

api_version = 3
name = "wake"
version = "3.3.0"
desc = "Timer-driven Content and Drift scoped react"
author = "Akashic Core"
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()
dashboard_module = "dashboard.py"
web_module = "web_module.js"
web_requires = ("workbench.panels.v2",)
web_provides = ()
web_contract_digests = {
    "workbench.panels.v2": "fb6417c9bf532c1fdb344767d06065d5d3293da85deb64eff1e8088889a33bcb",
}

_AttemptOutcome = Literal[
    "no_due",
    "content_insufficient",
    "admission_rejected",
    "shared",
    "model_skip",
    "deferred",
    "cancelled_after_fire",
    "delivery_unknown",
    "failed",
]

_WAKE_MAINTENANCE_INTERVAL = timedelta(minutes=5)
_CONTENT_MIN_RESIDENCE = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class _AdmissionAttempt:
    turn_owner: Literal["alert", "content", "drift"] | None
    outcome: _AttemptOutcome
    detail: str
    checked_owner: Literal["alert", "content", "drift"] | None = None


@dataclass(frozen=True, slots=True)
class _ContentPool:
    snapshot_seq: int
    items: tuple[Mapping[str, object], ...]
    active_count: int
    due_count: int
    expired_count: int
    scored_count: int

    @property
    def detail(self) -> str:
        return (
            f"Content 池 active={self.active_count}, due={self.due_count}, "
            f"expired={self.expired_count}, scored={self.scored_count}"
        )


class DeliveryTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str
    recipient: str
    session_id: str

    @field_validator("channel", "recipient", "session_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("Wake delivery target 必须非空且无首尾空白")
        return value


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delivery: DeliveryTarget | None = None
    timezone: str = "Asia/Shanghai"

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("Wake timezone 必须非空且无首尾空白")
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Wake timezone 无效: {value}") from error
        return value


ConfigModel = Config


class ContentWakeServices(Protocol):
    def snapshot(self, now: datetime) -> Mapping[str, object]: ...

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def expire(
        self,
        item_refs: Sequence[Mapping[str, object]],
        now: datetime,
    ) -> Mapping[str, object]: ...

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

    def select_batch(
        self,
        item_refs: Sequence[Mapping[str, object]],
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
        selected_refs: Sequence[Mapping[str, object]] | None = None,
    ) -> Mapping[str, object]: ...

    def mail_watermark(self) -> int: ...

    def alert_deadline(self, now: datetime) -> datetime | None: ...

    def alert_status(self, source_id: str, event_id: str) -> str | None: ...

    def select_alert(
        self, accepted_turn: Mapping[str, object], now: datetime
    ) -> Mapping[str, object] | None: ...

    def selected_alert(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def selected_alerts(self) -> tuple[Mapping[str, object], ...]: ...

    def expire_alert(self, source_id: str, event_id: str, now: datetime) -> bool: ...

    def defer_alert(
        self, source_id: str, event_id: str, not_before: datetime
    ) -> None: ...

    def close_alert(self, source_id: str, event_id: str, status: str) -> None: ...

    def active_context(self, now: datetime) -> tuple[Mapping[str, object], ...]: ...


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


class DeliveryServices(Protocol):
    def pending(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def settle(
        self, selection_token: str, settlement_ref: str
    ) -> Mapping[str, object]: ...


EVENTMAIL_WAKE = ServiceKey[ContentWakeServices]("eventmail.wake.v1")
EVENTMAIL_DELIVERY = ServiceKey[DeliveryServices]("eventmail.delivery.v1")
EVENTMAIL_ALERT_DELIVERY = ServiceKey[object]("eventmail.alert_delivery.v1")
DRIFT_WAKE = ServiceKey[DriftWakeServices]("drift.wake.v1")
DRIFT_DELIVERY = ServiceKey[DeliveryServices]("drift.delivery.v1")
MEMORY_RECALL = ServiceKey[object]("memory.recall.v1")
EVENTMAIL_CHANGED = EmitEventKey[None]("eventmail.changed")
inject = (
    TIMERS,
    SCOPED_TURNS,
    DURABLE_DELIVERIES,
    EVENTMAIL_WAKE,
    EVENTMAIL_DELIVERY,
    EVENTMAIL_ALERT_DELIVERY,
    DRIFT_WAKE,
    DRIFT_DELIVERY,
    TOOL_CATALOG,
    MEMORY_RECALL,
    CONVERSATION_SEMANTIC_INTEREST,
)

_SCREEN_CONTENT = "screen_content"
_SHARE_ALERT = "share_alert"
_SHARE_CONTENT = "share_content"
_SKIP_CONTENT = "skip_content"
_DECISION_TOOLS = frozenset({_SHARE_CONTENT, _SKIP_CONTENT})
_SCREEN_LIMIT = 8
_INVESTIGATION_STEP_BUDGET = 20
_WEB_FETCH = "web_fetch"


@dataclass(frozen=True, slots=True)
class _WakeDecision:
    action: Literal["share", "skip"]
    message: str | None = None
    item_ids: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _ScreenedItem:
    candidate_id: str
    initial_interest: str
    question: str


class WakeRuntime:
    """Drive one durable due loop and settle its scoped Turn selections."""

    def __init__(
        self,
        timers: PluginTimers,
        turns: PluginScopedTurns,
        content: ContentWakeServices,
        drift: DriftWakeServices,
        *,
        deliveries: PluginDurableDeliveries | None = None,
        content_delivery: DeliveryServices | None = None,
        drift_delivery: DeliveryServices | None = None,
        target: DeliveryTarget | None = None,
        state: WakeState | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        semantic_interest: ConversationSemanticInterest | None = None,
        tools: PluginTools | None = None,
        proactive_context: str | None = None,
        timezone: str = "Asia/Shanghai",
    ) -> None:
        self._timers = timers
        self._turns = turns
        self._deliveries = deliveries
        self._content = content
        self._content_delivery = content_delivery
        self._drift_delivery = drift_delivery
        self._drift = drift
        self._target = target
        self._state = state
        self._active_owner: Literal["alert", "content", "drift"] | None = None
        self._phase: (
            Literal["alert", "content_screen", "content_investigate", "drift"] | None
        ) = None
        self._active_alert: Mapping[str, object] | None = None
        self._admitted_content: tuple[int, tuple[Mapping[str, object], ...]] | None = (
            None
        )
        self._content_proposal: tuple[int, DutyProposal] | None = None
        self._screened_content: tuple[_ScreenedItem, ...] = ()
        self._flow_run_id: str | None = None
        self._now = now
        self._semantic_interest = semantic_interest
        self._tools = tools
        self._proactive_context = proactive_context
        self._timezone = ZoneInfo(timezone)
        self._dirty = asyncio.Event()
        self._dirty_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: asyncio.Task[None] | None = None
        self._handle: TimerHandle | None = None
        self._maintenance_runner: asyncio.Task[None] | None = None
        self._maintenance_handle: TimerHandle | None = None
        self._maintenance_lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> None:
        """Reconcile accepted Turns, then recover the earliest durable deadline."""

        if self._closed:
            raise RuntimeError("Wake runtime 已关闭")
        if self._state is None:
            raise RuntimeError("Wake runtime 启动缺少 durable state")
        if self._runner is not None:
            return
        loop = asyncio.get_running_loop()
        with self._dirty_lock:
            self._loop = loop
        self._state.close_interrupted_attempts(self._aware_now())
        await self._reconcile_alerts()
        await self._reconcile_selected()
        await self._reconcile_deliveries()
        self._runner = asyncio.create_task(self._run(), name="wake:due-loop")
        self._maintenance_runner = asyncio.create_task(
            self._run_maintenance(), name="wake:pool-maintenance"
        )

    async def close(self) -> None:
        """Cancel and await the owned wait/task without changing domain facts."""

        self._closed = True
        self._dirty.set()
        handle = self._handle
        maintenance_handle = self._maintenance_handle
        runner = self._runner
        maintenance_runner = self._maintenance_runner
        self._handle = None
        self._maintenance_handle = None
        self._runner = None
        self._maintenance_runner = None
        if handle is not None:
            _ = await handle.cancel()
        if maintenance_handle is not None:
            _ = await maintenance_handle.cancel()
        owned_runners = tuple(
            task
            for task in (runner, maintenance_runner)
            if task is not None and task is not asyncio.current_task()
        )
        if owned_runners:
            _ = await asyncio.gather(*owned_runners, return_exceptions=True)
        if handle is not None:
            await handle.cleanup()
        if maintenance_handle is not None:
            await maintenance_handle.cleanup()

    def content_changed(self) -> None:
        """Request a durable reread without treating the hint as authority."""

        with self._dirty_lock:
            loop = self._loop
            if loop is None:
                return
        loop.call_soon_threadsafe(self._dirty.set)

    def _commit_admitted_content(self) -> None:
        """Commit every due arrival only after one durable Content selection exists."""

        admitted = self._admitted_content
        state = self._state
        proposal_state = self._content_proposal
        if admitted is None or state is None or proposal_state is None:
            return
        state.commit_content_admission(admitted[1])

    async def prepare(self, ctx: BeforeTurnCtx) -> None:
        """Prepare the current Content screening/investigation or Drift phase."""

        if ctx.channel != "wake":
            return
        accepted = self._accepted(ctx)
        now = self._aware_now(ctx.timestamp)

        if self._phase == "alert":
            state = self._state
            if state is None:
                raise RuntimeError("Wake Alert phase 缺少 durable state")
            alert = self._content.select_alert(accepted, now)
            if alert is None:
                self._quiet(ctx)
                return
            self._active_alert = alert
            state.record_screen(
                run_id=_run_id(ctx.session_key, cast(str, ctx.turn_id)),
                owner="alert",
                candidates_seen=1,
                screening=(
                    {"payload": _mapping(alert.get("payload"), "Alert payload")},
                ),
                started_at=now,
            )
            ctx.extra_hints.append(self._alert_prompt(alert))
            return

        if self._phase == "content_investigate":
            proposal_state = self._content_proposal
            if proposal_state is None or not self._screened_content:
                raise RuntimeError("Wake Content investigation 缺少 screening state")
            snapshot_seq, proposal = proposal_state
            candidates = _content_candidates(proposal)
            by_id = {
                _candidate_id(_mapping(item.get("ref"), "Content candidate ref")): item
                for item in candidates
            }
            refs = tuple(
                _mapping(by_id[item.candidate_id].get("ref"), "Content candidate ref")
                for item in self._screened_content
            )
            selected = self._content.select_batch(refs, snapshot_seq, accepted, now)
            if selected.get("selected") is not True:
                self._quiet(ctx)
                return
            self._commit_admitted_content()
            ctx.extra_hints.append(self._investigation_prompt(proposal))
            return

        if self._phase == "drift":
            self._prepare_drift(ctx, accepted, now)
            return

        # 1. Content screening reads a frozen page but does not claim it yet.
        if self._active_owner != "drift":
            admitted = (
                self._admitted_content if self._active_owner == "content" else None
            )
            if self._active_owner == "content" and admitted is None:
                raise RuntimeError("Wake Content Turn 缺少已通过阈值的固定池快照")
            if admitted is None:
                content_snapshot = self._content.snapshot(now)
                content_items = _sequence(
                    content_snapshot.get("items"), "Content items"
                )
                content_snapshot_seq = _integer(
                    content_snapshot.get("snapshot_seq"), "snapshot_seq"
                )
            else:
                content_snapshot_seq, content_items = admitted
                content_snapshot = {}
            content_proposal = propose_content(content_items, now=now)
        else:
            content_snapshot = {}
            content_proposal = None
        if content_proposal is not None:
            if content_proposal.decision == "select":
                self._content_proposal = (content_snapshot_seq, content_proposal)
                ctx.extra_hints.append(self._screen_prompt(content_proposal))
                return
            candidates = _content_candidates(content_proposal)
            selected = self._content.select_batch(
                tuple(
                    _mapping(candidate.get("ref"), "Content candidate ref")
                    for candidate in candidates
                ),
                content_snapshot_seq,
                accepted,
                now,
            )
            if selected.get("selected") is not True:
                self._quiet(ctx)
                return
            self._commit_admitted_content()
            self._content.transition(
                _string(selected.get("selection_token"), "Content selection"),
                "await_change",
            )

        # 2. Drift runs only after Content has no winning duty.
        self._prepare_drift(ctx, accepted, now)

    def _prepare_drift(
        self,
        ctx: BeforeTurnCtx,
        accepted: Mapping[str, object],
        now: datetime,
    ) -> None:
        """Claim and prepare one legacy Drift duty."""

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
                ctx.extra_hints.append(self._drift_prompt(drift_proposal))
                return
            action = (
                "defer"
                if _proposal_next_due(drift_proposals, drift_proposal)
                else "await_change"
            )
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
                receipt = await handle.cancel()
                await handle.cleanup()
                self._handle = None
                if receipt.status is TimerStatus.CANCELLED:
                    continue
            else:
                receipt = wait_timer.result()
                await handle.cleanup()
                self._handle = None
            if receipt.status is TimerStatus.CANCELLED:
                continue
            state = self._state
            if state is None:
                raise RuntimeError("Wake Timer fire 缺少 durable state")
            attempt_id = _attempt_id(
                receipt.timer_id,
                receipt.deadline,
                receipt.settled_at,
            )
            state.begin_attempt(
                attempt_id=attempt_id,
                timer_id=receipt.timer_id,
                scheduled_for=receipt.deadline,
                fired_at=receipt.settled_at,
            )
            owner: Literal["alert", "content", "drift"] | None = None
            try:
                if self._closed:
                    state.finish_attempt(
                        attempt_id=attempt_id,
                        outcome="cancelled_after_fire",
                        owner=None,
                        detail="Timer 已触发，但 runtime 在检查职责前关闭",
                        completed_at=self._aware_now(),
                    )
                    continue
                state.set_attempt_mail_watermark(
                    attempt_id=attempt_id,
                    mail_watermark=self._content.mail_watermark(),
                )
                admission = await self._admit_attempt()
                owner = admission.turn_owner
                if owner is None:
                    state.finish_attempt(
                        attempt_id=attempt_id,
                        outcome=admission.outcome,
                        owner=admission.checked_owner,
                        detail=admission.detail,
                        completed_at=self._aware_now(),
                    )
                    continue
                outcome = await self._start_turn(owner)
            except Exception as error:
                state.finish_attempt(
                    attempt_id=attempt_id,
                    outcome="failed",
                    owner=owner,
                    detail=f"{type(error).__name__}: {error}",
                    completed_at=self._aware_now(),
                )
                raise
            state.finish_attempt(
                attempt_id=attempt_id,
                outcome=outcome,
                owner=owner,
                detail=f"{admission.detail}；{_attempt_detail(outcome)}",
                completed_at=self._aware_now(),
            )

    async def _run_maintenance(self) -> None:
        """Record one pool-only audit at least every five minutes."""

        state = self._state
        if state is None:
            raise RuntimeError("Wake maintenance 启动缺少 durable state")
        while not self._closed:
            deadline = state.next_maintenance_deadline(
                self._aware_now(), interval=_WAKE_MAINTENANCE_INTERVAL
            )
            handle = self._timers.schedule(deadline)
            self._maintenance_handle = handle
            receipt = await handle.result()
            await handle.cleanup()
            self._maintenance_handle = None
            if receipt.status is TimerStatus.CANCELLED:
                continue
            attempt_id = _attempt_id(
                receipt.timer_id, receipt.deadline, receipt.settled_at
            )
            state.begin_attempt(
                attempt_id=attempt_id,
                timer_id=receipt.timer_id,
                scheduled_for=receipt.deadline,
                fired_at=receipt.settled_at,
            )
            try:
                if self._closed:
                    state.finish_attempt(
                        attempt_id=attempt_id,
                        outcome="cancelled_after_fire",
                        owner=None,
                        detail="Timer 已触发，但 runtime 在维护 Content 池前关闭",
                        completed_at=self._aware_now(),
                    )
                    continue
                now = self._aware_now()
                state.set_attempt_mail_watermark(
                    attempt_id=attempt_id,
                    mail_watermark=self._content.mail_watermark(),
                )
                pool = await self._maintain_content_pool(now)
                new_count = state.unseen_due_count(pool.items, now)
                audit = state.audit_pool(pool.items, now=now)
                has_content_work = pool.due_count > 0 or pool.expired_count > 0
                detail = (
                    f"{_pool_detail(pool.detail, new_count, audit)}；"
                    "maintenance_only=1，不启动 Turn"
                )
                state.finish_attempt(
                    attempt_id=attempt_id,
                    outcome=("content_insufficient" if has_content_work else "no_due"),
                    owner=("content" if has_content_work else None),
                    detail=detail,
                    completed_at=self._aware_now(),
                )
            except Exception as error:
                state.finish_attempt(
                    attempt_id=attempt_id,
                    outcome="failed",
                    owner=None,
                    detail=f"{type(error).__name__}: {error}",
                    completed_at=self._aware_now(),
                )
                logger.exception(
                    "Wake pool maintenance failed; the next heartbeat stays scheduled"
                )

    async def _start_turn(
        self, owner: Literal["alert", "content", "drift"] | None = None
    ) -> _AttemptOutcome:
        """Run Content screening then investigation, or one Drift decision."""

        target_session = (
            self._target.session_id if self._target is not None else "wake:default"
        )
        session = await self._turns.ensure_session(
            target_session,
            metadata={"programmatic": True, "wake": True},
        )
        self._active_owner = owner
        outcome: _AttemptOutcome = "deferred"
        accepted: TurnAcceptedReceipt | None = None
        try:
            if owner == "alert":
                self._phase = "alert"
                handle = await self._start_scoped_turn(
                    session,
                    target_session,
                    tools=(_SHARE_ALERT,),
                    terminal_tools=(_SHARE_ALERT,),
                    disabled_prompt_sections=frozenset({"memory", "long_term_memory"}),
                    max_iterations=1,
                )
                accepted = handle.accepted
                try:
                    outcome = await self._settle_alert(
                        handle.accepted,
                        _view_from_result(await handle.result()),
                    )
                finally:
                    await handle.cleanup()
            elif owner == "content":
                outcome, accepted = await self._run_content_flow(
                    session, target_session
                )
            else:
                self._phase = "drift"
                handle = await self._start_scoped_turn(
                    session,
                    target_session,
                    tools=(_SHARE_CONTENT, _SKIP_CONTENT),
                    terminal_tools=(_SHARE_CONTENT, _SKIP_CONTENT),
                    disabled_prompt_sections=frozenset(),
                )
                accepted = handle.accepted
                try:
                    result = await handle.result()
                    outcome = (
                        await self._settle_accepted(
                            handle.accepted, _view_from_result(result)
                        )
                        or "deferred"
                    )
                finally:
                    await handle.cleanup()
            await self._reconcile_deliveries()
            if outcome == "shared":
                return self._delivery_outcome(accepted)
            return outcome
        finally:
            self._active_owner = None
            self._phase = None
            self._admitted_content = None
            self._content_proposal = None
            self._screened_content = ()
            self._active_alert = None
            self._flow_run_id = None

    async def _run_content_flow(
        self, session: str, target_session: str
    ) -> tuple[_AttemptOutcome, TurnAcceptedReceipt | None]:
        """Use one memory-aware screen Turn and one evidence Turn."""

        self._phase = "content_screen"
        screen = await self._start_scoped_turn(
            session,
            target_session,
            tools=(_SCREEN_CONTENT,),
            terminal_tools=(_SCREEN_CONTENT,),
            disabled_prompt_sections=frozenset(),
            max_iterations=1,
        )
        try:
            screen_view = _view_from_result(await screen.result())
            screened = _screen_decision(screen_view)
            proposal_state = self._content_proposal
            allowed_ids = (
                set()
                if proposal_state is None
                else {
                    cast(str, candidate["candidate_id"])
                    for candidate in _candidate_payloads(proposal_state[1])
                }
            )
            self._flow_run_id = _run_id(
                screen.accepted.session_id,
                screen.accepted.turn_id,
            )
            state = self._state
            if state is not None:
                state.record_screen(
                    run_id=self._flow_run_id,
                    owner="content",
                    candidates_seen=(
                        0
                        if proposal_state is None
                        else len(_content_candidates(proposal_state[1]))
                    ),
                    screening=tuple(
                        {
                            "candidate_id": item.candidate_id,
                            "initial_interest": item.initial_interest,
                            "question": item.question,
                        }
                        for item in (screened or ())
                    ),
                    started_at=self._aware_now(),
                )
            if screened is None or any(
                item.candidate_id not in allowed_ids for item in screened
            ):
                if state is not None:
                    state.record_decision(
                        run_id=self._flow_run_id,
                        decision="defer",
                        detail="初筛没有提交有效候选",
                        completed_at=self._aware_now(),
                    )
                await self._select_failed_screen(screen.accepted, screen_view)
                return "deferred", None
            self._screened_content = screened
        finally:
            await screen.cleanup()

        recall_tool = self._recall_tool_name()
        self._phase = "content_investigate"
        investigate = await self._start_scoped_turn(
            session,
            target_session,
            tools=(recall_tool, _WEB_FETCH, _SHARE_CONTENT, _SKIP_CONTENT),
            terminal_tools=(_SHARE_CONTENT, _SKIP_CONTENT),
            disabled_prompt_sections=frozenset({"memory", "long_term_memory"}),
            max_iterations=_INVESTIGATION_STEP_BUDGET,
        )
        try:
            investigate_view = _view_from_result(await investigate.result())
            self._record_content_decision(investigate_view)
            outcome = await self._settle_accepted(
                investigate.accepted,
                investigate_view,
            )
            return outcome or "deferred", investigate.accepted
        finally:
            await investigate.cleanup()

    def _record_content_decision(self, view: DurableTurnView) -> None:
        state = self._state
        run_id = self._flow_run_id
        if state is None or run_id is None:
            return
        try:
            decision = _content_decision(view)
        except ValueError as error:
            decision = None
            detail = str(error)
        else:
            detail = (
                "没有有效的 share/skip 终态"
                if decision is None
                else decision.message or decision.reason or ""
            )
        state.record_decision(
            run_id=run_id,
            decision=("defer" if decision is None else decision.action),
            detail=detail,
            completed_at=self._aware_now(),
        )

    async def _start_scoped_turn(
        self,
        session: str,
        target_session: str,
        *,
        tools: tuple[str, ...],
        terminal_tools: tuple[str, ...],
        disabled_prompt_sections: frozenset[str],
        max_iterations: int | None = None,
    ) -> ScopedTurnHandle:
        """Start one in-memory Wake Turn with an exact Tool grant."""

        return await self._turns.start(
            session,
            "Check durable Wake duties.",
            scope=TurnExecutionScope(
                preloaded_tools=tools,
                terminal_tools=terminal_tools,
                max_iterations=max_iterations,
                tool_source="wake",
                tool_grant=ToolGrant.only(tools),
                storage=TurnStorage.IN_MEMORY,
                post_commit_effect=PostCommitEffect.SUPPRESS,
                session_history_read=self._target is not None,
                disabled_prompt_sections=disabled_prompt_sections,
            ),
            channel="wake",
            chat_id=target_session,
            sender="wake",
        )

    async def _select_failed_screen(
        self,
        accepted: TurnAcceptedReceipt,
        view: DurableTurnView,
    ) -> None:
        """Make a failed screen recoverable through the existing defer path."""

        proposal_state = self._content_proposal
        if proposal_state is None:
            return
        snapshot_seq, proposal = proposal_state
        refs = tuple(
            _mapping(item.get("ref"), "Content candidate ref")
            for item in _content_candidates(proposal)[:_SCREEN_LIMIT]
        )
        selected = self._content.select_batch(
            refs,
            snapshot_seq,
            {"session_id": accepted.session_id, "turn_id": accepted.turn_id},
            self._aware_now(),
        )
        if selected.get("selected") is True:
            self._commit_admitted_content()
            await self._settle_accepted(accepted, view)

    async def _reconcile_alerts(self) -> None:
        """Resume Alert Turns accepted before the current process generation."""

        state = self._state
        if state is None:
            return
        for alert in self._content.selected_alerts():
            accepted = _accepted_receipt(alert)
            view = self._turns.read(accepted)
            if view.status in {TurnStatus.QUEUED, TurnStatus.IN_PROGRESS}:
                raise RuntimeError(
                    "Wake runtime.started 早于 Core Alert Turn recovery/handoff: "
                    f"accepted={accepted!r}"
                )
            await self._settle_alert(accepted, view, alert=alert)

    async def _settle_alert(
        self,
        accepted: TurnAcceptedReceipt,
        view: DurableTurnView,
        *,
        alert: Mapping[str, object] | None = None,
    ) -> _AttemptOutcome:
        """Settle one selected Alert through delivery or a five-minute retry."""

        state = self._state
        if state is None:
            raise RuntimeError("Wake Alert settlement 缺少 durable state")
        selected = alert or self._content.selected_alert(
            {"session_id": accepted.session_id, "turn_id": accepted.turn_id}
        )
        if selected is None:
            return "model_skip"
        source_id = _string(selected.get("source_id"), "Alert source_id")
        event_id = _string(selected.get("event_id"), "Alert event_id")
        delivery = (
            None if self._deliveries is None else self._deliveries.lookup(accepted)
        )
        if delivery is None or delivery.state == "prepared":
            _ = self._content.expire_alert(source_id, event_id, self._aware_now())
            selected = self._content.selected_alert(
                {"session_id": accepted.session_id, "turn_id": accepted.turn_id}
            )
            if selected is None:
                return "model_skip"
        run_id = _run_id(accepted.session_id, accepted.turn_id)
        state.record_screen(
            run_id=run_id,
            owner="alert",
            candidates_seen=1,
            screening=(
                {"payload": _mapping(selected.get("payload"), "Alert payload")},
            ),
            started_at=_datetime(selected.get("observed_at")),
        )
        try:
            decision = _alert_decision(view)
        except ValueError as error:
            logger.error(
                "Wake completed Alert Turn has invalid decision accepted=%s/%s error=%s",
                accepted.session_id,
                accepted.turn_id,
                error,
            )
            decision = None
        if view.status is TurnStatus.COMPLETED and decision is not None:
            state.record_decision(
                run_id=run_id,
                decision="share",
                detail=decision,
                completed_at=self._aware_now(),
            )
            await self._deliver_alert(accepted, selected, decision)
            return "shared"
        if view.status is TurnStatus.FAILED and view.error_retryable is False:
            state.record_decision(
                run_id=run_id,
                decision="skip",
                detail=view.error_message or "Alert Turn 不可重试失败",
                completed_at=self._aware_now(),
            )
            self._content.close_alert(source_id, event_id, "skipped")
            return "model_skip"
        state.record_decision(
            run_id=run_id,
            decision="defer",
            detail=view.error_message or "Alert 未提交 share_alert",
            completed_at=self._aware_now(),
        )
        self._content.defer_alert(
            source_id,
            event_id,
            self._aware_now() + timedelta(minutes=5),
        )
        return "deferred"

    async def _deliver_alert(
        self,
        accepted: TurnAcceptedReceipt,
        alert: Mapping[str, object],
        message: str,
    ) -> None:
        """Deliver one forced Alert and close its source identity exactly once."""

        state = self._state
        target = self._target
        deliveries = self._deliveries
        if state is None:
            raise RuntimeError("Wake Alert delivery 缺少 durable state")
        source_id = _string(alert.get("source_id"), "Alert source_id")
        event_id = _string(alert.get("event_id"), "Alert event_id")
        if target is None:
            self._content.close_alert(source_id, event_id, "skipped")
            return
        if deliveries is None:
            raise RuntimeError("Wake Alert delivery target 缺少 durable capability")
        logical_id = _logical_delivery_id(accepted)
        current = deliveries.lookup(accepted)
        if current is None:
            current = await deliveries.submit(
                DurableDeliveryRequest(
                    logical_delivery_id=logical_id,
                    accepted_turn=accepted,
                    target_service=EVENTMAIL_ALERT_DELIVERY.name,
                    channel=target.channel,
                    recipient=target.recipient,
                    projection_session_id=target.session_id,
                    body=message,
                    metadata={
                        "proactive": True,
                        "wake_type": "alert",
                        "source_id": source_id,
                        "event_id": event_id,
                        "effects": {
                            "post_commit": PostCommitEffect.SUPPRESS.value,
                        },
                    },
                )
            )
        elif current.state in {"prepared", "delivered"}:
            current = await deliveries.resume(accepted)
        if current.state == "projected":
            if self._content.alert_status(source_id, event_id) == "selected":
                self._content.close_alert(source_id, event_id, "delivered")
            _ = deliveries.confirm_settled(logical_id, f"alert:{source_id}:{event_id}")

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
    ) -> _AttemptOutcome | None:
        content = self._content.selection(
            {"session_id": accepted.session_id, "turn_id": accepted.turn_id}
        )
        if content is not None and content.get("status") == "selected":
            return self._settle("content", content, view)
        drift = self._drift.selection(
            {"session_id": accepted.session_id, "turn_id": accepted.turn_id}
        )
        if drift is not None and drift.get("status") == "selected":
            return self._settle("drift", drift, view)
        return None

    def _settle(
        self,
        owner: str,
        receipt: Mapping[str, object],
        view: DurableTurnView,
    ) -> _AttemptOutcome:
        token = _string(receipt.get("selection_token"), "selection_token")
        outcome: _AttemptOutcome
        if view.status is TurnStatus.COMPLETED:
            legacy_single = receipt.get("decision_format") == "legacy_single"
            try:
                decision = _content_decision(view, allow_legacy_items=legacy_single)
            except ValueError as exc:
                logger.error(
                    "Wake completed Turn has invalid decision accepted=%s/%s error=%s",
                    view.session_id,
                    view.turn_id,
                    exc,
                )
                decision = None
            if (
                owner == "content"
                and decision is not None
                and decision.action == "share"
                and not decision.item_ids
                and not legacy_single
            ):
                logger.error(
                    "Wake Content share decision has no candidate ids accepted=%s/%s",
                    view.session_id,
                    view.turn_id,
                )
                decision = None
            if decision is None:
                logger.error(
                    "Wake completed Turn without one valid decision accepted=%s/%s",
                    view.session_id,
                    view.turn_id,
                )
                action = "defer" if owner == "content" else "await_change"
                outcome = "model_skip"
            else:
                action = (
                    "ready_for_delivery" if decision.action == "share" else "release"
                )
                outcome = "shared" if decision.action == "share" else "model_skip"
        elif view.status is TurnStatus.FAILED and view.error_retryable is False:
            action = "invalidated"
            outcome = "model_skip"
        elif view.status in {
            TurnStatus.FAILED,
            TurnStatus.CANCELLED,
            TurnStatus.INTERRUPTED,
        }:
            action = "defer"
            outcome = "deferred"
        else:
            raise RuntimeError(f"Wake 无法 settle 非终态 Turn: {view.status.value}")
        if owner == "content":
            selected_refs = None
            if action == "ready_for_delivery" and decision is not None:
                try:
                    selected_refs = _selected_content_refs(
                        receipt,
                        decision.item_ids,
                        allow_legacy_single=legacy_single,
                    )
                except ValueError as exc:
                    logger.error(
                        "Wake Content share decision references invalid candidates "
                        "accepted=%s/%s error=%s",
                        view.session_id,
                        view.turn_id,
                        exc,
                    )
                    action = "defer"
                    outcome = "deferred"
            deadline = (
                self._aware_now() + timedelta(minutes=5) if action == "defer" else None
            )
            transition = self._content.transition(
                token,
                action,
                not_before=deadline,
                selected_refs=selected_refs,
            )
        else:
            if action in {"abandoned", "release"}:
                action = "await_change"
            if action == "defer" and receipt.get("next_due") is None:
                action = "await_change"
            transition = self._drift.transition(token, action)
        if transition.get("changed") is not True:
            raise RuntimeError(
                f"Wake selected transition 未提交: owner={owner}, "
                f"token={token}, result={dict(transition)!r}"
            )
        return outcome

    def _delivery_outcome(
        self, accepted: TurnAcceptedReceipt | None
    ) -> _AttemptOutcome:
        """Classify the durable delivery state after forward completion."""

        if accepted is None or self._deliveries is None:
            return "delivery_unknown"
        delivery = self._deliveries.lookup(accepted)
        return (
            "shared"
            if delivery is not None and delivery.state == "settled"
            else "delivery_unknown"
        )

    async def _reconcile_deliveries(self) -> None:
        """Forward-complete delivery, projection, and domain settlement windows."""

        target = self._target
        if target is None:
            return
        deliveries = self._deliveries
        if deliveries is None:
            raise RuntimeError("Wake delivery target 缺少 durable capability")

        # 1. Resume every Core row before creating missing domain rows.
        await self._resume_deliveries(deliveries)

        # 2. Create one stable logical delivery for each ready domain selection.
        for target_service, domain in self._delivery_domains():
            for pending in domain.pending(100):
                await self._deliver_pending(
                    pending,
                    deliveries,
                    target,
                    target_service=target_service,
                    domain=domain,
                )

    async def _resume_deliveries(self, deliveries: PluginDurableDeliveries) -> None:
        """Advance existing Core rows without consulting domain payload state."""

        for delivery in deliveries.recoverable():
            if delivery.target_service == EVENTMAIL_ALERT_DELIVERY.name:
                await self._resume_alert_delivery(delivery, deliveries)
                continue
            domain = self._delivery_domain(delivery.target_service)
            if domain is None:
                continue
            current = delivery
            if current.state in {"prepared", "delivered"}:
                current = await deliveries.resume(current.accepted_turn)
            if current.state == "projected":
                self._settle_projected(current, domain)

    async def _resume_alert_delivery(
        self,
        delivery: DurableDeliveryView,
        deliveries: PluginDurableDeliveries,
    ) -> None:
        state = self._state
        if state is None:
            raise RuntimeError("Wake Alert recovery 缺少 durable state")
        current = delivery
        source_id = _string(current.metadata.get("source_id"), "Alert source_id")
        event_id = _string(current.metadata.get("event_id"), "Alert event_id")
        if current.state == "prepared":
            _ = self._content.expire_alert(source_id, event_id, self._aware_now())
            source_status = self._content.alert_status(source_id, event_id)
            if source_status == "expired":
                _ = deliveries.cancel_prepared(
                    current.accepted_turn,
                    reason="Wake Alert expired before provider I/O",
                )
                return
            if source_status != "selected":
                raise RuntimeError(
                    "Wake prepared Alert 缺少 selected source row: "
                    f"source={source_id}, event={event_id}, status={source_status}"
                )
        if current.state in {"prepared", "delivered"}:
            current = await deliveries.resume(current.accepted_turn)
        if current.state != "projected":
            return
        status = self._content.alert_status(source_id, event_id)
        if status == "selected":
            self._content.close_alert(source_id, event_id, "delivered")
            status = "delivered"
        if status != "delivered":
            raise RuntimeError(
                "Wake projected Alert 缺少 selected/delivered source row"
            )
        _ = deliveries.confirm_settled(
            current.logical_delivery_id,
            f"alert:{source_id}:{event_id}",
        )

    async def _deliver_pending(
        self,
        pending: Mapping[str, object],
        deliveries: PluginDurableDeliveries,
        target: DeliveryTarget,
        *,
        target_service: str,
        domain: DeliveryServices,
    ) -> None:
        """Create or forward-complete one ready domain delivery."""

        accepted = _accepted_receipt(pending)
        current = deliveries.lookup(accepted)
        if current is None:
            turn = self._turns.read(accepted)
            if turn.status is not TurnStatus.COMPLETED:
                raise RuntimeError(
                    "Wake ready selection 不属于 completed Turn: "
                    f"{accepted!r}/{turn.status.value}"
                )
            decision = _content_decision(
                turn,
                allow_legacy_items=pending.get("decision_format") == "legacy_single",
            )
            if decision is None or decision.action != "share" or not decision.message:
                raise RuntimeError("Wake ready Turn 缺少 share_content 决策")
            metadata = _delivery_metadata(pending)
            current = await deliveries.submit(
                DurableDeliveryRequest(
                    logical_delivery_id=_logical_delivery_id(accepted),
                    accepted_turn=accepted,
                    target_service=target_service,
                    channel=target.channel,
                    recipient=target.recipient,
                    projection_session_id=target.session_id,
                    body=_message_with_source_links(decision.message, metadata),
                    metadata={
                        **metadata,
                        "proactive": True,
                        "effects": {
                            "post_commit": PostCommitEffect.SUPPRESS.value,
                        },
                    },
                )
            )
        elif current.state in {"prepared", "delivered"}:
            current = await deliveries.resume(accepted)
        elif current.state in {"rejected", "uncertain"}:
            logger.warning(
                "Wake durable delivery terminal without resend "
                "accepted=%s/%s delivery_id=%s state=%s receipt=%r",
                accepted.session_id,
                accepted.turn_id,
                current.logical_delivery_id,
                current.state,
                current.provider_receipt,
            )
        if current.state == "projected":
            self._settle_projected(current, domain)

    def _settle_projected(
        self,
        delivery: DurableDeliveryView,
        domain: DeliveryServices,
    ) -> None:
        """Commit domain state first, then close the Core settlement receipt."""

        deliveries = self._deliveries
        if deliveries is None:
            raise RuntimeError("Wake delivery settlement capability 缺失")
        selected = domain.lookup(
            {
                "session_id": delivery.accepted_turn.session_id,
                "turn_id": delivery.accepted_turn.turn_id,
            }
        )
        if selected is None:
            raise RuntimeError(
                "durable Wake delivery 缺少 accepted Turn selection: "
                f"{delivery.accepted_turn!r}"
            )
        token = _string(selected.get("selection_token"), "selection_token")
        settled = domain.settle(
            token,
            delivery.logical_delivery_id,
        )
        if settled.get("settled") is not True:
            raise RuntimeError(f"Wake delivery settlement 未提交: {dict(settled)!r}")
        receipt = _string(settled.get("receipt"), "Wake delivery receipt")
        _ = deliveries.confirm_settled(delivery.logical_delivery_id, receipt)

    def _delivery_domains(
        self,
    ) -> tuple[tuple[str, DeliveryServices], ...]:
        content = self._content_delivery
        drift = self._drift_delivery
        if content is None or drift is None:
            raise RuntimeError("Wake delivery target 缺少 Content/Drift capability")
        return (
            (EVENTMAIL_DELIVERY.name, content),
            (DRIFT_DELIVERY.name, drift),
        )

    def _delivery_domain(self, target_service: str) -> DeliveryServices | None:
        return dict(self._delivery_domains()).get(target_service)

    def _earliest_deadline(self) -> datetime | None:
        now = self._aware_now()
        content_snapshot = self._content.snapshot(now)
        content_items = _sequence(content_snapshot.get("items"), "Content items")
        content_deadlines: list[datetime] = []
        if self._state is None:
            raw_content = content_snapshot.get("earliest_not_before")
            if raw_content is not None:
                content_deadlines.append(_datetime(raw_content))
        else:
            unseen = self._state.unseen_deadline(content_items)
            if unseen is not None:
                content_deadlines.append(unseen)
        drift = self._drift.snapshot(now).get("next_due")
        alert_deadline = self._content.alert_deadline(now)
        deadlines = [
            *([alert_deadline] if alert_deadline is not None else []),
            *content_deadlines,
            *([_datetime(drift)] if drift is not None else []),
        ]
        return min(deadlines) if deadlines else None

    async def _admit_owner(self) -> Literal["alert", "content", "drift"] | None:
        """Choose the due owner for callers that do not need rejection detail."""

        return (await self._admit_attempt()).turn_owner

    async def _admit_attempt(
        self,
    ) -> _AdmissionAttempt:
        """Maintain the Content pool, then choose at most one due owner."""

        now = self._aware_now()
        pool = await self._maintain_content_pool(now)
        items = pool.items
        pool_detail = pool.detail
        state = self._state
        alert_deadline = self._content.alert_deadline(now)
        if alert_deadline is not None and alert_deadline <= now:
            if state is None:
                audit = PoolResult(False, 0.0, 0.0, 1.0, 0, "")
                new_count = 0
            else:
                audit = state.audit_pool(items, now=now)
                new_count = state.unseen_due_count(items, now)
            return _AdmissionAttempt(
                "alert",
                "shared",
                f"{_pool_detail(pool_detail, new_count, audit)}；Alert 已到期",
                "alert",
            )
        if state is None and any(item.get("due") is True for item in items):
            return _AdmissionAttempt("content", "shared", pool_detail, "content")
        rejected: _AdmissionAttempt | None = None
        if state is not None and state.has_unseen_due(items, now):
            new_count = state.unseen_due_count(items, now)
            result = state.evaluate(
                items,
                snapshot_seq=pool.snapshot_seq,
                now=now,
            )
            if result.should_wake:
                self._admitted_content = (
                    pool.snapshot_seq,
                    tuple(items),
                )
                return _AdmissionAttempt(
                    "content",
                    "shared",
                    _pool_detail(pool_detail, new_count, result),
                    "content",
                )
            rejected = _AdmissionAttempt(
                None,
                "content_insufficient",
                _pool_detail(pool_detail, new_count, result),
                "content",
            )
        elif state is not None and (pool.due_count or pool.expired_count):
            audit = state.audit_pool(items, now=now)
            rejected = _AdmissionAttempt(
                None,
                "content_insufficient",
                f"{_pool_detail(pool_detail, 0, audit)}；"
                "本轮只维护池子，没有新 Content",
                "content",
            )
        drift = self._drift.snapshot(now)
        if any(
            item.get("due") is True
            for item in _sequence(drift.get("proposals"), "Drift proposals")
        ):
            detail = (
                "Drift 已到期"
                if rejected is None
                else f"{rejected.detail}；Drift 已到期"
            )
            return _AdmissionAttempt("drift", "shared", detail, "drift")
        if rejected is not None:
            return rejected
        audit = (
            state.audit_pool(items, now=now)
            if state is not None
            else PoolResult(False, 0.0, 0.0, 1.0, 0, "")
        )
        return _AdmissionAttempt(
            None,
            "no_due",
            f"{_pool_detail(pool_detail, 0, audit)}；没有到期职责",
            None,
        )

    async def _maintain_content_pool(self, now: datetime) -> _ContentPool:
        """Expire old low-mass Content and return one current pool view."""

        async with self._maintenance_lock:
            content = self._content.snapshot(now)
            items = _sequence(content.get("items"), "Content items")
            expired_count = 0
            scored_count = 0
            if self._state is not None:
                scored_count += len(self._state.unscored_due_items(items))
                items = await self._ensure_content_scores(items, now)
                expired_refs = self._state.expired_content_refs(
                    items,
                    now=now,
                    minimum_residence=_CONTENT_MIN_RESIDENCE,
                )
                if expired_refs:
                    expired = self._content.expire(expired_refs, now)
                    expired_items = _sequence(expired.get("expired"), "expired Content")
                    expired_count = len(expired_items)
                    content = self._content.snapshot(now)
                    items = _sequence(content.get("items"), "Content items")
                    scored_count += len(self._state.unscored_due_items(items))
                    items = await self._ensure_content_scores(items, now)
            return _ContentPool(
                snapshot_seq=_integer(content.get("snapshot_seq"), "snapshot_seq"),
                items=tuple(items),
                active_count=sum(
                    1 for item in items if item.get("status") in {"pending", "deferred"}
                ),
                due_count=sum(1 for item in items if item.get("due") is True),
                expired_count=expired_count,
                scored_count=scored_count,
            )

    async def _ensure_content_scores(
        self, items: Sequence[Mapping[str, object]], now: datetime
    ) -> tuple[Mapping[str, object], ...]:
        """Calculate and persist each due Content initial score exactly once."""

        state = self._state
        if state is None:
            return tuple(items)
        unscored = state.unscored_due_items(items)
        if not unscored:
            return state.scored_items(items)
        service = self._semantic_interest
        semantic_scores = (
            tuple(0.0 for _ in unscored)
            if service is None
            else await service.score(
                [_content_text(item) for item in unscored],
                cutoff=now.isoformat(),
            )
        )
        if len(semantic_scores) != len(unscored):
            raise RuntimeError("semantic interest 返回数量与 Content 不一致")
        records: list[ContentScore] = []
        for item, raw_semantic in zip(unscored, semantic_scores, strict=True):
            semantic = _semantic_score(raw_semantic)
            payload = _mapping(item.get("payload"), "Content item payload")
            base = _preprocess_interest(payload)
            interest = 1 - (1 - base) * (1 - semantic)
            ref = _mapping(item.get("ref"), "Content item ref")
            records.append(
                ContentScore(
                    source_id=_string(ref.get("source_id"), "Content source_id"),
                    item_id=_string(ref.get("item_id"), "Content item_id"),
                    revision=_string(ref.get("revision"), "Content revision"),
                    initial_score=build_initial_score(
                        interest,
                        has_published_at=bool(payload.get("published_at")),
                        wake_eligible=payload.get("wake_eligible") is not False,
                    ),
                    semantic_interest=semantic,
                    scored_at=now,
                )
            )
        state.record_content_scores(records)
        return state.scored_items(items)

    def _aware_now(self, value: datetime | None = None) -> datetime:
        instant = value or self._now()
        if instant.tzinfo is None:
            raise ValueError("Wake clock 必须带时区")
        return instant.astimezone(UTC)

    def _screen_prompt(self, proposal: DutyProposal) -> str:
        """Build the memory-aware first-stage Content prompt."""

        return (
            "【Wake Content 初筛】先读本轮已注入的 MEMORY.md 和下方主动偏好规则。"
            "只判断哪些候选可能让用户感兴趣，不调查事实真假，也不要调用外部工具。"
            "必须调用 screen_content，选择 1 到 "
            + str(_SCREEN_LIMIT)
            + " 条；每条写 initial_interest 和接下来最值得确认的 question。\n\n"
            + self._shared_prompt_context()
            + "\n\n候选：\n"
            + json.dumps(
                _candidate_payloads(proposal),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _investigation_prompt(self, proposal: DutyProposal) -> str:
        """Build the evidence-only second-stage Content prompt."""

        selected = {item.candidate_id: item for item in self._screened_content}
        candidates = [
            {
                **candidate,
                "initial_interest": selected[
                    cast(str, candidate["candidate_id"])
                ].initial_interest,
                "question": selected[cast(str, candidate["candidate_id"])].question,
            }
            for candidate in _candidate_payloads(proposal)
            if candidate["candidate_id"] in selected
        ]
        return (
            "【Wake Content 找证据】你总共有 "
            + str(_INVESTIGATION_STEP_BUDGET)
            + " 轮调查预算。不要重新读取或概括整个 MEMORY.md；初筛理由已经给你。"
            "用 "
            + self._recall_tool_name()
            + " 重点确认这是不是用户的雷点、用户是否真的喜欢；"
            "用 web_fetch 自由读取需要核实的网页。结合当前时间、主动偏好规则和上下文事件，"
            "最后必须且只能调用一次 share_content 或 skip_content。"
            "share_content 的 message 只能放可直接发送给用户的正文。\n\n"
            + self._shared_prompt_context(include_events=True)
            + "\n\n初筛结果与候选：\n"
            + json.dumps(
                candidates,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _drift_prompt(self, proposal: DutyProposal) -> str:
        return _hint("drift", proposal) + "\n\n" + self._shared_prompt_context()

    def _alert_prompt(self, alert: Mapping[str, object]) -> str:
        return (
            "【Wake Alert】这是来源明确上报的告警，不做 Content 兴趣初筛。"
            "结合主动偏好规则、当前时间和上下文事件，把它写成简洁、可行动的用户消息，"
            "然后必须调用 share_alert。\n\n"
            + self._shared_prompt_context(include_events=True)
            + "\n\n告警：\n"
            + json.dumps(
                dict(alert),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _shared_prompt_context(self, *, include_events: bool = False) -> str:
        local_now = self._aware_now().astimezone(self._timezone)
        rules = self._proactive_context or "（没有已归档的主动偏好规则）"
        prompt = (
            "当前时间："
            + local_now.isoformat()
            + "（时区 "
            + self._timezone.key
            + "）\n\nPROACTIVE_CONTEXT.md：\n"
            + rules
        )
        if include_events and self._state is not None:
            prompt += "\n\nContextEvent：\n" + json.dumps(
                self._content.active_context(self._aware_now()),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return prompt

    def _recall_tool_name(self) -> str:
        if self._tools is None:
            return "recall_memory"
        return self._tools.from_provide(MEMORY_RECALL)

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

    if not isinstance(config, Config):
        raise TypeError("Wake config 必须通过 ConfigModel 校验")
    tools = ctx.require(TOOL_CATALOG)
    state = WakeState(ctx.data_root / "wake.sqlite3")
    runtime = WakeRuntime(
        ctx.require(TIMERS),
        ctx.require(SCOPED_TURNS),
        ctx.require(EVENTMAIL_WAKE),
        ctx.require(DRIFT_WAKE),
        deliveries=ctx.require(DURABLE_DELIVERIES),
        content_delivery=ctx.require(EVENTMAIL_DELIVERY),
        drift_delivery=ctx.require(DRIFT_DELIVERY),
        target=config.delivery,
        state=state,
        semantic_interest=ctx.require(CONVERSATION_SEMANTIC_INTEREST),
        tools=tools,
        proactive_context=read_archived_rules(ctx.data_root),
        timezone=config.timezone,
    )

    async def screen_handler(
        context: ToolExecutionContext, arguments: Mapping[str, object]
    ) -> str:
        _validate_decision_context(
            context,
            (
                config.delivery.session_id
                if config.delivery is not None
                else "wake:default"
            ),
        )
        _parse_screen_arguments(arguments)
        return json.dumps(
            {"recorded": True, "turn_id": context.turn_id},
            ensure_ascii=False,
            sort_keys=True,
        )

    async def share_handler(
        context: ToolExecutionContext, arguments: Mapping[str, object]
    ) -> str:
        _validate_decision_context(
            context,
            (
                config.delivery.session_id
                if config.delivery is not None
                else "wake:default"
            ),
        )
        _validate_decision_argument(arguments, "message")
        return json.dumps(
            {"recorded": True, "turn_id": context.turn_id},
            ensure_ascii=False,
            sort_keys=True,
        )

    async def alert_handler(
        context: ToolExecutionContext, arguments: Mapping[str, object]
    ) -> str:
        _validate_decision_context(
            context,
            (
                config.delivery.session_id
                if config.delivery is not None
                else "wake:default"
            ),
        )
        _validate_decision_argument(arguments, "message")
        return json.dumps(
            {"recorded": True, "turn_id": context.turn_id},
            ensure_ascii=False,
            sort_keys=True,
        )

    async def skip_handler(
        context: ToolExecutionContext, arguments: Mapping[str, object]
    ) -> str:
        _validate_decision_context(
            context,
            (
                config.delivery.session_id
                if config.delivery is not None
                else "wake:default"
            ),
        )
        _validate_decision_argument(arguments, "reason")
        return json.dumps(
            {"recorded": True, "turn_id": context.turn_id},
            ensure_ascii=False,
            sort_keys=True,
        )

    screen_definition = _screen_definition()
    alert_definition = _alert_definition()
    share_definition, skip_definition = _decision_definitions()
    await tools.register(ctx, screen_definition, screen_handler)
    await tools.register(ctx, alert_definition, alert_handler)
    await tools.register(ctx, share_definition, share_handler)
    await tools.register(ctx, skip_definition, skip_handler)

    def setup() -> object:
        return runtime.close

    _ = await ctx.effect(setup, label="wake-runtime")
    _ = await ctx.on(EVENTMAIL_CHANGED, lambda _: runtime.content_changed())
    _ = await ctx.on(CONTEXT_PREPARED_EVENT, runtime.prepare)
    _ = await ctx.on(RUNTIME_STARTED, lambda _: runtime.start())
    _ = await ctx.on(RUNTIME_STOPPING, lambda _: runtime.close())


def _content_candidates(
    proposal: DutyProposal,
) -> tuple[Mapping[str, object], ...]:
    return proposal.candidates or ({"ref": proposal.ref, "payload": proposal.payload},)


def _candidate_payloads(proposal: DutyProposal) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for candidate in _content_candidates(proposal):
        ref = _mapping(candidate.get("ref"), "Content candidate ref")
        payload = _mapping(candidate.get("payload"), "Content candidate payload")
        candidates.append({"candidate_id": _candidate_id(ref), **dict(payload)})
    return candidates


def _hint(owner: str, proposal: DutyProposal) -> str:
    payload: Mapping[str, object] = proposal.payload
    if owner == "content" and proposal.candidates:
        payload = {"candidates": _candidate_payloads(proposal)}
    hint = "Wake duty:\n" + json.dumps(
        {"owner": owner, "payload": dict(payload)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hint + "\n\n" + _decision_prompt()


def _decision_prompt() -> str:
    return (
        "【Wake 决策合同】本轮普通回答不会发送给用户。处理 duty 后必须且只能"
        "提交一个结构化终态：值得主动告诉用户时调用 share_content，message 只写最终用户可见"
        "正文；不值得发送时调用 skip_content，reason 只写内部理由。"
    )


def _screen_definition() -> PluginToolDefinition:
    return PluginToolDefinition(
        name=_SCREEN_CONTENT,
        description="提交最多八条可能让用户感兴趣的 Content 候选和待确认问题。",
        parameters={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": _SCREEN_LIMIT,
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "initial_interest": {"type": "string"},
                            "question": {"type": "string"},
                        },
                        "required": [
                            "candidate_id",
                            "initial_interest",
                            "question",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        handler_export="screen_content",
        risk="read-only",
        search_hint="Wake Content 初筛候选",
    )


def _alert_definition() -> PluginToolDefinition:
    return PluginToolDefinition(
        name=_SHARE_ALERT,
        description="提交一条必须发送的 Wake Alert 用户消息。",
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "简洁、可行动、可直接发送给用户的告警正文",
                }
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        handler_export="share_alert",
        risk="read-write",
        search_hint="Wake Alert 告警发送",
    )


def _decision_definitions() -> tuple[PluginToolDefinition, ...]:
    return (
        PluginToolDefinition(
            name=_SHARE_CONTENT,
            description=(
                "确认当前 Wake Content 候选值得主动发送。message 是唯一用户可见正文；"
                "不要在其中写筛选过程、分数或内部判断。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "完整、自然、可直接发送给用户的主动消息",
                    },
                    "items": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 5,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                        "description": (
                            "Content duty 必填：本条消息实际采用的 1..5 个 candidate_id；"
                            "Drift duty 不填写。"
                        ),
                    },
                },
                "required": ["message", "items"],
                "additionalProperties": False,
            },
            handler_export="share_content",
            risk="read-write",
            search_hint="Wake Content 分享 主动发送",
        ),
        PluginToolDefinition(
            name=_SKIP_CONTENT,
            description="确认当前 Wake Content 候选不值得发送，并保持用户侧安静。",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "仅供内部审计的跳过理由",
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
            handler_export="skip_content",
            risk="read-write",
            search_hint="Wake Content 跳过 静默 不发送",
        ),
    )


def _validate_decision_context(
    context: ToolExecutionContext, expected_session_id: str
) -> None:
    """Confine Wake decision tools to their exact scoped Turn boundary."""

    if (
        context.origin_channel != "wake"
        or context.origin_session_key != expected_session_id
        or not context.turn_id
    ):
        raise PermissionError("Wake decision tool 只允许 configured Wake scoped Turn")


def _validate_decision_argument(arguments: Mapping[str, object], field: str) -> None:
    """Reject empty decision payloads at the plugin Tool boundary."""

    value = arguments.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串")


def _content_decision(
    view: DurableTurnView, *, allow_legacy_items: bool = False
) -> _WakeDecision | None:
    """Read one exact successful decision from the durable Turn items."""

    # 1. Collect only successful Wake-private terminal calls.
    calls: list[tuple[str, Mapping[str, object]]] = []
    for item in view.items:
        if item.kind is not TurnItemKind.TOOL_CALL:
            continue
        name = item.data.get("name")
        if name not in _DECISION_TOOLS or item.data.get("status") != "success":
            continue
        arguments = item.data.get("arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError("Wake decision tool arguments 必须是 object")
        calls.append((cast(str, name), cast(Mapping[str, object], arguments)))

    # 2. Exactly one terminal call owns the delivery decision.
    if len(calls) != 1:
        return None
    name, arguments = calls[0]
    if name == _SKIP_CONTENT:
        reason = arguments.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("skip_content reason 必须是非空字符串")
        return _WakeDecision("skip", reason=reason)
    message = arguments.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("share_content message 必须是非空字符串")
    raw_item_ids = arguments.get("items")
    if raw_item_ids is None and allow_legacy_items:
        return _WakeDecision("share", message)
    if not isinstance(raw_item_ids, (tuple, list)) or any(
        not isinstance(item_id, str) or not item_id for item_id in raw_item_ids
    ):
        raise ValueError("share_content items 必须是字符串数组")
    item_ids = tuple(cast(Sequence[str], raw_item_ids))
    if len(item_ids) > 5 or len(item_ids) != len(set(item_ids)):
        raise ValueError("share_content items 必须是不重复的 0..5 个候选")
    return _WakeDecision("share", message, item_ids)


def _screen_decision(view: DurableTurnView) -> tuple[_ScreenedItem, ...] | None:
    """Read one exact successful Content screening call."""

    calls = [
        item
        for item in view.items
        if item.kind is TurnItemKind.TOOL_CALL
        and item.data.get("name") == _SCREEN_CONTENT
        and item.data.get("status") == "success"
    ]
    if view.status is not TurnStatus.COMPLETED or len(calls) != 1:
        return None
    arguments = calls[0].data.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("screen_content arguments 必须是 object")
    return _parse_screen_arguments(cast(Mapping[str, object], arguments))


def _alert_decision(view: DurableTurnView) -> str | None:
    calls = [
        item
        for item in view.items
        if item.kind is TurnItemKind.TOOL_CALL
        and item.data.get("name") == _SHARE_ALERT
        and item.data.get("status") == "success"
    ]
    if view.status is not TurnStatus.COMPLETED or len(calls) != 1:
        return None
    arguments = calls[0].data.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("share_alert arguments 必须是 object")
    message = arguments.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("share_alert message 必须是非空字符串")
    return message


def _parse_screen_arguments(
    arguments: Mapping[str, object],
) -> tuple[_ScreenedItem, ...]:
    raw_items = arguments.get("items")
    if (
        not isinstance(raw_items, (tuple, list))
        or not 1 <= len(raw_items) <= _SCREEN_LIMIT
    ):
        raise ValueError(f"screen_content items 必须包含 1..{_SCREEN_LIMIT} 条")
    items: list[_ScreenedItem] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise ValueError("screen_content item 必须是 object")
        candidate_id = _string(raw.get("candidate_id"), "candidate_id")
        interest = _string(raw.get("initial_interest"), "initial_interest")
        question = _string(raw.get("question"), "question")
        items.append(_ScreenedItem(candidate_id, interest, question))
    ids = tuple(item.candidate_id for item in items)
    if len(ids) != len(set(ids)):
        raise ValueError("screen_content candidate_id 不得重复")
    return tuple(items)


def _accepted_receipt(receipt: Mapping[str, object]) -> TurnAcceptedReceipt:
    accepted = receipt.get("accepted_turn")
    if not isinstance(accepted, Mapping):
        raise RuntimeError("Wake selected receipt 缺少 accepted_turn")
    return TurnAcceptedReceipt(
        _string(accepted.get("session_id"), "session_id"),
        _string(accepted.get("turn_id"), "turn_id"),
    )


def _selected_content_refs(
    receipt: Mapping[str, object],
    item_ids: Sequence[str],
    *,
    allow_legacy_single: bool = False,
) -> tuple[Mapping[str, object], ...]:
    """Resolve the model's candidate ids only against the frozen Content batch."""

    raw_items = receipt.get("items")
    items = _sequence(raw_items, "Content selection items")
    if not item_ids and allow_legacy_single:
        if len(items) != 1:
            raise RuntimeError("legacy Content selection 必须恰好包含一个 member")
        return (_mapping(items[0].get("ref"), "Content selection ref"),)
    if not item_ids:
        raise ValueError("Content share_content 必须引用至少一个 candidate_id")
    candidates: dict[str, Mapping[str, object]] = {}
    for item in items:
        ref = _mapping(item.get("ref"), "Content selection ref")
        candidates[_candidate_id(ref)] = ref
    unknown = set(item_ids) - set(candidates)
    if unknown:
        raise ValueError(
            f"Content share_content 引用了批次外 candidate_id: {sorted(unknown)}"
        )
    return tuple(candidates[item_id] for item_id in item_ids)


def _candidate_id(ref: Mapping[str, object]) -> str:
    fields = (
        _string(ref.get("source_id"), "source_id"),
        _string(ref.get("item_id"), "item_id"),
        _string(ref.get("revision"), "revision"),
    )
    payload = "\x00".join(fields).encode("utf-8")
    return "candidate_" + hashlib.sha256(payload).hexdigest()[:16]


def _delivery_metadata(receipt: Mapping[str, object]) -> dict[str, object]:
    raw = receipt.get("message_metadata")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("Wake delivery message_metadata 必须是 Mapping")
    return dict(cast(Mapping[str, object], raw))


def _message_with_source_links(message: str, metadata: Mapping[str, object]) -> str:
    """Append selected source links so the user and later Turns retain provenance."""

    raw_refs = metadata.get("source_refs")
    if not isinstance(raw_refs, (list, tuple)):
        return message
    links: list[str] = []
    seen: set[str] = set()
    for raw_ref in cast(Sequence[object], raw_refs):
        if not isinstance(raw_ref, Mapping):
            continue
        source_ref = cast(Mapping[str, object], raw_ref)
        raw_url = source_ref.get("url")
        if not isinstance(raw_url, str):
            continue
        url = raw_url.strip()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if url in seen or url in message:
            continue
        seen.add(url)
        raw_title = source_ref.get("title")
        title = (
            " ".join(raw_title.split())
            if isinstance(raw_title, str) and raw_title.strip()
            else f"来源 {len(links) + 1}"
        )
        links.append(f"- {title}：<{url}>")
    if not links:
        return message
    return f"{message.rstrip()}\n\n来源：\n" + "\n".join(links)


def _logical_delivery_id(accepted: TurnAcceptedReceipt) -> str:
    payload = f"{accepted.session_id}\x00{accepted.turn_id}".encode("utf-8")
    return "wake:" + hashlib.sha256(payload).hexdigest()


def _run_id(session_id: str, turn_id: str) -> str:
    payload = f"{session_id}\x00{turn_id}".encode("utf-8")
    return "run_" + hashlib.sha256(payload).hexdigest()[:24]


def _attempt_id(timer_id: str, deadline: datetime, fired_at: datetime) -> str:
    payload = (
        f"{timer_id}\x00{deadline.astimezone(UTC).isoformat()}\x00"
        f"{fired_at.astimezone(UTC).isoformat()}"
    ).encode("utf-8")
    return "attempt_" + hashlib.sha256(payload).hexdigest()[:24]


def _attempt_detail(outcome: _AttemptOutcome) -> str:
    """Explain one Timer result with the same terms used by the Dashboard."""

    return {
        "no_due": "定时检查完成，没有到期信件",
        "content_insufficient": "Content 到期，但证据不足以进入 Wake Turn",
        "admission_rejected": "旧版随机 admission 未通过",
        "shared": "Wake Turn 已完成并确认送达",
        "model_skip": "Wake Turn 已完成，模型决定不发送",
        "deferred": "Wake Turn 未形成可发送结果，已延期重试",
        "cancelled_after_fire": "Timer 已触发，但 runtime 在检查职责前关闭",
        "delivery_unknown": "Wake Turn 决定发送，但送达结果未知",
        "failed": "Wake 检查失败",
    }[outcome]


def _pool_detail(
    pool: str,
    new_count: int,
    result: PoolResult,
) -> str:
    """Persist the values needed to reconstruct one fixed-score pool check."""

    return (
        f"{pool}, new={new_count}, new_mass={result.new_mass:.6f}, "
        f"pool_mass={result.pool_mass:.6f}, threshold={result.threshold:.6f}, "
        f"below_floor={result.below_floor}, "
        f"driver={result.driver_item_id or '-'}"
    )


def _view_from_result(result: object) -> DurableTurnView:
    status = getattr(result, "status", None)
    if not isinstance(status, TurnStatus):
        raise TypeError("Wake scoped Turn result 缺少 typed status")
    error = getattr(result, "error", None)
    raw_items = getattr(result, "items", None)
    if not isinstance(raw_items, list) or any(
        not isinstance(item, TurnItem) for item in raw_items
    ):
        raise TypeError("Wake scoped Turn result 缺少 typed items")
    return DurableTurnView(
        session_id=_string(getattr(result, "thread_id", None), "thread_id"),
        turn_id=_string(getattr(result, "id", None), "turn_id"),
        status=status,
        final_response=getattr(result, "final_response", None),
        error_type=getattr(error, "type", None),
        error_message=getattr(error, "message", None),
        error_retryable=getattr(error, "retryable", None),
        items=tuple(raw_items),
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


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是 Mapping")
    return cast(Mapping[str, object], value)


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


def _content_text(item: Mapping[str, object]) -> str:
    payload = _mapping(item.get("payload"), "Content item payload")
    text = "\n".join(
        part
        for part in (
            str(payload.get("title") or "").strip(),
            str(payload.get("content") or payload.get("body") or "").strip(),
        )
        if part
    )
    return text


def _preprocess_interest(payload: Mapping[str, object]) -> float:
    features = payload.get("preprocess_features")
    raw = (
        features.get("interest")
        if isinstance(features, Mapping)
        else payload.get("preprocess_score")
    )
    if not isinstance(raw, (int, float, str)):
        return 0.0
    try:
        return min(0.999, max(0.0, float(raw or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _semantic_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("semantic interest 必须是数字")
    score = float(value)
    if not 0.0 <= score <= 0.999:
        raise RuntimeError("semantic interest 必须在 0..0.999")
    return score

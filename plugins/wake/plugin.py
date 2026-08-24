from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, field_validator

from agent.control.models import TurnItem, TurnItemKind, TurnStatus
from agent.control.scoped_turn import DurableTurnView, TurnAcceptedReceipt
from agent.control.timer import TimerHandle, TimerStatus
from agent.tools.base import ToolExecutionContext
from agent.lifecycle.composition import CONTEXT_PREPARED_EVENT
from agent.lifecycle.types import BeforeTurnCtx
from agent.plugin_composition import (
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    SCOPED_TURNS,
    TIMERS,
    TOOL_CATALOG,
    DURABLE_DELIVERIES,
    Context,
    DurableDeliveryRequest,
    DurableDeliveryView,
    EmitEventKey,
    PluginScopedTurns,
    PluginDurableDeliveries,
    PluginTimers,
    PluginToolDefinition,
    ServiceKey,
    TurnExecutionScope,
    ToolGrant,
)
from plugins.content.plugin import CONTENT_DELIVERY, ContentDeliveryServices
from plugins.wake.legacy_rules import ArchivedRules
from plugins.wake.selection import DutyProposal, propose_content, propose_drift

logger = logging.getLogger(__name__)

api_version = 3
name = "wake"
version = "3.0.0"
desc = "Timer-driven Content and Drift scoped react"
author = "Akashic Core"
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


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


ConfigModel = Config


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
inject = (
    TIMERS,
    SCOPED_TURNS,
    DURABLE_DELIVERIES,
    CONTENT_WAKE,
    CONTENT_DELIVERY,
    DRIFT_WAKE,
    TOOL_CATALOG,
)

_SHARE_CONTENT = "share_content"
_SKIP_CONTENT = "skip_content"
_DECISION_TOOLS = frozenset({_SHARE_CONTENT, _SKIP_CONTENT})


@dataclass(frozen=True, slots=True)
class _ContentDecision:
    action: Literal["share", "skip"]
    message: str | None = None


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
        content_delivery: ContentDeliveryServices | None = None,
        target: DeliveryTarget | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._timers = timers
        self._turns = turns
        self._deliveries = deliveries
        self._content = content
        self._content_delivery = content_delivery
        self._drift = drift
        self._target = target
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
        await self._reconcile_deliveries()
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
                preloaded_tools=(_SHARE_CONTENT, _SKIP_CONTENT),
                tool_source="wake",
                tool_grant=ToolGrant.except_names(("message_push",)),
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
            await self._reconcile_deliveries()
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
        if view.status is TurnStatus.COMPLETED and owner == "content":
            decision = _content_decision(view)
            if decision is None:
                logger.error(
                    "Wake completed Content Turn without one valid decision "
                    "accepted=%s/%s",
                    view.session_id,
                    view.turn_id,
                )
                action = "defer"
            else:
                action = (
                    "ready_for_delivery" if decision.action == "share" else "abandoned"
                )
        elif view.status is TurnStatus.COMPLETED:
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
            deadline = (
                self._aware_now() + timedelta(minutes=5) if action == "defer" else None
            )
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

    async def _reconcile_deliveries(self) -> None:
        """Forward-complete delivery, projection, and Content settlement windows."""

        target = self._target
        if target is None:
            return
        deliveries = self._deliveries
        content_delivery = self._content_delivery
        if deliveries is None or content_delivery is None:
            raise RuntimeError("Wake delivery target 缺少 durable/Content capability")

        # 1. Resume every Core row owned by Content before creating missing rows.
        await self._resume_deliveries(deliveries)

        # 2. Create a stable logical delivery for each ready Content selection.
        for pending in content_delivery.pending(100):
            await self._deliver_pending(pending, deliveries, target)

    async def _resume_deliveries(self, deliveries: PluginDurableDeliveries) -> None:
        """Advance existing Core rows without consulting Content payload state."""

        for delivery in deliveries.recoverable():
            if delivery.target_service != CONTENT_DELIVERY.name:
                continue
            current = delivery
            if current.state in {"prepared", "delivered"}:
                current = await deliveries.resume(current.accepted_turn)
            if current.state == "projected":
                self._settle_projected(current)

    async def _deliver_pending(
        self,
        pending: Mapping[str, object],
        deliveries: PluginDurableDeliveries,
        target: DeliveryTarget,
    ) -> None:
        """Create or forward-complete one ready Content delivery."""

        accepted = _accepted_receipt(pending)
        current = deliveries.lookup(accepted)
        if current is None:
            turn = self._turns.read(accepted)
            if turn.status is not TurnStatus.COMPLETED:
                raise RuntimeError(
                    "Content ready selection 不属于 completed Turn: "
                    f"{accepted!r}/{turn.status.value}"
                )
            decision = _content_decision(turn)
            if decision is None or decision.action != "share" or not decision.message:
                raise RuntimeError("Content ready Turn 缺少 share_content 决策")
            current = await deliveries.submit(
                DurableDeliveryRequest(
                    logical_delivery_id=_logical_delivery_id(accepted),
                    accepted_turn=accepted,
                    target_service=CONTENT_DELIVERY.name,
                    channel=target.channel,
                    recipient=target.recipient,
                    projection_session_id=target.session_id,
                    body=decision.message,
                    metadata={"proactive": True},
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
            self._settle_projected(current)

    def _settle_projected(
        self,
        delivery: DurableDeliveryView,
    ) -> None:
        """Commit Content first, then close the Core settlement receipt."""

        content_delivery = self._content_delivery
        deliveries = self._deliveries
        if content_delivery is None or deliveries is None:
            raise RuntimeError("Wake delivery settlement capability 缺失")
        selected = content_delivery.lookup(
            {
                "session_id": delivery.accepted_turn.session_id,
                "turn_id": delivery.accepted_turn.turn_id,
            }
        )
        if selected is None:
            raise RuntimeError(
                "durable Content delivery 缺少 accepted Turn selection: "
                f"{delivery.accepted_turn!r}"
            )
        token = _string(selected.get("selection_token"), "selection_token")
        settled = content_delivery.settle(
            token,
            delivery.logical_delivery_id,
        )
        if settled.get("settled") is not True:
            raise RuntimeError(f"Content delivery settlement 未提交: {dict(settled)!r}")
        receipt = _string(settled.get("receipt"), "Content delivery receipt")
        _ = deliveries.confirm_settled(delivery.logical_delivery_id, receipt)

    def _earliest_deadline(self) -> datetime | None:
        now = self._aware_now()
        content = self._content.snapshot(now).get("earliest_not_before")
        drift = self._drift.snapshot(now).get("next_due")
        deadlines = [
            _datetime(value) for value in (content, drift) if value is not None
        ]
        return min(deadlines) if deadlines else None

    def _has_due(self) -> bool:
        now = self._aware_now()
        content = self._content.snapshot(now)
        if any(
            item.get("due") is True
            for item in _sequence(content.get("items"), "Content items")
        ):
            return True
        drift = self._drift.snapshot(now)
        return any(
            item.get("due") is True
            for item in _sequence(drift.get("proposals"), "Drift proposals")
        )

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

    if not isinstance(config, Config):
        raise TypeError("Wake config 必须通过 ConfigModel 校验")
    runtime = WakeRuntime(
        ctx.require(TIMERS),
        ctx.require(SCOPED_TURNS),
        ctx.require(CONTENT_WAKE),
        ctx.require(DRIFT_WAKE),
        deliveries=ctx.require(DURABLE_DELIVERIES),
        content_delivery=ctx.require(CONTENT_DELIVERY),
        target=config.delivery,
    )
    archived_rules = ArchivedRules(ctx.data_root)

    async def share_handler(
        context: ToolExecutionContext, arguments: Mapping[str, object]
    ) -> str:
        _validate_decision_context(context)
        _validate_decision_argument(arguments, "message")
        return json.dumps(
            {"recorded": True, "turn_id": context.turn_id},
            ensure_ascii=False,
            sort_keys=True,
        )

    async def skip_handler(
        context: ToolExecutionContext, arguments: Mapping[str, object]
    ) -> str:
        _validate_decision_context(context)
        _validate_decision_argument(arguments, "reason")
        return json.dumps(
            {"recorded": True, "turn_id": context.turn_id},
            ensure_ascii=False,
            sort_keys=True,
        )

    tools = ctx.require(TOOL_CATALOG)
    share_definition, skip_definition = _decision_definitions()
    await tools.register(ctx, share_definition, share_handler)
    await tools.register(ctx, skip_definition, skip_handler)

    def setup() -> object:
        return runtime.close

    _ = await ctx.effect(setup, label="wake-runtime")
    _ = await ctx.on(CONTENT_CHANGED, lambda _: runtime.content_changed())
    _ = await ctx.on(CONTEXT_PREPARED_EVENT, archived_rules.prepare)
    _ = await ctx.on(CONTEXT_PREPARED_EVENT, runtime.prepare)
    _ = await ctx.on(RUNTIME_STARTED, lambda _: runtime.start())
    _ = await ctx.on(RUNTIME_STOPPING, lambda _: runtime.close())


def _hint(owner: str, proposal: DutyProposal) -> str:
    hint = "Wake duty:\n" + json.dumps(
        {"owner": owner, "payload": dict(proposal.payload)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hint + ("\n\n" + _decision_prompt() if owner == "content" else "")


def _decision_prompt() -> str:
    return (
        "【Wake Content 决策合同】本轮普通回答不会发送给用户。处理 Content duty 后必须且只能"
        "提交一个结构化终态：值得主动告诉用户时调用 share_content，message 只写最终用户可见"
        "正文；不值得发送时调用 skip_content，reason 只写内部理由。"
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
                    }
                },
                "required": ["message"],
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


def _validate_decision_context(context: ToolExecutionContext) -> None:
    """Confine Wake decision tools to their exact scoped Turn boundary."""

    if (
        context.origin_channel != "wake"
        or context.origin_session_key != "wake:default"
        or not context.turn_id
    ):
        raise PermissionError("Wake decision tool 只允许 wake:default scoped Turn")


def _validate_decision_argument(arguments: Mapping[str, object], field: str) -> None:
    """Reject empty decision payloads at the plugin Tool boundary."""

    value = arguments.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串")


def _content_decision(view: DurableTurnView) -> _ContentDecision | None:
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
        return _ContentDecision("skip")
    message = arguments.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("share_content message 必须是非空字符串")
    return _ContentDecision("share", message)


def _accepted_receipt(receipt: Mapping[str, object]) -> TurnAcceptedReceipt:
    accepted = receipt.get("accepted_turn")
    if not isinstance(accepted, Mapping):
        raise RuntimeError("Wake selected receipt 缺少 accepted_turn")
    return TurnAcceptedReceipt(
        _string(accepted.get("session_id"), "session_id"),
        _string(accepted.get("turn_id"), "turn_id"),
    )


def _logical_delivery_id(accepted: TurnAcceptedReceipt) -> str:
    payload = f"{accepted.session_id}\x00{accepted.turn_id}".encode("utf-8")
    return "wake:" + hashlib.sha256(payload).hexdigest()


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

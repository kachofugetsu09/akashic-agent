from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.control.models import TurnError, TurnItem, TurnItemKind, TurnStatus
from agent.control.scoped_turn import DurableTurnView, TurnAcceptedReceipt
from agent.control.timer import TimerReceipt, TimerStatus
from agent.lifecycle.types import BeforeTurnCtx
from agent.turn_effects import PostCommitEffect, TurnStorage
from agent.plugin_composition import PluginScopedTurns, PluginTimers
from agent.plugin_composition.durable_deliveries import PluginDurableDeliveries
from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
from plugins.wake.plugin import EVENTMAIL_ALERT_DELIVERY
from plugins.wake.plugin import (
    ContentWakeServices,
    DeliveryTarget,
    DriftWakeServices,
    WakeRuntime,
    _ScreenedItem,
    _candidate_id,
)
from plugins.wake.state import WakeState

_CONTENT_REF = {
    "source_id": "fixture",
    "item_id": "item:1",
    "revision": "1",
    "state_version": 1,
}
_CONTENT_CANDIDATE = _candidate_id(_CONTENT_REF)


def _content_receipt() -> dict[str, object]:
    return {
        "selection_token": "content:selection",
        "items": ({"ref": dict(_CONTENT_REF), "payload": {}},),
    }


def _decision_item(name: str, arguments: dict[str, object]) -> TurnItem:
    return TurnItem(
        TurnItemKind.TOOL_CALL,
        f"item:{name}",
        {
            "callId": f"call:{name}",
            "name": name,
            "status": "success",
            "arguments": arguments,
            "resultPreview": '{"recorded":true}',
        },
    )


class _TimerHandle:
    def __init__(self, deadline: datetime) -> None:
        self.deadline = deadline
        self.future: asyncio.Future[TimerReceipt] = (
            asyncio.get_running_loop().create_future()
        )

    @property
    def id(self) -> str:
        return "timer:wake"

    async def result(self) -> TimerReceipt:
        return await asyncio.shield(self.future)

    async def cancel(self) -> TimerReceipt:
        if not self.future.done():
            self.future.set_result(self._receipt(TimerStatus.CANCELLED))
        return await self.future

    async def cleanup(self) -> None:
        _ = await self.cancel()

    def fire(self) -> None:
        self.future.set_result(self._receipt(TimerStatus.FIRED))

    def _receipt(self, status: TimerStatus) -> TimerReceipt:
        return TimerReceipt(self.id, self.deadline, self.deadline, status)


class _Timers:
    def __init__(self) -> None:
        self.handles: list[_TimerHandle] = []

    def schedule(self, deadline: datetime) -> _TimerHandle:
        handle = _TimerHandle(deadline)
        self.handles.append(handle)
        return handle


class _TurnHandle:
    def __init__(
        self,
        status: TurnStatus = TurnStatus.COMPLETED,
        *,
        turn_id: str = "turn:1",
        items: list[TurnItem] | None = None,
    ) -> None:
        self.accepted = TurnAcceptedReceipt("wake:default", turn_id)
        self._result = SimpleNamespace(
            id=turn_id,
            thread_id="wake:default",
            status=status,
            final_response="hello" if status is TurnStatus.COMPLETED else None,
            error=None,
            items=items
            or [
                _decision_item(
                    "share_content",
                    {"message": "hello", "items": [_CONTENT_CANDIDATE]},
                )
            ],
        )

    async def result(self):
        return self._result

    async def cleanup(self) -> None:
        return None


class _Turns:
    def __init__(self) -> None:
        self.starts: list[dict[str, object]] = []
        self.reads: dict[TurnAcceptedReceipt, DurableTurnView] = {}
        self.started = asyncio.Event()

    async def ensure_session(self, key: str, *, metadata) -> str:
        assert metadata == {"programmatic": True, "wake": True}
        return key

    async def start(self, session_id: str, content: str, **kwargs):
        self.starts.append({"session_id": session_id, "content": content, **kwargs})
        self.started.set()
        scope = kwargs["scope"]
        turn_id = f"turn:{len(self.starts)}"
        if scope.terminal_tools == ("screen_content",):
            return _TurnHandle(
                turn_id=turn_id,
                items=[
                    _decision_item(
                        "screen_content",
                        {
                            "items": [
                                {
                                    "candidate_id": _CONTENT_CANDIDATE,
                                    "initial_interest": "likely_interesting",
                                    "question": "值得进一步确认吗？",
                                }
                            ]
                        },
                    )
                ],
            )
        return _TurnHandle(turn_id=turn_id)

    def read(self, accepted: TurnAcceptedReceipt) -> DurableTurnView:
        if accepted not in self.reads:
            raise KeyError(accepted)
        return self.reads[accepted]


class _BlockingTurnHandle:
    def __init__(self, release: asyncio.Event) -> None:
        self.accepted = TurnAcceptedReceipt("wake:default", "turn:blocking")
        self._release = release

    async def result(self):
        await self._release.wait()
        return SimpleNamespace(
            id="turn:blocking",
            thread_id="wake:default",
            status=TurnStatus.FAILED,
            final_response=None,
            error=None,
            items=[],
        )

    async def cleanup(self) -> None:
        return None


class _BlockingTurns(_Turns):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def start(self, session_id: str, content: str, **kwargs):
        self.starts.append({"session_id": session_id, "content": content, **kwargs})
        self.started.set()
        return _BlockingTurnHandle(self.release)


class _Content:
    def __init__(self, now: datetime, payload: dict[str, object] | None = None) -> None:
        self.now = now
        self.payload = payload
        self.cas_wins = True
        self.snapshots = 0
        self.selects = 0
        self.expired_refs: set[tuple[str, str, str]] = set()
        self.transitions: list[tuple[str, str, datetime | None]] = []
        self.selected_rows: list[dict[str, object]] = []
        self.alerts: list[dict[str, object]] = []
        self.closed_alerts: dict[tuple[str, str], str] = {}
        self.contexts: list[dict[str, object]] = []

    def snapshot(self, now: datetime):
        self.snapshots += 1
        items = ()
        if self.payload is not None:
            item = {
                "ref": dict(_CONTENT_REF),
                "payload": self.payload,
                "snapshot_seq": 1,
                "status": "pending",
                "observed_at": self.now.isoformat(),
                "not_before": self.now.isoformat(),
                "due": now >= self.now,
            }
            ref = cast(dict[str, object], item["ref"])
            identity = (
                str(ref["source_id"]),
                str(ref["item_id"]),
                str(ref["revision"]),
            )
            if identity not in self.expired_refs:
                items = (item,)
        return {
            "snapshot_seq": 1,
            "earliest_not_before": self.now.isoformat() if items else None,
            "items": items,
        }

    def expire(self, item_refs, now):
        expired = []
        for item_ref in item_refs:
            identity = (
                str(item_ref["source_id"]),
                str(item_ref["item_id"]),
                str(item_ref["revision"]),
            )
            if identity in self.expired_refs:
                continue
            self.expired_refs.add(identity)
            expired.append(dict(item_ref))
        return {"expired": tuple(expired), "stale": ()}

    def select(self, item_ref, snapshot_seq, accepted_turn, now):
        return self.select_batch((item_ref,), snapshot_seq, accepted_turn, now)

    def select_batch(self, item_refs, snapshot_seq, accepted_turn, now):
        self.selects += 1
        if not self.cas_wins:
            return {"selected": False, "selection_token": None}
        row = {
            "selection_token": "content:selection",
            "status": "selected",
            "accepted_turn": dict(accepted_turn),
            "payload": self.payload or {},
            "items": tuple(
                {"ref": dict(item_ref), "payload": self.payload or {}}
                for item_ref in item_refs
            ),
        }
        self.selected_rows = [row]
        return {"selected": True, "selection_token": "content:selection"}

    def selection(self, accepted_turn):
        return next(
            (
                row
                for row in self.selected_rows
                if row["accepted_turn"] == dict(accepted_turn)
            ),
            None,
        )

    def selected(self, limit: int = 100):
        return tuple(self.selected_rows[:limit])

    def transition(self, token, action, *, not_before=None, selected_refs=None):
        self.transitions.append((token, action, not_before))
        self.selected_rows = [
            row for row in self.selected_rows if row["selection_token"] != token
        ]
        return {"changed": True, "status": action}

    def mail_watermark(self):
        return 1 if self.payload is not None else 0

    def report_alert(
        self, *, source_id, event_id, payload, observed_at, expires_at=None
    ):
        self.alerts = [
            {
                "source_id": source_id,
                "event_id": event_id,
                "payload": dict(payload),
                "observed_at": observed_at.isoformat(),
                "not_before": observed_at.isoformat(),
                "expires_at": None if expires_at is None else expires_at.isoformat(),
                "accepted_turn": None,
            }
        ]

    def report_context(
        self, *, source_id, event_id, payload, observed_at, expires_at=None
    ):
        self.contexts = [
            {
                "source_id": source_id,
                "event_id": event_id,
                "payload": dict(payload),
                "observed_at": observed_at.isoformat(),
                "expires_at": None if expires_at is None else expires_at.isoformat(),
            }
        ]

    def alert_deadline(self, now):
        for alert in tuple(self.alerts):
            expires_at = alert["expires_at"]
            if (
                expires_at is not None
                and datetime.fromisoformat(str(expires_at)) <= now
            ):
                self.expire_alert(alert["source_id"], alert["event_id"], now)
        due = [
            datetime.fromisoformat(str(alert["not_before"]))
            for alert in self.alerts
            if alert["accepted_turn"] is None
            and (
                alert["expires_at"] is None
                or datetime.fromisoformat(str(alert["expires_at"])) > now
            )
        ]
        return min(due) if due else None

    def select_alert(self, accepted_turn, now):
        if self.alert_deadline(now) is None:
            return None
        self.alerts[0]["accepted_turn"] = dict(accepted_turn)
        return dict(self.alerts[0])

    def selected_alert(self, accepted_turn):
        return next(
            (
                dict(alert)
                for alert in self.alerts
                if alert["accepted_turn"] == dict(accepted_turn)
            ),
            None,
        )

    def selected_alerts(self):
        return tuple(dict(alert) for alert in self.alerts if alert["accepted_turn"])

    def expire_alert(self, source_id, event_id, now):
        before = len(self.alerts)
        self.alerts = [
            alert
            for alert in self.alerts
            if not (
                alert["source_id"] == source_id
                and alert["event_id"] == event_id
                and alert["expires_at"] is not None
                and datetime.fromisoformat(str(alert["expires_at"])) <= now
            )
        ]
        changed = len(self.alerts) != before
        if changed:
            self.closed_alerts[(source_id, event_id)] = "expired"
        return changed

    def defer_alert(self, source_id, event_id, not_before):
        self.alerts[0]["not_before"] = not_before.isoformat()
        self.alerts[0]["accepted_turn"] = None

    def close_alert(self, source_id, event_id, status):
        self.closed_alerts[(source_id, event_id)] = status
        self.alerts = [
            alert
            for alert in self.alerts
            if not (alert["source_id"] == source_id and alert["event_id"] == event_id)
        ]

    def alert_status(self, source_id, event_id):
        if (source_id, event_id) in self.closed_alerts:
            return self.closed_alerts[(source_id, event_id)]
        return next(
            (
                "selected" if alert["accepted_turn"] else "pending"
                for alert in self.alerts
                if alert["source_id"] == source_id and alert["event_id"] == event_id
            ),
            None,
        )

    def active_context(self, now):
        return tuple(
            context
            for context in self.contexts
            if context["expires_at"] is None
            or datetime.fromisoformat(str(context["expires_at"])) > now
        )


class _BatchContent(_Content):
    count = 20

    def snapshot(self, now: datetime):
        self.snapshots += 1
        items = tuple(
            {
                "ref": {
                    "source_id": "fixture",
                    "item_id": f"item:{index}",
                    "revision": "1",
                    "state_version": 1,
                },
                "payload": {
                    "title": f"Title {index}",
                    "preprocess_score": 1 - index / 100,
                    "published_at": self.now.isoformat(),
                },
                "snapshot_seq": index + 1,
                "status": "pending",
                "observed_at": self.now.isoformat(),
                "not_before": self.now.isoformat(),
                "due": now >= self.now,
            }
            for index in range(self.count)
            if ("fixture", f"item:{index}", "1") not in self.expired_refs
        )
        return {
            "snapshot_seq": self.count,
            "earliest_not_before": self.now.isoformat(),
            "items": items,
        }


class _DeferredContent(_Content):
    def snapshot(self, now: datetime):
        snapshot = super().snapshot(now)
        snapshot["items"] = tuple(
            {**dict(item), "status": "deferred"}
            for item in cast(tuple[dict[str, object], ...], snapshot["items"])
        )
        return snapshot


class _Drift:
    def __init__(self, now: datetime, payload: dict[str, object] | None = None) -> None:
        self.now = now
        self.payload = payload
        self.cas_wins = True
        self.snapshots = 0
        self.selects = 0
        self.transitions: list[tuple[str, str]] = []
        self.selected_rows: list[dict[str, object]] = []

    def snapshot(self, now: datetime):
        self.snapshots += 1
        proposals = ()
        if self.payload is not None:
            proposals = (
                {
                    "ref": {
                        "proposal_id": "reflection",
                        "revision": "1",
                        "state_version": 1,
                    },
                    "payload": self.payload,
                    "due": now >= self.now,
                    "next_due": (self.now + timedelta(minutes=5)).isoformat(),
                },
            )
        return {
            "next_due": self.now.isoformat() if proposals else None,
            "proposals": proposals,
        }

    def select(self, ref, accepted_turn, now):
        self.selects += 1
        if not self.cas_wins:
            return {"selected": False, "selection_token": None}
        row = {
            "selection_token": "drift:selection",
            "status": "selected",
            "accepted_turn": dict(accepted_turn),
            "payload": self.payload or {},
        }
        self.selected_rows = [row]
        return {"selected": True, "selection_token": "drift:selection"}

    def transition(self, token, action):
        self.transitions.append((token, action))
        self.selected_rows = [
            row for row in self.selected_rows if row["selection_token"] != token
        ]
        return {"changed": True, "status": action}

    def selected(self, limit: int = 100):
        return tuple(self.selected_rows[:limit])

    def selection(self, accepted_turn):
        return next(
            (
                row
                for row in self.selected_rows
                if row["accepted_turn"] == dict(accepted_turn)
            ),
            None,
        )


def _runtime(
    now: datetime,
    content: _Content,
    drift: _Drift,
    *,
    state: WakeState | None = None,
):
    timers = _Timers()
    turns = _Turns()
    runtime = WakeRuntime(
        cast(PluginTimers, timers),
        cast(PluginScopedTurns, turns),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, drift),
        state=state,
        now=lambda: now,
    )
    return runtime, timers, turns


def _ctx(now: datetime, *, channel: str = "wake") -> BeforeTurnCtx:
    return BeforeTurnCtx(
        session_key="wake:default",
        channel=channel,
        chat_id="wake:default",
        content="check",
        timestamp=now,
        history_messages=(),
        turn_id="turn:1",
    )


@pytest.mark.asyncio
async def test_content_wins_without_reading_or_writing_drift() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now, {"kind": "fitbit", "preprocess_score": 0.9})
    drift = _Drift(now, {"prompt": "reflect"})
    runtime, _timers, _turns = _runtime(now, content, drift)
    ctx = _ctx(now)

    await runtime.prepare(ctx)

    assert ctx.abort is False
    assert "【Wake Content 初筛】" in ctx.extra_hints[0]
    assert '"kind":"fitbit"' in ctx.extra_hints[0]
    assert content.selects == 0
    assert drift.snapshots == drift.selects == 0


@pytest.mark.asyncio
async def test_content_screen_receives_one_frozen_twenty_candidate_page() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _BatchContent(now, {"kind": "fixture"})
    runtime, _timers, _turns = _runtime(now, content, _Drift(now))
    ctx = _ctx(now)

    await runtime.prepare(ctx)

    assert content.selects == 0
    candidates = json.loads(ctx.extra_hints[0].split("候选：\n", 1)[1])
    assert len(candidates) == 20
    assert all(
        candidate["candidate_id"].startswith("candidate_") for candidate in candidates
    )


@pytest.mark.asyncio
async def test_successful_selection_consumes_kick_from_full_snapshot(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)

    class LargeBatchContent(_BatchContent):
        count = 101

    content = LargeBatchContent(now, {"kind": "fixture"})
    state = WakeState(tmp_path / "wake.sqlite3")
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=state,
        now=lambda: now,
    )

    assert await runtime._admit_owner() == "content"
    runtime._active_owner = "content"
    await runtime.prepare(_ctx(now))
    proposal = runtime._content_proposal
    assert proposal is not None
    first_ref = cast(dict[str, object], proposal[1].candidates[0]["ref"])
    runtime._screened_content = (
        _ScreenedItem(_candidate_id(first_ref), "likely", "Confirm?"),
    )
    runtime._phase = "content_investigate"
    second = _ctx(now)
    second.turn_id = "turn:2"
    await runtime.prepare(second)

    assert state.has_unseen_due(content.snapshot(now)["items"], now) is False


@pytest.mark.asyncio
async def test_context_events_enter_only_content_investigation(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    content = _Content(
        now,
        {
            "title": "Model update",
            "preprocess_score": 0.9,
            "published_at": now.isoformat(),
        },
    )
    content.report_context(
        source_id="steam",
        event_id="current",
        payload={"presence": "in_game"},
        observed_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=state,
        now=lambda: now,
        proactive_context="Do not interrupt sleep.",
    )
    assert await runtime._admit_owner() == "content"
    runtime._active_owner = "content"
    first = _ctx(now)
    await runtime.prepare(first)

    assert "PROACTIVE_CONTEXT.md" in first.extra_hints[0]
    assert "ContextEvent" not in first.extra_hints[0]

    runtime._screened_content = (
        _ScreenedItem(_CONTENT_CANDIDATE, "likely", "Is this substantial?"),
    )
    runtime._phase = "content_investigate"
    second = _ctx(now)
    second.turn_id = "turn:2"
    await runtime.prepare(second)

    prompt = second.extra_hints[0]
    assert content.selects == 1
    assert "你总共有 20 轮调查预算" in prompt
    assert '"presence":"in_game"' in prompt
    assert "Is this substantial?" in prompt


@pytest.mark.asyncio
async def test_alert_bypasses_interest_and_receives_context(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    content = _Content(now, {"preprocess_score": 0.001})
    content.report_alert(
        source_id="calendar",
        event_id="meeting:1",
        payload={"title": "Meeting in ten minutes"},
        observed_at=now,
    )
    content.report_context(
        source_id="steam",
        event_id="current",
        payload={"presence": "in_game"},
        observed_at=now,
        expires_at=now + timedelta(minutes=30),
    )
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=state,
        now=lambda: now,
        proactive_context="Do not interrupt sleep.",
    )

    assert await runtime._admit_owner() == "alert"
    assert content.snapshots == 1
    runtime._phase = "alert"
    ctx = _ctx(now)
    await runtime.prepare(ctx)

    assert "【Wake Alert】" in ctx.extra_hints[0]
    assert "Do not interrupt sleep." in ctx.extra_hints[0]
    assert '"presence":"in_game"' in ctx.extra_hints[0]
    view = DurableTurnView(
        "wake:default",
        "turn:1",
        TurnStatus.FAILED,
        None,
        "fixture",
        "invalid alert",
        False,
        (),
    )
    await runtime._settle_alert(TurnAcceptedReceipt("wake:default", "turn:1"), view)
    assert content.alert_status("calendar", "meeting:1") == "skipped"
    assert state.list_runs()[0]["decision"] == "skip"


@pytest.mark.asyncio
async def test_due_alert_does_not_block_content_pool_expiry(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now - timedelta(hours=25), {"preprocess_score": 0.001})
    content.report_alert(
        source_id="calendar",
        event_id="meeting:pool-maintenance",
        payload={"title": "Meeting now"},
        observed_at=now,
    )
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=WakeState(tmp_path / "wake.sqlite3"),
        now=lambda: now,
    )

    admission = await runtime._admit_attempt()

    assert admission.turn_owner == "alert"
    assert content.expired_refs == {("fixture", "item:1", "1")}
    assert "active=0" in admission.detail
    assert "expired=1" in admission.detail
    assert "threshold=1.000000" in admission.detail


@pytest.mark.asyncio
async def test_pool_maintenance_records_while_scoped_turn_is_running(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now, None)
    content.report_alert(
        source_id="calendar",
        event_id="meeting:long-turn",
        payload={"title": "Long running alert"},
        observed_at=now,
    )
    timers = _Timers()
    turns = _BlockingTurns()
    state = WakeState(tmp_path / "wake.sqlite3")
    runtime = WakeRuntime(
        cast(PluginTimers, timers),
        cast(PluginScopedTurns, turns),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=state,
        now=lambda: now,
    )
    await runtime.start()
    await asyncio.sleep(0)

    assert len(timers.handles) == 2
    duty_handle = min(timers.handles, key=lambda handle: handle.deadline)
    maintenance_handle = max(timers.handles, key=lambda handle: handle.deadline)
    duty_handle.fire()
    await asyncio.wait_for(turns.started.wait(), timeout=1)
    assert turns.release.is_set() is False

    maintenance_handle.fire()
    for _ in range(20):
        await asyncio.sleep(0)
        terminal = [
            attempt
            for attempt in state.list_attempts()
            if attempt["outcome"] != "checking"
        ]
        if terminal:
            break

    assert len(turns.starts) == 1
    assert terminal[0]["outcome"] == "no_due"
    assert "maintenance_only=1" in str(terminal[0]["detail"])
    assert "threshold=1.000000" in str(terminal[0]["detail"])
    turns.release.set()
    for _ in range(20):
        await asyncio.sleep(0)
        if not any(
            attempt["outcome"] == "checking" for attempt in state.list_attempts()
        ):
            break
    await runtime.close()


@pytest.mark.asyncio
async def test_expired_alert_is_not_admitted_after_restart(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    path = tmp_path / "wake.sqlite3"
    state = WakeState(path)
    content = _Content(now)
    content.report_alert(
        source_id="calendar",
        event_id="old-meeting",
        payload={"title": "Old meeting"},
        observed_at=now - timedelta(hours=1),
        expires_at=now - timedelta(minutes=1),
    )
    recovered = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=WakeState(path),
        now=lambda: now,
    )

    assert await recovered._admit_owner() is None
    assert content.alert_status("calendar", "old-meeting") == "expired"


@pytest.mark.asyncio
async def test_content_declines_then_drift_wins() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now, {"wake_action": "decline"})
    drift = _Drift(now, {"prompt": "reflect"})
    runtime, _timers, _turns = _runtime(now, content, drift)
    ctx = _ctx(now)

    await runtime.prepare(ctx)

    assert content.transitions == [("content:selection", "await_change", None)]
    assert drift.selects == 1
    assert '"owner":"drift"' in ctx.extra_hints[0]


@pytest.mark.asyncio
async def test_both_decline_commit_transitions_then_quiet_abort() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now, {"wake_action": "decline"})
    drift = _Drift(now, {"wake_action": "decline"})
    runtime, _timers, _turns = _runtime(now, content, drift)
    ctx = _ctx(now)

    await runtime.prepare(ctx)

    assert ctx.abort is True and ctx.abort_reply == ""
    assert content.transitions[0][1] == "await_change"
    assert drift.transitions == [("drift:selection", "defer")]


@pytest.mark.asyncio
async def test_content_cas_lost_is_quiet_and_never_falls_through_to_drift() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now, {"kind": "calendar"})
    content.cas_wins = False
    drift = _Drift(now, {"prompt": "reflect"})
    runtime, _timers, _turns = _runtime(now, content, drift)
    ctx = _ctx(now)

    await runtime.prepare(ctx)
    runtime._screened_content = (
        _ScreenedItem(_CONTENT_CANDIDATE, "likely", "confirm"),
    )
    runtime._phase = "content_investigate"
    second = _ctx(now)
    second.turn_id = "turn:2"
    await runtime.prepare(second)

    assert second.abort is True and drift.snapshots == drift.selects == 0


@pytest.mark.asyncio
async def test_non_wake_channel_has_zero_domain_reads_or_writes() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now, {"kind": "feed"})
    drift = _Drift(now, {"prompt": "reflect"})
    runtime, _timers, _turns = _runtime(now, content, drift)

    await runtime.prepare(_ctx(now, channel="scheduler"))

    assert content.snapshots == content.selects == 0
    assert drift.snapshots == drift.selects == 0


@pytest.mark.asyncio
async def test_timer_no_due_rechecks_without_starting_turn(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    future = now + timedelta(hours=1)
    content = _Content(future, {"kind": "future"})
    drift = _Drift(future, None)
    state = WakeState(tmp_path / "wake.sqlite3")
    runtime = WakeRuntime(
        cast(PluginTimers, timers := _Timers()),
        cast(PluginScopedTurns, turns := _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, drift),
        state=state,
        now=lambda: now,
    )
    await runtime.start()
    await asyncio.sleep(0)

    assert len(timers.handles) == 2
    timers.handles[0].fire()
    for _ in range(10):
        await asyncio.sleep(0)
        attempts = state.list_attempts()
        if attempts and attempts[0]["outcome"] == "no_due":
            break
    assert turns.starts == []
    attempts = state.list_attempts()
    assert len(attempts) == 1
    assert attempts[0]["outcome"] == "no_due"
    assert attempts[0]["owner"] is None
    assert "new_mass=0.000000" in str(attempts[0]["detail"])
    assert "pool_mass=0.000000" in str(attempts[0]["detail"])
    assert "threshold=1.000000" in str(attempts[0]["detail"])
    await runtime.close()


@pytest.mark.asyncio
async def test_start_rejects_runtime_without_durable_state() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, _Content(now)),
        cast(DriftWakeServices, _Drift(now)),
        now=lambda: now,
    )

    with pytest.raises(RuntimeError, match="缺少 durable state"):
        await runtime.start()


@pytest.mark.asyncio
async def test_mail_watermark_fault_still_records_failed_timer_attempt(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    future = now + timedelta(hours=1)

    class BrokenWatermarkContent(_Content):
        def mail_watermark(self):
            raise RuntimeError("watermark unavailable")

    state = WakeState(tmp_path / "wake.sqlite3")
    runtime = WakeRuntime(
        cast(PluginTimers, timers := _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, BrokenWatermarkContent(future, {"kind": "future"})),
        cast(DriftWakeServices, _Drift(future, None)),
        state=state,
        now=lambda: now,
    )
    await runtime.start()
    await asyncio.sleep(0)

    timers.handles[0].fire()
    for _ in range(10):
        await asyncio.sleep(0)
        attempts = state.list_attempts()
        if attempts and attempts[0]["outcome"] == "failed":
            break

    attempts = state.list_attempts()
    assert len(attempts) == 1
    assert attempts[0]["outcome"] == "failed"
    assert attempts[0]["mail_watermark"] is None
    assert attempts[0]["detail"] == "RuntimeError: watermark unavailable"
    await runtime.close()


@pytest.mark.asyncio
async def test_maintenance_fault_records_failure_and_rearms(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)

    class OneFaultContent(_Content):
        watermark_calls = 0

        def mail_watermark(self):
            self.watermark_calls += 1
            if self.watermark_calls == 1:
                raise RuntimeError("one maintenance fault")
            return 0

    state = WakeState(tmp_path / "wake.sqlite3")
    runtime = WakeRuntime(
        cast(PluginTimers, timers := _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, OneFaultContent(now, None)),
        cast(DriftWakeServices, _Drift(now, None)),
        state=state,
        now=lambda: now,
    )
    await runtime.start()
    await asyncio.sleep(0)

    assert len(timers.handles) == 1
    first = timers.handles[0]
    first.fire()
    for _ in range(20):
        await asyncio.sleep(0)
        if len(timers.handles) == 2:
            break

    first_attempt = state.list_attempts()[0]
    assert first_attempt["outcome"] == "failed"
    assert first_attempt["detail"] == "RuntimeError: one maintenance fault"
    second = timers.handles[1]
    assert second.deadline == first.deadline + timedelta(minutes=5)

    second.fire()
    for _ in range(20):
        await asyncio.sleep(0)
        attempts = state.list_attempts()
        if len(attempts) == 2 and attempts[0]["outcome"] != "checking":
            break

    assert attempts[0]["outcome"] == "no_due"
    assert "maintenance_only=1" in str(attempts[0]["detail"])
    await runtime.close()


@pytest.mark.asyncio
async def test_deferred_content_is_maintained_without_starting_a_turn(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    turns = _BlockingTurns()
    runtime = WakeRuntime(
        cast(PluginTimers, timers := _Timers()),
        cast(PluginScopedTurns, turns),
        cast(
            ContentWakeServices,
            _DeferredContent(now, {"preprocess_score": 0.9}),
        ),
        cast(DriftWakeServices, _Drift(now, None)),
        state=state,
        now=lambda: now,
    )
    await runtime.start()
    await asyncio.sleep(0)

    assert len(timers.handles) == 1
    timers.handles[0].fire()
    for _ in range(20):
        await asyncio.sleep(0)
        attempts = state.list_attempts()
        if attempts and attempts[0]["outcome"] != "checking":
            break

    assert turns.started.is_set() is False
    assert attempts[0]["outcome"] == "content_insufficient"
    detail = str(attempts[0]["detail"])
    assert "maintenance_only=1" in detail
    assert "deferred_retry" not in detail
    assert all(
        field in detail
        for field in (
            "active=",
            "due=",
            "expired=",
            "scored=",
            "new=",
            "new_mass=",
            "pool_mass=",
            "threshold=",
            "below_floor=",
            "driver=",
        )
    )
    await runtime.close()


@pytest.mark.asyncio
async def test_fired_timer_closed_before_duty_check_records_terminal_attempt(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    future = now + timedelta(hours=1)
    state = WakeState(tmp_path / "wake.sqlite3")
    runtime = WakeRuntime(
        cast(PluginTimers, timers := _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, _Content(future, {"kind": "future"})),
        cast(DriftWakeServices, _Drift(future, None)),
        state=state,
        now=lambda: now,
    )
    await runtime.start()
    await asyncio.sleep(0)

    timers.handles[0].fire()
    await runtime.close()

    attempts = state.list_attempts()
    assert len(attempts) == 1
    assert attempts[0]["outcome"] == "cancelled_after_fire"
    assert attempts[0]["mail_watermark"] is None


@pytest.mark.asyncio
async def test_restart_closes_interrupted_timer_attempt_as_delivery_unknown(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    state.begin_attempt(
        attempt_id="attempt:crashed",
        timer_id="timer:crashed",
        scheduled_for=now,
        fired_at=now,
    )
    state.set_attempt_mail_watermark(attempt_id="attempt:crashed", mail_watermark=5)
    future = now + timedelta(hours=1)
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, _Content(future, {"kind": "future"})),
        cast(DriftWakeServices, _Drift(future, None)),
        state=WakeState(state.path),
        now=lambda: now,
    )

    await runtime.start()

    attempt = state.get_attempt("attempt:crashed")
    assert attempt is not None
    assert attempt["outcome"] == "delivery_unknown"
    assert attempt["detail"] == "进程重启前检查未闭合，外部效果未知"
    assert state.count_attempts() == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_below_threshold_content_is_recorded_for_real_timer_fire(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    runtime = WakeRuntime(
        cast(PluginTimers, timers := _Timers()),
        cast(PluginScopedTurns, turns := _Turns()),
        cast(ContentWakeServices, _Content(now, {"preprocess_score": 0.001})),
        cast(DriftWakeServices, _Drift(now)),
        state=state,
        now=lambda: now,
    )
    await runtime.start()
    await asyncio.sleep(0)

    timers.handles[0].fire()
    for _ in range(10):
        await asyncio.sleep(0)
        attempts = state.list_attempts()
        if attempts and attempts[0]["outcome"] == "content_insufficient":
            break

    assert turns.starts == []
    assert state.list_attempts()[0]["outcome"] == "content_insufficient"
    assert state.list_attempts()[0]["owner"] == "content"
    detail = str(state.list_attempts()[0]["detail"])
    assert all(
        field in detail
        for field in (
            "active=",
            "due=",
            "expired=",
            "scored=",
            "new=",
            "new_mass=",
            "pool_mass=",
            "threshold=",
            "below_floor=",
            "driver=",
        )
    )
    await runtime.close()


@pytest.mark.asyncio
async def test_below_threshold_content_stays_pending_without_repeated_check(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now, {"preprocess_score": 0.3})
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=WakeState(tmp_path / "wake.sqlite3"),
        now=lambda: now,
    )

    first = await runtime._admit_attempt()
    second = await runtime._admit_attempt()

    assert first.outcome == "content_insufficient"
    assert second.outcome == "content_insufficient"
    assert "没有新 Content" in second.detail
    assert len(content.snapshot(now)["items"]) == 1


@pytest.mark.asyncio
async def test_due_timer_starts_memory_aware_screen_turn(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(
        now,
        {
            "kind": "due",
            "preprocess_score": 0.9,
            "published_at": now.isoformat(),
        },
    )
    drift = _Drift(now, None)
    runtime, timers, turns = _runtime(
        now, content, drift, state=WakeState(tmp_path / "wake.sqlite3")
    )
    await runtime.start()
    await asyncio.sleep(0)
    timers.handles[0].fire()
    await asyncio.wait_for(turns.started.wait(), timeout=1)

    assert len(turns.starts) == 1
    start = turns.starts[0]
    assert start["channel"] == "wake"
    scope = start["scope"]
    assert scope.storage is TurnStorage.IN_MEMORY
    assert scope.post_commit_effect is PostCommitEffect.SUPPRESS
    assert scope.disabled_prompt_sections == frozenset()
    assert scope.tool_grant.allows("message_push") is False
    assert scope.tool_grant.allows("tool_search") is False
    assert scope.tool_grant.allows("screen_content") is True
    assert scope.tool_grant.allows("share_content") is False
    assert scope.terminal_tools == ("screen_content",)
    assert scope.max_iterations == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_source_report_from_worker_thread_wakes_runtime(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    runtime, _timers, _turns = _runtime(
        now,
        _Content(now),
        _Drift(now),
    )
    runtime._loop = asyncio.get_running_loop()
    content = cast(_Content, runtime._content)

    waiter = asyncio.create_task(runtime._dirty.wait())
    await asyncio.to_thread(
        content.report_alert,
        source_id="fixture",
        event_id="worker",
        payload={"title": "worker report"},
        observed_at=now,
    )
    runtime.content_changed()

    await asyncio.wait_for(waiter, timeout=1)


@pytest.mark.asyncio
async def test_context_source_can_report_before_wake_runtime_starts(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    content = _Content(
        now,
        {
            "title": "Candidate",
            "preprocess_score": 0.9,
            "published_at": now.isoformat(),
        },
    )
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=state,
        now=lambda: now,
    )
    await asyncio.to_thread(
        content.report_context,
        source_id="steam",
        event_id="current",
        payload={"presence": "in_game"},
        observed_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    runtime.content_changed()
    await runtime.start()
    assert await runtime._admit_owner() == "content"
    runtime._active_owner = "content"
    await runtime.prepare(_ctx(now))
    runtime._screened_content = (
        _ScreenedItem(_CONTENT_CANDIDATE, "likely", "Confirm?"),
    )
    runtime._phase = "content_investigate"
    second = _ctx(now)
    second.turn_id = "turn:2"
    await runtime.prepare(second)

    assert content.active_context(now)[0]["payload"] == {"presence": "in_game"}
    assert '"presence":"in_game"' in second.extra_hints[0]
    await runtime.close()


@pytest.mark.asyncio
async def test_expired_prepared_alert_is_cancelled_before_restart_send(
    tmp_path,
) -> None:
    selected_at = datetime(2026, 8, 23, 9, tzinfo=UTC)
    recovered_at = selected_at + timedelta(minutes=2)
    state = WakeState(tmp_path / "wake.sqlite3")
    content = _Content(selected_at)
    content.report_alert(
        source_id="calendar",
        event_id="meeting:expired",
        payload={"title": "Old meeting"},
        observed_at=selected_at,
        expires_at=selected_at + timedelta(minutes=1),
    )
    accepted = TurnAcceptedReceipt("wake:default", "turn:expired")
    assert (
        content.select_alert(
            {"session_id": accepted.session_id, "turn_id": accepted.turn_id},
            selected_at,
        )
        is not None
    )
    ledger = DurableDeliveryStore(tmp_path / "settlements.sqlite")
    _ = ledger.prepare(
        {
            "logical_delivery_id": "wake:alert:expired",
            "accepted_session_id": accepted.session_id,
            "accepted_turn_id": accepted.turn_id,
            "target_service": EVENTMAIL_ALERT_DELIVERY.name,
            "channel": "recording",
            "recipient": "recipient",
            "projection_session_id": "recipient-session",
            "body": "Do not send",
            "metadata": {
                "source_id": "calendar",
                "event_id": "meeting:expired",
            },
        }
    )
    sender_calls = 0

    async def sender(*_args: object) -> object:
        nonlocal sender_calls
        sender_calls += 1
        raise AssertionError("expired prepared Alert reached provider sender")

    async def projector(_request: object) -> str:
        raise AssertionError("expired prepared Alert reached Session projector")

    deliveries = PluginDurableDeliveries(
        ledger,
        cast(Any, sender),
        projector,
        recover_started=False,
    )
    turns = _Turns()
    turns.reads[accepted] = DurableTurnView(
        accepted.session_id,
        accepted.turn_id,
        TurnStatus.COMPLETED,
        "share",
        None,
        None,
        None,
        (),
    )
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, turns),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(recovered_at)),
        deliveries=deliveries,
        content_delivery=cast(Any, SimpleNamespace(pending=lambda _limit: ())),
        drift_delivery=cast(Any, SimpleNamespace(pending=lambda _limit: ())),
        target=DeliveryTarget(
            channel="recording",
            recipient="recipient",
            session_id="recipient-session",
        ),
        state=state,
        now=lambda: recovered_at,
    )

    await runtime.start()

    assert sender_calls == 0
    assert content.alert_status("calendar", "meeting:expired") == "expired"
    delivery = deliveries.lookup(accepted)
    assert delivery is not None and delivery.state == "rejected"
    await runtime.close()


@pytest.mark.asyncio
async def test_content_crash_before_selection_retries_then_commits(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(
        now,
        {
            "title": "High value",
            "preprocess_score": 0.9,
            "published_at": now.isoformat(),
        },
    )
    path = tmp_path / "wake.sqlite3"
    first_state = WakeState(path)
    first = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=first_state,
        now=lambda: now,
    )

    assert await first._admit_owner() == "content"
    assert first_state.has_unseen_due(content.snapshot(now)["items"], now) is True

    recovered_state = WakeState(path)
    recovered = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=recovered_state,
        now=lambda: now,
    )
    assert await recovered._admit_owner() == "content"
    recovered._active_owner = "content"
    await recovered.prepare(_ctx(now))
    recovered._screened_content = (
        _ScreenedItem(_CONTENT_CANDIDATE, "likely_interesting", "Confirm?"),
    )
    recovered._phase = "content_investigate"
    second = _ctx(now)
    second.turn_id = "turn:2"
    await recovered.prepare(second)

    assert recovered_state.has_unseen_due(content.snapshot(now)["items"], now) is False


@pytest.mark.asyncio
async def test_low_value_content_batch_does_not_admit_scoped_turn(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now, {"preprocess_score": 0.001})
    turns = _Turns()
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, turns),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=WakeState(tmp_path / "wake.sqlite3"),
        now=lambda: now,
    )

    assert await runtime._admit_owner() is None


@pytest.mark.asyncio
async def test_passive_semantic_interest_can_admit_low_preprocess_content(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)

    class RankedContent(_Content):
        def snapshot(self, _now):
            items = tuple(
                {
                    "ref": {
                        "source_id": "fixture",
                        "item_id": item_id,
                        "revision": "1",
                        "state_version": 1,
                    },
                    "payload": payload,
                    "snapshot_seq": index,
                    "status": "pending",
                    "observed_at": now.isoformat(),
                    "not_before": now.isoformat(),
                    "due": True,
                }
                for index, (item_id, payload) in enumerate(
                    (
                        (
                            "generic",
                            {
                                "title": "generic headline",
                                "preprocess_score": 0.2,
                                "published_at": now.isoformat(),
                            },
                        ),
                        (
                            "matched",
                            {
                                "title": "matched memory topic",
                                "preprocess_score": 0.001,
                                "published_at": now.isoformat(),
                            },
                        ),
                    ),
                    start=1,
                )
            )
            return {"snapshot_seq": 2, "items": items}

    content = RankedContent(now)

    class SemanticInterest:
        async def score(self, texts, *, cutoff):
            assert texts == ["generic headline", "matched memory topic"]
            assert cutoff == now.isoformat()
            return (0.0, 0.999)

    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=WakeState(tmp_path / "wake.sqlite3"),
        now=lambda: now,
        semantic_interest=cast(Any, SemanticInterest()),
    )

    assert await runtime._admit_owner() == "content"
    runtime._active_owner = "content"
    ctx = _ctx(now)
    await runtime.prepare(ctx)
    candidates = json.loads(ctx.extra_hints[0].split("候选：\n", 1)[1])
    assert candidates[0]["title"] == "matched memory topic"


@pytest.mark.asyncio
async def test_content_semantic_interest_is_calculated_only_once_per_revision(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(
        now,
        {
            "title": "one-time score",
            "preprocess_score": 0.1,
            "published_at": now.isoformat(),
        },
    )

    class SemanticInterest:
        calls = 0

        async def score(self, texts, *, cutoff):
            assert texts == ["one-time score"]
            assert cutoff == now.isoformat()
            self.calls += 1
            return (0.1,)

    semantic = SemanticInterest()
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=WakeState(tmp_path / "wake.sqlite3"),
        now=lambda: now,
        semantic_interest=cast(Any, semantic),
    )

    first = await runtime._admit_attempt()
    second = await runtime._admit_attempt()

    assert first.outcome == "content_insufficient"
    assert second.outcome == "content_insufficient"
    assert semantic.calls == 1


@pytest.mark.asyncio
async def test_static_confidence_is_stored_then_only_time_decay_changes_mass(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now, {"title": "undated", "preprocess_score": 0.9})
    state = WakeState(tmp_path / "wake.sqlite3")
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, _Turns()),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, _Drift(now)),
        state=state,
        now=lambda: now,
    )

    assert (await runtime._admit_attempt()).outcome == "content_insufficient"
    scored = state.scored_items(content.snapshot(now)["items"])
    payload = cast(Mapping[str, object], scored[0]["payload"])
    assert payload["_wake_initial_score"] == pytest.approx(
        -math.log1p(-0.9) * 0.03
    )
    assert state.audit_pool(scored, now=now).pool_mass == pytest.approx(
        -math.log1p(-0.9) * 0.03
    )
    assert state.audit_pool(scored, now=now + timedelta(hours=72)).pool_mass == 0.0


@pytest.mark.asyncio
async def test_targeted_wake_turn_reads_mobile_history_without_writing_memory() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    turns = _Turns()
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, turns),
        cast(ContentWakeServices, _Content(now)),
        cast(DriftWakeServices, _Drift(now)),
        target=DeliveryTarget(
            channel="mobile",
            recipient="device:one",
            session_id="mobile:conversation",
        ),
        now=lambda: now,
    )

    with pytest.raises(RuntimeError, match="缺少 durable capability"):
        await runtime._start_turn()

    start = turns.starts[0]
    scope = start["scope"]
    assert start["session_id"] == "mobile:conversation"
    assert scope.storage is TurnStorage.IN_MEMORY
    assert scope.session_history_read is True
    assert scope.disabled_prompt_sections == frozenset()
    assert scope.post_commit_effect is PostCommitEffect.SUPPRESS


@pytest.mark.asyncio
async def test_content_second_turn_removes_memory_and_keeps_evidence_tools() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    turns = _Turns()
    runtime = WakeRuntime(
        cast(PluginTimers, _Timers()),
        cast(PluginScopedTurns, turns),
        cast(ContentWakeServices, _Content(now, {"title": "Model update"})),
        cast(DriftWakeServices, _Drift(now)),
        now=lambda: now,
    )
    runtime._admitted_content = (
        1,
        tuple(_Content(now, {"title": "Model update"}).snapshot(now)["items"]),
    )
    runtime._active_owner = "content"
    await runtime.prepare(_ctx(now))

    await runtime._start_turn("content")

    assert len(turns.starts) == 2
    screen_scope = turns.starts[0]["scope"]
    evidence_scope = turns.starts[1]["scope"]
    assert screen_scope.disabled_prompt_sections == frozenset()
    assert evidence_scope.disabled_prompt_sections == frozenset(
        {"memory", "long_term_memory"}
    )
    assert evidence_scope.preloaded_tools == (
        "recall_memory",
        "web_fetch",
        "share_content",
        "skip_content",
    )


@pytest.mark.parametrize(
    ("status", "retryable", "action"),
    [
        (TurnStatus.COMPLETED, None, "ready_for_delivery"),
        (TurnStatus.FAILED, True, "defer"),
        (TurnStatus.FAILED, False, "invalidated"),
        (TurnStatus.CANCELLED, None, "defer"),
        (TurnStatus.INTERRUPTED, None, "defer"),
    ],
)
def test_terminal_matrix(status, retryable, action) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now)
    drift = _Drift(now)
    runtime, _timers, _turns = _runtime(now, content, drift)
    error = TurnError("fixture", "failed", retryable) if retryable is not None else None
    view = DurableTurnView(
        "wake:default",
        "turn:1",
        status,
        None,
        error.type if error else None,
        error.message if error else None,
        error.retryable if error else None,
        (
            (
                _decision_item(
                    "share_content",
                    {"message": "hello", "items": [_CONTENT_CANDIDATE]},
                ),
            )
            if status is TurnStatus.COMPLETED
            else ()
        ),
    )

    runtime._settle(
        "content",
        _content_receipt(),
        view,
    )

    assert content.transitions[0][1] == action


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "arguments", "expected_action", "expected_timers"),
    [
        (
            "share_content",
            {"message": "done", "items": [_CONTENT_CANDIDATE]},
            "ready_for_delivery",
            1,
        ),
        ("skip_content", {"reason": "not relevant"}, "release", 1),
        (None, {}, "defer", 1),
    ],
)
async def test_startup_reconciles_durable_typed_decision_before_arming(
    tmp_path,
    decision: str | None,
    arguments: dict[str, object],
    expected_action: str,
    expected_timers: int,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now)
    content.selected_rows = [
        {
            "selection_token": "content:selection",
            "status": "selected",
            "accepted_turn": {
                "session_id": "wake:default",
                "turn_id": "turn:old",
            },
            "items": ({"ref": dict(_CONTENT_REF), "payload": {}},),
        }
    ]
    drift = _Drift(now)
    runtime, timers, turns = _runtime(
        now, content, drift, state=WakeState(tmp_path / "wake.sqlite3")
    )
    accepted = TurnAcceptedReceipt("wake:default", "turn:old")
    turns.reads[accepted] = DurableTurnView(
        "wake:default",
        "turn:old",
        TurnStatus.COMPLETED,
        "过滤，不推送（事故诱饵）",
        None,
        None,
        None,
        (() if decision is None else (_decision_item(decision, arguments),)),
    )

    await runtime.start()
    await asyncio.sleep(0)

    assert content.transitions[0][1] == expected_action
    assert len(timers.handles) == expected_timers
    await runtime.close()


@pytest.mark.parametrize(
    ("items", "action"),
    [
        ((_decision_item("skip_content", {"reason": "not relevant"}),), "release"),
        ((), "defer"),
        (
            (
                _decision_item("share_content", {"message": "share"}),
                _decision_item("skip_content", {"reason": "conflict"}),
            ),
            "defer",
        ),
        (
            (_decision_item("share_content", {"message": "share", "items": []}),),
            "defer",
        ),
        (
            (
                _decision_item(
                    "share_content",
                    {"message": "share", "items": ["candidate_unknown"]},
                ),
            ),
            "defer",
        ),
        (
            (
                _decision_item(
                    "share_content",
                    {
                        "message": "share",
                        "items": [_CONTENT_CANDIDATE, _CONTENT_CANDIDATE],
                    },
                ),
            ),
            "defer",
        ),
    ],
)
def test_completed_content_requires_one_structured_decision(
    items: tuple[TurnItem, ...], action: str
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now)
    runtime, _timers, _turns = _runtime(now, content, _Drift(now))

    runtime._settle(
        "content",
        _content_receipt(),
        DurableTurnView(
            "wake:default",
            "turn:decision",
            TurnStatus.COMPLETED,
            "internal diagnostic text",
            None,
            None,
            None,
            items,
        ),
    )

    assert content.transitions[0][1] == action


@pytest.mark.parametrize(
    ("items", "action"),
    [
        (
            (
                _decision_item(
                    "share_content", {"message": "drift thought", "items": []}
                ),
            ),
            "ready_for_delivery",
        ),
        ((_decision_item("skip_content", {"reason": "stay quiet"}),), "await_change"),
        ((), "await_change"),
    ],
)
def test_completed_drift_uses_same_typed_delivery_decision(
    items: tuple[TurnItem, ...], action: str
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    drift = _Drift(now)
    runtime, _timers, _turns = _runtime(now, _Content(now), drift)

    runtime._settle(
        "drift",
        {"selection_token": "drift:selection", "next_due": None},
        DurableTurnView(
            "wake:default",
            "turn:drift",
            TurnStatus.COMPLETED,
            "过滤，不推送（事故诱饵）",
            None,
            None,
            None,
            items,
        ),
    )

    assert drift.transitions[0][1] == action


@pytest.mark.asyncio
async def test_startup_active_selection_fails_loud_without_timer_or_second_turn(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now)
    content.selected_rows = [
        {
            "selection_token": "content:active",
            "status": "selected",
            "accepted_turn": {
                "session_id": "wake:default",
                "turn_id": "turn:active",
            },
        }
    ]
    drift = _Drift(now)
    runtime, timers, turns = _runtime(
        now, content, drift, state=WakeState(tmp_path / "wake.sqlite3")
    )
    accepted = TurnAcceptedReceipt("wake:default", "turn:active")
    turns.reads[accepted] = DurableTurnView(
        "wake:default",
        "turn:active",
        TurnStatus.IN_PROGRESS,
        None,
        None,
        None,
        None,
    )

    with pytest.raises(RuntimeError, match="早于 Core Turn recovery/handoff"):
        await runtime.start()

    assert content.selected_rows[0]["selection_token"] == "content:active"
    assert content.transitions == []
    assert timers.handles == [] and turns.starts == []
    await runtime.close()


@pytest.mark.parametrize(
    ("next_due", "action"),
    [
        ("2026-08-23T09:05:00+00:00", "defer"),
        (None, "await_change"),
    ],
)
@pytest.mark.parametrize(
    "status",
    [TurnStatus.FAILED, TurnStatus.CANCELLED, TurnStatus.INTERRUPTED],
)
def test_drift_retry_transition_respects_proposal_owned_next_due(
    status, next_due, action
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now)
    drift = _Drift(now)
    runtime, _timers, _turns = _runtime(now, content, drift)
    view = DurableTurnView(
        "wake:default",
        "turn:drift",
        status,
        None,
        "fixture" if status is TurnStatus.FAILED else None,
        "retry" if status is TurnStatus.FAILED else None,
        True if status is TurnStatus.FAILED else None,
    )

    runtime._settle(
        "drift",
        {"selection_token": "drift:selection", "next_due": next_due},
        view,
    )

    assert drift.transitions == [("drift:selection", action)]


def test_startup_transition_rejection_fails_loud_instead_of_looping() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now)
    drift = _Drift(now)
    runtime, _timers, _turns = _runtime(now, content, drift)
    view = DurableTurnView(
        "wake:default",
        "turn:1",
        TurnStatus.COMPLETED,
        "done",
        None,
        None,
        None,
    )

    def rejected(token, action, *, not_before=None, selected_refs=None):
        return {"changed": False, "reason": "status:ready_for_delivery"}

    content.transition = rejected
    with pytest.raises(RuntimeError, match="selected transition 未提交"):
        runtime._settle(
            "content",
            {"selection_token": "content:selection"},
            view,
        )


@pytest.mark.asyncio
async def test_startup_reconciles_more_than_one_selected_page(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now)
    content.selected_rows = [
        {
            "selection_token": f"content:{index}",
            "status": "selected",
            "accepted_turn": {
                "session_id": "wake:default",
                "turn_id": f"turn:{index}",
            },
        }
        for index in range(101)
    ]
    drift = _Drift(now)
    runtime, _timers, turns = _runtime(
        now, content, drift, state=WakeState(tmp_path / "wake.sqlite3")
    )
    for index in range(101):
        accepted = TurnAcceptedReceipt("wake:default", f"turn:{index}")
        turns.reads[accepted] = DurableTurnView(
            "wake:default",
            f"turn:{index}",
            TurnStatus.COMPLETED,
            "done",
            None,
            None,
            None,
        )

    await runtime.start()

    assert content.selected_rows == []
    assert len(content.transitions) == 101
    await runtime.close()

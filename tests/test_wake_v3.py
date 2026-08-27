from __future__ import annotations

import asyncio
import json
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
from plugins.wake.plugin import (
    ContentWakeServices,
    DeliveryTarget,
    DriftWakeServices,
    WakeRuntime,
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
    def __init__(self, status: TurnStatus = TurnStatus.COMPLETED) -> None:
        self.accepted = TurnAcceptedReceipt("wake:default", "turn:1")
        self._result = SimpleNamespace(
            id="turn:1",
            thread_id="wake:default",
            status=status,
            final_response="hello" if status is TurnStatus.COMPLETED else None,
            error=None,
            items=[
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

    async def ensure_session(self, key: str, *, metadata) -> str:
        assert metadata == {"programmatic": True, "wake": True}
        return key

    async def start(self, session_id: str, content: str, **kwargs):
        self.starts.append({"session_id": session_id, "content": content, **kwargs})
        return _TurnHandle()

    def read(self, accepted: TurnAcceptedReceipt) -> DurableTurnView:
        if accepted not in self.reads:
            raise KeyError(accepted)
        return self.reads[accepted]


class _Content:
    def __init__(self, now: datetime, payload: dict[str, object] | None = None) -> None:
        self.now = now
        self.payload = payload
        self.cas_wins = True
        self.snapshots = 0
        self.selects = 0
        self.transitions: list[tuple[str, str, datetime | None]] = []
        self.selected_rows: list[dict[str, object]] = []

    def snapshot(self, now: datetime):
        self.snapshots += 1
        items = ()
        if self.payload is not None:
            items = (
                {
                    "ref": dict(_CONTENT_REF),
                    "payload": self.payload,
                    "snapshot_seq": 1,
                    "status": "pending",
                    "not_before": self.now.isoformat(),
                    "due": now >= self.now,
                },
            )
        return {
            "snapshot_seq": 1,
            "earliest_not_before": self.now.isoformat() if items else None,
            "items": items,
        }

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


class _BatchContent(_Content):
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
                },
                "snapshot_seq": index + 1,
                "status": "pending",
                "not_before": self.now.isoformat(),
                "due": now >= self.now,
            }
            for index in range(20)
        )
        return {
            "snapshot_seq": 20,
            "earliest_not_before": self.now.isoformat(),
            "items": items,
        }


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


def _runtime(now: datetime, content: _Content, drift: _Drift):
    timers = _Timers()
    turns = _Turns()
    runtime = WakeRuntime(
        cast(PluginTimers, timers),
        cast(PluginScopedTurns, turns),
        cast(ContentWakeServices, content),
        cast(DriftWakeServices, drift),
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
    assert '"owner":"content"' in ctx.extra_hints[0]
    assert content.selects == 1
    assert drift.snapshots == drift.selects == 0


@pytest.mark.asyncio
async def test_one_content_turn_receives_one_frozen_twenty_candidate_page() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _BatchContent(now, {"kind": "fixture"})
    runtime, _timers, _turns = _runtime(now, content, _Drift(now))
    ctx = _ctx(now)

    await runtime.prepare(ctx)

    assert content.selects == 1
    selected_items = content.selected_rows[0]["items"]
    assert isinstance(selected_items, tuple)
    assert len(selected_items) == 20
    duty = json.loads(ctx.extra_hints[0].split("\n", 2)[1])
    assert duty["owner"] == "content"
    assert len(duty["payload"]["candidates"]) == 20
    assert all(
        candidate["candidate_id"].startswith("candidate_")
        for candidate in duty["payload"]["candidates"]
    )


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

    assert ctx.abort is True and drift.snapshots == drift.selects == 0


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
async def test_timer_no_due_rechecks_without_starting_turn() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    future = now + timedelta(hours=1)
    content = _Content(future, {"kind": "future"})
    drift = _Drift(future, None)
    runtime, timers, turns = _runtime(now, content, drift)
    await runtime.start()
    await asyncio.sleep(0)

    assert len(timers.handles) == 1
    timers.handles[0].fire()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert turns.starts == []
    await runtime.close()


@pytest.mark.asyncio
async def test_due_timer_starts_memoryless_wake_scoped_turn() -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    content = _Content(now, {"kind": "due"})
    drift = _Drift(now, None)
    runtime, timers, turns = _runtime(now, content, drift)
    await runtime.start()
    await asyncio.sleep(0)
    timers.handles[0].fire()
    for _ in range(10):
        await asyncio.sleep(0)
        if turns.starts:
            break

    assert len(turns.starts) == 1
    start = turns.starts[0]
    assert start["channel"] == "wake"
    scope = start["scope"]
    assert scope.storage is TurnStorage.IN_MEMORY
    assert scope.post_commit_effect is PostCommitEffect.SUPPRESS
    assert scope.disabled_prompt_sections == frozenset({"memory"})
    assert scope.tool_grant.allows("message_push") is False
    assert scope.tool_grant.allows("tool_search") is False
    assert scope.tool_grant.allows("share_content") is True
    assert scope.tool_grant.allows("skip_content") is True
    assert scope.terminal_tools == ("share_content", "skip_content")
    await runtime.close()


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
        random_draw=lambda: 0.0,
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
        random_draw=lambda: 0.1,
        now=lambda: now,
        semantic_interest=cast(Any, SemanticInterest()),
    )

    assert await runtime._admit_owner() == "content"
    runtime._active_owner = "content"
    ctx = _ctx(now)
    await runtime.prepare(ctx)
    duty = json.loads(ctx.extra_hints[0].split("\n", 2)[1])
    assert duty["payload"]["candidates"][0]["title"] == "matched memory topic"


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
            0,
        ),
        ("skip_content", {"reason": "not relevant"}, "release", 0),
        (None, {}, "defer", 0),
    ],
)
async def test_startup_reconciles_durable_typed_decision_before_arming(
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
    runtime, timers, turns = _runtime(now, content, drift)
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
async def test_startup_active_selection_fails_loud_without_timer_or_second_turn() -> (
    None
):
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
    runtime, timers, turns = _runtime(now, content, drift)
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
async def test_startup_reconciles_more_than_one_selected_page() -> None:
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
    runtime, _timers, turns = _runtime(now, content, drift)
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

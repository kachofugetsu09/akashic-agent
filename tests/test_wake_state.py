import math
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.plugin_composition import DashboardContext
from plugins.wake.dashboard import register as register_dashboard
from plugins.wake.pool import build_initial_score
from plugins.wake.state import ContentScore, WakeState


def _item(sequence: int, score: float) -> dict[str, object]:
    now = "2026-08-23T09:00:00+00:00"
    return {
        "ref": {
            "source_id": "feed",
            "item_id": f"item:{sequence}",
            "revision": "1",
            "state_version": 1,
        },
        "payload": {
            "preprocess_score": score,
            "published_at": now,
        },
        "snapshot_seq": sequence,
        "status": "pending",
        "observed_at": now,
        "not_before": now,
        "due": True,
    }


def _scored(
    state: WakeState,
    items: tuple[dict[str, object], ...] | list[dict[str, object]],
    now: datetime,
) -> tuple[Mapping[str, object], ...]:
    due = state.unscored_due_items(items)
    state.record_content_scores(
        tuple(
            ContentScore(
                source_id=str(item["ref"]["source_id"]),  # type: ignore[index]
                item_id=str(item["ref"]["item_id"]),  # type: ignore[index]
                revision=str(item["ref"]["revision"]),  # type: ignore[index]
                initial_score=build_initial_score(
                    float(item["payload"]["preprocess_score"]),  # type: ignore[index]
                    has_published_at=True,
                    wake_eligible=True,
                ),
                semantic_interest=0.0,
                scored_at=now,
            )
            for item in due
        )
    )
    return state.scored_items(items)


def test_low_value_batch_advances_watermark_without_starting_turn(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    items = [_item(index, 0.001) for index in range(1, 21)]

    result = state.evaluate(_scored(state, items, now), snapshot_seq=20, now=now)

    assert result.should_wake is False
    assert state.has_unseen_due(items, now) is False
    assert state.unseen_deadline(items) is None


def test_new_content_rechecks_pool_without_a_hidden_refractory(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    first = [_item(1, 0.9)]
    first_result = state.evaluate(
        _scored(state, first, now), snapshot_seq=1, now=now
    )
    assert first_result.should_wake
    assert first_result.pool_mass == pytest.approx(-math.log1p(-0.9))
    assert state.has_unseen_due(first, now) is True
    state.commit_content_admission(first)

    second = [*first, _item(2, 0.05)]
    result = state.evaluate(_scored(state, second, now), snapshot_seq=2, now=now)

    assert result.should_wake is True
    assert state.has_unseen_due(second, now) is True


def test_future_item_remains_unseen_after_current_batch_is_evaluated(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    tomorrow = now + timedelta(days=1)
    state = WakeState(tmp_path / "wake.sqlite3")
    current = _item(1, 0.5)
    future = _item(2, 0.9)
    future["not_before"] = tomorrow.isoformat()
    future["due"] = False

    first = state.evaluate(
        _scored(state, [current, future], now), snapshot_seq=2, now=now
    )

    assert first.should_wake is False
    assert state.unseen_deadline((current, future)) == tomorrow
    future["due"] = True
    second = state.evaluate(
        _scored(state, [current, future], tomorrow),
        snapshot_seq=2,
        now=tomorrow,
    )
    assert second.should_wake is True
    assert second.driver_item_id == "item:2"


def test_pool_expiry_requires_low_mass_and_minimum_residence(tmp_path) -> None:
    now = datetime(2026, 8, 25, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    old_low = _item(1, 0.001)
    old_low["observed_at"] = (now - timedelta(hours=25)).isoformat()
    old_high = _item(2, 0.9)
    old_high["observed_at"] = (now - timedelta(hours=25)).isoformat()
    young_low = _item(3, 0.001)
    young_low["observed_at"] = (now - timedelta(hours=23)).isoformat()

    items = _scored(state, [old_low, old_high, young_low], now)
    expired = state.expired_content_refs(
        items,
        now=now,
        minimum_residence=timedelta(hours=24),
    )

    assert [ref["item_id"] for ref in expired] == ["item:1"]


def test_maintenance_deadline_uses_last_durable_fire(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")

    assert state.next_maintenance_deadline(
        now, interval=timedelta(minutes=5)
    ) == now + timedelta(minutes=5)
    state.begin_attempt(
        attempt_id="attempt:heartbeat",
        timer_id="timer:heartbeat",
        scheduled_for=now,
        fired_at=now,
    )

    assert state.next_maintenance_deadline(
        now, interval=timedelta(minutes=5)
    ) == now + timedelta(minutes=5)


def test_pool_audit_measures_seen_content_from_its_stored_score(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    item = _item(1, 0.9)
    scored = _scored(state, [item], now)
    state.commit_content_admission(scored)

    audit = state.audit_pool(scored, now=now + timedelta(hours=1))

    assert audit.should_wake is False
    assert audit.new_mass == 0.0
    assert audit.pool_mass > 0.0


@pytest.mark.parametrize(
    ("source_id", "revision"),
    (("other-feed", "1"), ("feed", "2")),
)
def test_new_mass_uses_full_content_identity(
    tmp_path, source_id: str, revision: str
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    old_high = _item(1, 0.7)
    old_ref = old_high["ref"]
    assert isinstance(old_ref, dict)
    old_ref["item_id"] = "shared-id"
    old_support = _item(3, 0.7)
    old_support["ref"]["source_id"] = "support-feed"  # type: ignore[index]
    old_scored = _scored(state, [old_high, old_support], now)
    state.commit_content_admission(old_scored)

    new_low = _item(2, 0.001)
    new_ref = new_low["ref"]
    assert isinstance(new_ref, dict)
    new_ref.update(
        {"source_id": source_id, "item_id": "shared-id", "revision": revision}
    )
    items = _scored(
        state, [old_high, old_support, new_low], now + timedelta(hours=12)
    )
    result = state.evaluate(
        items,
        snapshot_seq=2,
        now=now + timedelta(hours=12),
    )

    assert result.should_wake is True
    assert result.new_mass == 0.0
    assert state.has_unseen_due(items, now) is True


def test_seen_content_stays_in_pool_and_amplifies_next_new_kick(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    old_high = _item(1, 0.999)
    new_low = _item(2, 0.05)
    old_high["ref"]["source_id"] = "old-feed"  # type: ignore[index]
    pooled = WakeState(tmp_path / "pooled.sqlite3")
    old_scored = _scored(pooled, [old_high], now)
    pooled.commit_content_admission(old_scored)

    pooled_items = _scored(pooled, [old_high, new_low], now + timedelta(hours=1))
    pooled_result = pooled.evaluate(
        pooled_items,
        snapshot_seq=2,
        now=now + timedelta(hours=1),
    )
    isolated_old = _item(1, 0.001)
    isolated_old["ref"]["source_id"] = "old-feed"  # type: ignore[index]
    isolated = WakeState(tmp_path / "isolated.sqlite3")
    isolated_scored = _scored(isolated, [isolated_old], now)
    isolated.commit_content_admission(isolated_scored)
    isolated_items = _scored(
        isolated, [isolated_old, new_low], now + timedelta(hours=1)
    )
    isolated_result = isolated.evaluate(
        isolated_items,
        snapshot_seq=2,
        now=now + timedelta(hours=1),
    )

    assert pooled_result.new_mass == isolated_result.new_mass
    assert pooled_result.pool_mass > isolated_result.pool_mass
    assert pooled_result.should_wake is True
    assert isolated_result.should_wake is False


def test_pool_sums_fixed_scores_and_uses_one_threshold(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    items = [_item(1, 0.61), _item(2, 0.4)]

    result = state.evaluate(_scored(state, items, now), snapshot_seq=2, now=now)

    assert result.should_wake is True
    assert result.pool_mass == pytest.approx(
        build_initial_score(0.61, has_published_at=True, wake_eligible=True)
        + build_initial_score(0.4, has_published_at=True, wake_eligible=True)
    )
    assert result.pool_mass > result.threshold
    assert result.threshold == 1.0


def test_fixed_score_decays_monotonically_then_stops_contributing(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    item = _item(1, 0.5)
    scored = _scored(state, [item], now)

    fresh = state.audit_pool(scored, now=now)
    older = state.audit_pool(scored, now=now + timedelta(hours=36))
    stale = state.audit_pool(scored, now=now + timedelta(days=8))

    assert fresh.pool_mass > older.pool_mass > 0.0
    assert stale.pool_mass == 0.0
    assert stale.below_floor == 1


def test_content_initial_score_is_immutable_per_revision(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    score = ContentScore("feed", "item:1", "1", 0.4, 0.2, now)
    state.record_content_scores((score,))
    state.record_content_scores((score,))

    with pytest.raises(RuntimeError, match="初始分 identity 冲突"):
        state.record_content_scores(
            (ContentScore("feed", "item:1", "1", 0.5, 0.2, now),)
        )


def test_old_schema_requires_installation_eventmail_migration(tmp_path) -> None:
    path = tmp_path / "wake.sqlite3"
    state = WakeState(path)
    state.initialize()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA user_version = 1")

    with pytest.raises(RuntimeError, match="EventMail 安装迁移"):
        WakeState(path).initialize()


def test_new_schema_contains_no_alert_or_context_source_tables(tmp_path) -> None:
    path = tmp_path / "wake.sqlite3"
    state = WakeState(path)
    state.initialize()

    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (8,)
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert tables == {
        "admission_state",
        "seen_content",
        "content_scores",
        "wake_runs",
        "wake_attempts",
    }


def test_same_version_schema_mutation_fails_loud(tmp_path) -> None:
    path = tmp_path / "wake.sqlite3"
    state = WakeState(path)
    state.initialize()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("ALTER TABLE seen_content RENAME TO old_seen_content")
        connection.execute("CREATE TABLE seen_content(item_identity TEXT)")

    with pytest.raises(RuntimeError, match="schema mismatch"):
        WakeState(path).initialize()


def test_dashboard_run_projection_records_one_decision(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    state.record_screen(
        run_id="run_fixture",
        owner="content",
        candidates_seen=4,
        screening=({"candidate_id": "candidate_1", "question": "New?"},),
        started_at=now,
    )
    state.record_decision(
        run_id="run_fixture",
        decision="skip",
        detail="No new capability",
        completed_at=now,
    )
    assert state.get_run("run_fixture")["decision"] == "skip"  # type: ignore[index]


def test_timer_attempt_records_no_due_check(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    state.begin_attempt(
        attempt_id="attempt:one",
        timer_id="timer:one",
        scheduled_for=now,
        fired_at=now,
    )
    state.set_attempt_mail_watermark(attempt_id="attempt:one", mail_watermark=7)
    state.finish_attempt(
        attempt_id="attempt:one",
        outcome="no_due",
        owner=None,
        detail="定时检查完成，没有可处理信件",
        completed_at=now,
    )

    assert state.count_attempts() == 1
    attempt = state.get_attempt("attempt:one")
    assert attempt is not None
    assert attempt["outcome"] == "no_due"
    assert attempt["mail_watermark"] == 7


@pytest.mark.parametrize(
    "outcome",
    (
        "content_insufficient",
        "admission_rejected",
        "shared",
        "model_skip",
        "deferred",
        "cancelled_after_fire",
        "delivery_unknown",
        "failed",
    ),
)
def test_timer_attempt_accepts_each_terminal_outcome(tmp_path, outcome: str) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / f"{outcome}.sqlite3")
    state.begin_attempt(
        attempt_id=f"attempt:{outcome}",
        timer_id="timer:one",
        scheduled_for=now,
        fired_at=now,
    )
    state.set_attempt_mail_watermark(attempt_id=f"attempt:{outcome}", mail_watermark=3)

    state.finish_attempt(
        attempt_id=f"attempt:{outcome}",
        outcome=outcome,
        owner="content",
        detail=outcome,
        completed_at=now,
    )

    assert state.get_attempt(f"attempt:{outcome}")["outcome"] == outcome  # type: ignore[index]


def test_dashboard_lists_no_due_timer_attempt(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    data_root = tmp_path / "plugin-data/wake-builtin"
    state = WakeState(data_root / "wake.sqlite3")
    state.begin_attempt(
        attempt_id="attempt:dashboard",
        timer_id="timer:dashboard",
        scheduled_for=now,
        fired_at=now,
    )
    state.set_attempt_mail_watermark(attempt_id="attempt:dashboard", mail_watermark=4)
    state.finish_attempt(
        attempt_id="attempt:dashboard",
        outcome="no_due",
        owner=None,
        detail="No due EventMail",
        completed_at=now,
    )
    app = FastAPI()
    register_dashboard(
        app,
        DashboardContext(
            plugin_id="wake",
            plugin_dir=tmp_path / "plugins/wake",
            data_root=data_root,
            validation=False,
        ),
    )

    response = TestClient(app).get("/api/dashboard/wake/attempts")

    assert response.status_code == 200
    assert response.json()["items"][0]["outcome"] == "no_due"
    assert response.json()["total"] == 1


def test_dashboard_shows_attempt_closed_by_restart(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    data_root = tmp_path / "plugin-data/wake-builtin"
    state = WakeState(data_root / "wake.sqlite3")
    state.begin_attempt(
        attempt_id="attempt:restart",
        timer_id="timer:restart",
        scheduled_for=now,
        fired_at=now,
    )
    state.set_attempt_mail_watermark(attempt_id="attempt:restart", mail_watermark=8)
    assert state.close_interrupted_attempts(now + timedelta(seconds=2)) == 1
    assert state.close_interrupted_attempts(now + timedelta(seconds=3)) == 0
    app = FastAPI()
    register_dashboard(
        app,
        DashboardContext(
            plugin_id="wake",
            plugin_dir=tmp_path / "plugins/wake",
            data_root=data_root,
            validation=False,
        ),
    )

    response = TestClient(app).get("/api/dashboard/wake/attempts/attempt%3Arestart")

    assert response.status_code == 200
    assert response.json()["outcome"] == "delivery_unknown"


def test_dashboard_exposes_fired_then_closed_attempt_without_watermark(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    data_root = tmp_path / "plugin-data/wake-builtin"
    state = WakeState(data_root / "wake.sqlite3")
    state.begin_attempt(
        attempt_id="attempt:closed",
        timer_id="timer:closed",
        scheduled_for=now,
        fired_at=now,
    )
    state.finish_attempt(
        attempt_id="attempt:closed",
        outcome="cancelled_after_fire",
        owner=None,
        detail="Timer fired before close",
        completed_at=now,
    )
    app = FastAPI()
    register_dashboard(
        app,
        DashboardContext(
            plugin_id="wake",
            plugin_dir=tmp_path / "plugins/wake",
            data_root=data_root,
            validation=False,
        ),
    )

    response = TestClient(app).get("/api/dashboard/wake/attempts/attempt%3Aclosed")

    assert response.status_code == 200
    assert response.json()["outcome"] == "cancelled_after_fire"
    assert response.json()["mail_watermark"] is None
    source = (Path(__file__).parents[1] / "plugins/wake/dashboard_panel.ts").read_text()
    assert 'cancelled_after_fire: "触发后关闭"' in source
    assert 'return value === null || value === undefined ? "未读取"' in source

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.plugin_composition import DashboardContext
from plugins.wake.dashboard import register as register_dashboard
from plugins.wake.state import WakeState


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
        "not_before": now,
        "due": True,
    }


def test_low_value_batch_advances_watermark_without_starting_turn(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    items = [_item(index, 0.001) for index in range(1, 21)]

    result = state.evaluate(items, snapshot_seq=20, now=now, random_draw=0.0)

    assert result.should_wake is False
    assert state.has_unseen_due(items, now) is False
    assert state.unseen_deadline(items) is None


def test_successful_admission_applies_refractory_to_immediate_new_item(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    first = [_item(1, 0.9)]
    assert state.evaluate(first, snapshot_seq=1, now=now, random_draw=0.0).should_wake
    assert state.has_unseen_due(first, now) is True
    state.commit_content_admission(first, now=now)

    second = [*first, _item(2, 0.9)]
    result = state.evaluate(second, snapshot_seq=2, now=now, random_draw=0.0)

    assert result.should_wake is False
    assert state.has_unseen_due(second, now) is False


def test_future_item_remains_unseen_after_current_batch_is_evaluated(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    tomorrow = now + timedelta(days=1)
    state = WakeState(tmp_path / "wake.sqlite3")
    current = _item(1, 0.001)
    future = _item(2, 0.9)
    future["not_before"] = tomorrow.isoformat()
    future["due"] = False

    first = state.evaluate((current, future), snapshot_seq=2, now=now, random_draw=0.0)

    assert first.should_wake is False
    assert state.unseen_deadline((current, future)) == tomorrow
    future["due"] = True
    second = state.evaluate(
        (current, future), snapshot_seq=2, now=tomorrow, random_draw=0.0
    )
    assert second.should_wake is True
    assert second.driver_item_id == "item:2"


@pytest.mark.parametrize(
    ("source_id", "revision"),
    (("other-feed", "1"), ("feed", "2")),
)
def test_new_mass_uses_full_content_identity(
    tmp_path, source_id: str, revision: str
) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    state = WakeState(tmp_path / "wake.sqlite3")
    old_high = _item(1, 0.9)
    old_ref = old_high["ref"]
    assert isinstance(old_ref, dict)
    old_ref["item_id"] = "shared-id"
    assert state.evaluate(
        (old_high,), snapshot_seq=1, now=now, random_draw=0.0
    ).should_wake
    state.commit_content_admission((old_high,), now=now)

    new_low = _item(2, 0.001)
    new_ref = new_low["ref"]
    assert isinstance(new_ref, dict)
    new_ref.update(
        {"source_id": source_id, "item_id": "shared-id", "revision": revision}
    )
    result = state.evaluate(
        (old_high, new_low),
        snapshot_seq=2,
        now=now + timedelta(hours=12),
        random_draw=0.0,
    )

    assert result.should_wake is False
    assert state.has_unseen_due((old_high, new_low), now) is False


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
        assert connection.execute("PRAGMA user_version").fetchone() == (7,)
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
    state.set_attempt_mail_watermark(
        attempt_id="attempt:one", mail_watermark=7
    )
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
    state.set_attempt_mail_watermark(
        attempt_id=f"attempt:{outcome}", mail_watermark=3
    )

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
    state.set_attempt_mail_watermark(
        attempt_id="attempt:dashboard", mail_watermark=4
    )
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
    state.set_attempt_mail_watermark(
        attempt_id="attempt:restart", mail_watermark=8
    )
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

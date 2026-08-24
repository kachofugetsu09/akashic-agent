import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

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


def test_v1_watermark_migrates_without_reclassifying_legacy_rows(tmp_path) -> None:
    now = datetime(2026, 8, 23, 9, tzinfo=UTC)
    path = tmp_path / "wake.sqlite3"
    state = WakeState(path)
    state.initialize()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP TABLE seen_content")
        connection.execute("UPDATE admission_state SET content_high_watermark = 3")
        connection.execute("PRAGMA user_version = 1")

    migrated = WakeState(path)
    migrated.initialize()

    assert migrated.has_unseen_due((_item(3, 0.9),), now) is False
    assert migrated.has_unseen_due((_item(4, 0.9),), now) is True


def test_same_version_schema_mutation_fails_loud(tmp_path) -> None:
    path = tmp_path / "wake.sqlite3"
    state = WakeState(path)
    state.initialize()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("ALTER TABLE seen_content RENAME TO old_seen_content")
        connection.execute("CREATE TABLE seen_content(item_identity TEXT)")

    with pytest.raises(RuntimeError, match="schema mismatch"):
        WakeState(path).initialize()

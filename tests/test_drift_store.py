from datetime import UTC, datetime, timedelta

import pytest

from plugins.drift.store import DriftStore


def test_drift_store_freezes_due_selects_and_transitions(tmp_path) -> None:
    now = datetime(2026, 8, 23, 8, tzinfo=UTC)
    store = DriftStore(tmp_path / "drift.sqlite3")
    store.initialize()
    assert store.propose(
        "reflection",
        "1",
        {"prompt": "想一想今天"},
        now,
        next_due=now + timedelta(minutes=5),
    )["inserted"] is True

    snapshot = store.snapshot(now)
    proposal = snapshot["proposals"][0]
    receipt = store.select(
        proposal["ref"],
        {"session_id": "wake:default", "turn_id": "turn:1"},
        now,
    )
    assert receipt["selected"] is True
    selected = store.selected()
    assert selected[0]["accepted_turn"]["turn_id"] == "turn:1"
    result = store.transition(selected[0]["selection_token"], "defer")
    assert result == {
        "changed": True,
        "status": "deferred",
        "next_due": (now + timedelta(minutes=5)).isoformat(),
    }


def test_drift_store_cas_loser_cannot_select_same_revision(tmp_path) -> None:
    now = datetime(2026, 8, 23, 8, tzinfo=UTC)
    store = DriftStore(tmp_path / "drift.sqlite3")
    store.initialize()
    store.propose("reflection", "1", {}, now)
    proposal = store.snapshot(now)["proposals"][0]

    first = store.select(
        proposal["ref"],
        {"session_id": "wake:default", "turn_id": "turn:1"},
        now,
    )
    second = store.select(
        proposal["ref"],
        {"session_id": "wake:default", "turn_id": "turn:2"},
        now,
    )
    assert first["selected"] is True
    assert second["selected"] is False


def test_drift_read_only_candidate_validates_without_writing(tmp_path) -> None:
    path = tmp_path / "drift.sqlite3"
    formal = DriftStore(path)
    formal.initialize()
    candidate = DriftStore(path, data_access="read_only")
    candidate.initialize()
    assert candidate.snapshot(datetime.now(UTC))["proposals"] == ()
    with pytest.raises(PermissionError, match="read-only candidate"):
        candidate.propose("forbidden", "1", {}, datetime.now(UTC))

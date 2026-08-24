import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from plugins.drift.store import DriftStore


def test_drift_store_freezes_due_selects_and_transitions(tmp_path) -> None:
    now = datetime(2026, 8, 23, 8, tzinfo=UTC)
    store = DriftStore(tmp_path / "drift.sqlite3")
    store.initialize()
    proposed = store.propose(
        "reflection",
        "1",
        {"prompt": "想一想今天"},
        now,
        next_due=now + timedelta(minutes=5),
    )
    assert proposed == {
        "inserted": True,
        "ref": {
            "proposal_id": "reflection",
            "revision": "1",
            "state_version": 1,
        },
    }

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


def test_drift_same_turn_second_proposal_is_explicit_cas_loser(tmp_path) -> None:
    now = datetime(2026, 8, 23, 8, tzinfo=UTC)
    store = DriftStore(tmp_path / "drift.sqlite3")
    store.initialize()
    store.propose("one", "1", {}, now)
    store.propose("two", "1", {}, now)
    first, second = store.snapshot(now)["proposals"]
    accepted = {"session_id": "wake:default", "turn_id": "turn:1"}

    assert store.select(first["ref"], accepted, now)["selected"] is True
    loser = store.select(second["ref"], accepted, now)

    assert loser == {
        "selected": False,
        "reason": "turn_already_selected",
        "selection_token": None,
        "accepted_turn": None,
    }
    assert store.selection(accepted)["ref"] == {
        "proposal_id": "one",
        "revision": "1",
    }


def test_drift_ready_delivery_preserves_turn_and_settles_once(tmp_path) -> None:
    now = datetime(2026, 8, 23, 8, tzinfo=UTC)
    store = DriftStore(tmp_path / "drift.sqlite3")
    store.initialize()
    store.propose("reflection", "1", {}, now)
    proposal = store.snapshot(now)["proposals"][0]
    accepted = {"session_id": "wake:default", "turn_id": "turn:share"}
    selected = store.select(proposal["ref"], accepted, now)
    token = selected["selection_token"]
    assert isinstance(token, str)

    assert store.transition(token, "ready_for_delivery") == {
        "changed": True,
        "status": "ready_for_delivery",
        "next_due": None,
    }
    assert store.pending_delivery() == (
        {
            "selection_token": token,
            "accepted_turn": accepted,
            "message_metadata": {
                "tools_used": ["message_push"],
                "evidence_item_ids": [],
                "source_refs": [],
                "state_summary_tag": "none",
            },
        },
    )
    first = store.settle_delivery(token, "wake:delivery")
    second = store.settle_delivery(token, "wake:delivery")

    assert first["settled"] is True and first["duplicate"] is False
    assert second["settled"] is True and second["duplicate"] is True
    assert first["receipt"] == second["receipt"]
    assert store.delivery(accepted)["status"] == "settled"


def test_drift_v1_orphaned_ready_row_is_invalidated_without_guessing_body(
    tmp_path,
) -> None:
    path = tmp_path / "drift.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE proposals(
            proposal_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            due_at TEXT NOT NULL,
            next_due TEXT,
            state_version INTEGER NOT NULL,
            selection_token TEXT UNIQUE,
            selected_session_id TEXT,
            selected_turn_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(proposal_id, revision)
        );
        CREATE UNIQUE INDEX proposals_selected_turn_idx
        ON proposals(selected_session_id, selected_turn_id)
        WHERE selected_turn_id IS NOT NULL;
        CREATE INDEX proposals_due_idx ON proposals(status, due_at);
        INSERT INTO proposals VALUES(
            'reflection', '1', '{}', 'ready_for_delivery',
            '2026-08-23T08:00:00+00:00', NULL, 2,
            NULL, NULL, NULL,
            '2026-08-23T08:00:00+00:00', '2026-08-23T08:01:00+00:00'
        );
        PRAGMA user_version = 1;
        """)
    connection.close()

    store = DriftStore(path)
    store.initialize()

    assert store.snapshot(datetime(2026, 8, 23, 9, tzinfo=UTC))["proposals"] == ()
    connection = sqlite3.connect(path)
    status, state_version = connection.execute(
        "SELECT status, state_version FROM proposals"
    ).fetchone()
    connection.close()
    assert (status, state_version) == ("invalidated", 3)


@pytest.mark.parametrize("mutation", ["missing_index", "extra_table"])
def test_drift_rejects_same_version_schema_topology_drift(tmp_path, mutation) -> None:
    path = tmp_path / "drift.sqlite3"
    store = DriftStore(path)
    store.initialize()
    connection = sqlite3.connect(path)
    if mutation == "missing_index":
        connection.execute("DROP INDEX proposals_due_idx")
    else:
        connection.execute("CREATE TABLE unexpected(value TEXT)")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="Drift .*不匹配"):
        store.initialize()


def test_drift_rejects_constraint_free_same_version_table(tmp_path) -> None:
    path = tmp_path / "drift.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE proposals(
            proposal_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            due_at TEXT NOT NULL,
            next_due TEXT,
            state_version INTEGER NOT NULL,
            selection_token TEXT,
            selected_session_id TEXT,
            selected_turn_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX proposals_selected_turn_idx
        ON proposals(selected_session_id, selected_turn_id)
        WHERE selected_turn_id IS NOT NULL;
        CREATE INDEX proposals_due_idx ON proposals(status, due_at);
        PRAGMA user_version = 1;
        """)
    connection.close()

    with pytest.raises(RuntimeError, match="constraint-bearing table SQL"):
        DriftStore(path).initialize()

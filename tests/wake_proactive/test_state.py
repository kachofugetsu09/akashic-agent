from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from plugins.wake_proactive.state import WakeStateStore


@pytest.fixture
def store(tmp_path):
    state = WakeStateStore(tmp_path / "wake.db")
    try:
        yield state
    finally:
        state.close()


def _snapshot(source_id: str, now: datetime, *, expires_at: datetime | None = None) -> dict:
    return {
        "_source": source_id,
        "presence": "active",
        "interruptibility": 0.75,
        "confidence": 0.9,
        "observed_at": now.isoformat(),
        "expires_at": expires_at.isoformat() if expires_at is not None else None,
    }


def test_list_contexts_reads_one_ordered_snapshot_and_roundtrips(store: WakeStateStore) -> None:
    now = datetime(2026, 7, 23, 3, 0, 0)

    assert store.list_contexts() == []
    store.ingest_context(
        [
            _snapshot("z-source", now, expires_at=now + timedelta(minutes=5)),
            _snapshot("a-source", now),
        ],
        now,
    )

    statements: list[str] = []
    store._conn.set_trace_callback(statements.append)
    contexts = store.list_contexts()
    store._conn.set_trace_callback(None)

    assert [context.raw["_source"] for context in contexts] == [
        "a-source",
        "z-source",
    ]
    assert contexts[0].interruptibility == 0.75
    assert contexts[1].expires_at == now + timedelta(minutes=5)
    select_statements = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT")
    ]
    assert select_statements == [
        "SELECT * FROM context_state ORDER BY source_id"
    ]


def test_load_context_missing_returns_none(store: WakeStateStore) -> None:
    assert store.load_context("missing-source") is None


@pytest.mark.parametrize(
    ("column", "value", "error_type"),
    [
        ("payload_json", "not-json", json.JSONDecodeError),
        ("observed_at", "not-time", ValueError),
    ],
)
def test_context_row_corruption_fails_loud(
    store: WakeStateStore,
    column: str,
    value: str,
    error_type: type[Exception],
) -> None:
    now = datetime(2026, 7, 23, 3, 0, 0)
    store.ingest_context([_snapshot("broken-source", now)], now)
    _ = store._conn.execute(
        f"UPDATE context_state SET {column} = ? WHERE source_id = ?",
        (value, "broken-source"),
    )
    store._conn.commit()

    with pytest.raises(error_type):
        store.list_contexts()


def test_reservoir_quarantine_is_identity_upsert_and_score_validation(store: WakeStateStore) -> None:
    now = datetime(2026, 7, 23, 3, 0, 0, tzinfo=UTC)
    store.ingest(
        "content",
        [
            {
                "ack_server": "feed",
                "source_id": "source",
                "event_id": "bad",
                "preprocess_score": float("nan"),
                "published_at": now.isoformat(),
            }
        ],
        now,
    )
    store.ingest(
        "content",
        [
            {
                "ack_server": "feed",
                "source_id": "source",
                "event_id": "bad",
                "preprocess_score": float("inf"),
                "published_at": now.isoformat(),
            }
        ],
        now,
    )

    items = store.quarantined()
    assert len(items) == 1
    assert items[0]["identity"] == "feed:feed:bad"
    assert items[0]["occurrences"] == 2


def test_expiry_ack_deletes_payload_only_after_successful_ack(store: WakeStateStore) -> None:
    now = datetime(2026, 7, 23, 3, 0, 0, tzinfo=UTC)
    store.ingest(
        "content",
        [
            {
                "ack_server": "feed",
                "source_id": "source",
                "event_id": "expire-me",
                "preprocess_score": 0.1,
                "published_at": now.isoformat(),
            }
        ],
        now,
    )
    store.queue_expiration(["feed:expire-me"], now)
    assert store.pending_acknowledgement_batches()[0]["action"] == "expire"
    assert store.unread("content") == []
    assert store._conn.execute(
        "SELECT payload_json FROM reservoir_events WHERE item_id = ?",
        ("feed:expire-me",),
    ).fetchone() is not None

    store.mark_acknowledged("feed", ["expire-me"])
    assert store._conn.execute(
        "SELECT 1 FROM reservoir_events WHERE item_id = ?",
        ("feed:expire-me",),
    ).fetchone() is None

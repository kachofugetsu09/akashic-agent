from __future__ import annotations

import json
from datetime import datetime, timedelta

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

from __future__ import annotations

import hashlib
import inspect
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from plugins.content import plugin as content_plugin
from plugins.content.store import ContentIdentityConflict, ContentStore


def _item(
    item_id: str,
    *,
    revision: str = "1",
    value: str | None = None,
    not_before: datetime | None = None,
    requires_ack: bool = True,
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "revision": revision,
        "payload": {"value": value or item_id},
        "not_before": not_before,
        "requires_ack": requires_ack,
    }


def _select(store: ContentStore, now: datetime, item_id: str = "one") -> str:
    snapshot = store.snapshot(now)
    candidate = next(
        item for item in snapshot["items"] if item["ref"]["item_id"] == item_id
    )
    result = store.select(
        candidate["ref"],
        snapshot["snapshot_seq"],
        {"session_id": "wake:fixture", "turn_id": f"turn:{item_id}"},
        now,
    )
    assert result["selected"] is True
    token = result["selection_token"]
    assert isinstance(token, str)
    return token


def test_submit_reuses_batch_receipt_and_revision_without_duplicate_item(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    items = [_item("one", not_before=now)]

    first = store.submit("fitbit", "poll:1", items)
    repeated = store.submit("fitbit", "poll:1", items)
    another_batch = store.submit("fitbit", "poll:2", items)

    assert first == repeated
    assert first["inserted"] == [
        {"source_id": "fitbit", "item_id": "one", "revision": "1"}
    ]
    assert another_batch["inserted"] == []
    assert another_batch["duplicates"] == first["inserted"]
    assert another_batch["high_watermark"] == 1
    assert store.state_counts() == {"pending": 1}


def test_missing_not_before_stays_idempotent_across_later_poll(tmp_path) -> None:
    store = ContentStore(tmp_path / "content.sqlite3")
    item = _item("one")

    _ = store.submit("feed", "poll:1", [item])
    repeated = store.submit("feed", "poll:1", [item])
    later_poll = store.submit("feed", "poll:2", [item])

    assert repeated["receipt_id"] == "content-submit:feed:poll:1"
    assert later_poll["duplicates"] == [
        {"source_id": "feed", "item_id": "one", "revision": "1"}
    ]


def test_stable_batch_and_revision_identity_reject_different_content(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit("feed", "poll:1", [_item("one", value="a", not_before=now)])

    with pytest.raises(ContentIdentityConflict, match="batch identity"):
        store.submit("feed", "poll:1", [_item("one", value="b", not_before=now)])
    with pytest.raises(ContentIdentityConflict, match="revision identity"):
        store.submit("feed", "poll:2", [_item("one", value="b", not_before=now)])
    with pytest.raises(ContentIdentityConflict, match="revision identity"):
        store.submit(
            "feed",
            "poll:3",
            [_item("one", value="a", not_before=now + timedelta(seconds=1))],
        )


def test_frozen_high_watermark_selection_keeps_new_item_for_next_wake(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit("feed", "poll:1", [_item("one", not_before=now)])
    frozen = store.snapshot(now)

    _ = store.submit("feed", "poll:2", [_item("two", not_before=now)])
    selected = store.select(
        frozen["items"][0]["ref"],
        frozen["snapshot_seq"],
        {"session_id": "wake:fixture", "turn_id": "turn:one"},
        now,
    )

    assert selected["selected"] is True
    assert selected["wake_needed"] is True
    assert [item["ref"]["item_id"] for item in store.snapshot(now)["items"]] == ["two"]
    assert store.state_counts() == {"pending": 1, "selected": 1}


def test_cas_selection_allows_only_one_turn_for_one_revision(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit("feed", "poll:1", [_item("one", not_before=now)])
    snapshot = store.snapshot(now)
    ref = snapshot["items"][0]["ref"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda turn: store.select(
                    ref,
                    snapshot["snapshot_seq"],
                    {"session_id": "wake:fixture", "turn_id": turn},
                    now,
                ),
                ("turn:a", "turn:b"),
            )
        )

    assert sum(result["selected"] is True for result in results) == 1
    assert store.state_counts() == {"selected": 1}


def test_selection_recovers_after_wake_loses_token(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    path = tmp_path / "content.sqlite3"
    store = ContentStore(path)
    _ = store.submit("feed", "poll:1", [_item("one", not_before=now)])
    snapshot = store.snapshot(now)
    accepted = {"session_id": "wake:recovery", "turn_id": "turn:accepted"}
    selected = store.select(
        snapshot["items"][0]["ref"], snapshot["snapshot_seq"], accepted, now
    )

    restarted = ContentStore(path)
    restarted.initialize()
    recovered = restarted.selection(accepted)

    assert recovered is not None
    assert recovered["selection_token"] == selected["selection_token"]
    assert recovered["ref"]["item_id"] == "one"
    assert recovered["payload"] == {"value": "one"}
    assert recovered["status"] == "selected"
    assert recovered["accepted_turn"] == accepted
    assert "settlement_ref" not in recovered


def test_selected_recovers_same_tokens_in_snapshot_order_with_limit(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    path = tmp_path / "content.sqlite3"
    store = ContentStore(path)
    _ = store.submit(
        "feed",
        "poll:1",
        (
            _item("one", not_before=now),
            _item("two", not_before=now),
            _item("three", not_before=now),
        ),
    )
    tokens = tuple(
        _select(store, now, item_id) for item_id in ("one", "two", "three")
    )

    restarted = ContentStore(path)
    recovered = restarted.selected(limit=2)

    assert tuple(row["selection_token"] for row in recovered) == tokens[:2]
    assert tuple(row["ref"]["item_id"] for row in recovered) == ("one", "two")
    assert tuple(row["accepted_turn"] for row in recovered) == (
        {"session_id": "wake:fixture", "turn_id": "turn:one"},
        {"session_id": "wake:fixture", "turn_id": "turn:two"},
    )
    assert restarted.selected(limit=1) == recovered[:1]


def test_selected_excludes_ready_for_delivery_and_rejects_invalid_limit(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit(
        "feed",
        "poll:1",
        (_item("one", not_before=now), _item("two", not_before=now)),
    )
    first = _select(store, now, "one")
    second = _select(store, now, "two")
    _ = store.transition(first, "ready_for_delivery")

    assert tuple(row["selection_token"] for row in store.selected()) == (second,)
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="limit 必须是正整数"):
            store.selected(limit)  # pyright: ignore[reportArgumentType]


def test_one_accepted_turn_cannot_select_two_items_concurrently(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit(
        "feed",
        "poll:1",
        [_item("one", not_before=now), _item("two", not_before=now)],
    )
    snapshot = store.snapshot(now)
    accepted = {"session_id": "wake:one", "turn_id": "turn:shared"}

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda item: store.select(
                    item["ref"], snapshot["snapshot_seq"], accepted, now
                ),
                snapshot["items"],
            )
        )

    assert sum(result["selected"] is True for result in results) == 1
    rejected = next(result for result in results if result["selected"] is False)
    assert rejected.get("reason") == "turn_already_selected"
    assert store.selection(accepted) is not None
    assert store.state_counts() == {"pending": 1, "selected": 1}


def test_selection_is_missing_or_isolated_by_full_turn_receipt(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit(
        "feed",
        "poll:1",
        [_item("one", not_before=now), _item("two", not_before=now)],
    )
    snapshot = store.snapshot(now)
    first = {"session_id": "wake:a", "turn_id": "turn:same"}
    second = {"session_id": "wake:b", "turn_id": "turn:same"}

    assert store.selection(first) is None
    assert (
        store.select(snapshot["items"][0]["ref"], snapshot["snapshot_seq"], first, now)[
            "selected"
        ]
        is True
    )
    assert (
        store.select(
            snapshot["items"][1]["ref"], snapshot["snapshot_seq"], second, now
        )["selected"]
        is True
    )

    assert store.selection(first)["ref"]["item_id"] == "one"
    assert store.selection(second)["ref"]["item_id"] == "two"
    assert (
        store.selection({"session_id": "wake:missing", "turn_id": "turn:same"}) is None
    )


def test_deferred_selection_keeps_turn_owner_and_recovery_token(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit(
        "feed",
        "poll:1",
        [_item("one", not_before=now), _item("two", not_before=now)],
    )
    snapshot = store.snapshot(now)
    accepted = {"session_id": "wake:defer", "turn_id": "turn:complete"}
    selected = store.select(
        snapshot["items"][0]["ref"], snapshot["snapshot_seq"], accepted, now
    )
    token = selected["selection_token"]
    assert isinstance(token, str)
    _ = store.transition(token, "defer", not_before=now)
    fresh = store.snapshot(now)
    other = next(item for item in fresh["items"] if item["ref"]["item_id"] == "two")

    repeated = store.select(other["ref"], fresh["snapshot_seq"], accepted, now)
    recovered = store.selection(accepted)

    assert repeated["selected"] is False
    assert repeated.get("reason") == "turn_already_selected"
    assert recovered is not None
    assert recovered["selection_token"] == token
    assert recovered["status"] == "deferred"


def test_item_state_version_rejects_stale_snapshot_after_defer(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit("feed", "poll:1", [_item("one", not_before=now)])
    stale = store.snapshot(now)
    token = _select(store, now)
    _ = store.transition(token, "defer", not_before=now)

    rejected = store.select(
        stale["items"][0]["ref"],
        stale["snapshot_seq"],
        {"session_id": "wake:fixture", "turn_id": "turn:stale"},
        now,
    )
    fresh = store.snapshot(now)
    accepted = store.select(
        fresh["items"][0]["ref"],
        fresh["snapshot_seq"],
        {"session_id": "wake:fixture", "turn_id": "turn:fresh"},
        now,
    )

    assert rejected["selected"] is False
    assert accepted["selected"] is True
    assert accepted["accepted_turn"] == {
        "session_id": "wake:fixture",
        "turn_id": "turn:fresh",
    }
    connection = sqlite3.connect(store.path)
    selected_owner = connection.execute("""
        SELECT selected_session_id, selected_turn_id FROM items
        WHERE source_id = 'feed' AND item_id = 'one' AND revision = '1'
        """).fetchone()
    connection.close()
    assert selected_owner == ("wake:fixture", "turn:fresh")


def test_decline_transitions_recompute_wake_without_timer_state(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit("feed", "poll:1", [_item("one", not_before=now)])
    token = _select(store, now)
    later = now + timedelta(hours=1)

    deferred = store.transition(token, "defer", not_before=later)
    before_due = store.snapshot(now)
    at_due = store.snapshot(later)

    assert deferred.get("status") == "deferred"
    assert before_due["wake_needed"] is True
    assert before_due["items"][0]["due"] is False
    assert at_due["items"][0]["due"] is True


def test_source_bound_unsettled_and_ack_cannot_cross_source(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit("fitbit", "poll:1", [_item("sleep", not_before=now)])
    token = _select(store, now, "sleep")
    assert store.transition(token, "ready_for_delivery")["changed"] is True
    assert (
        store.transition(token, "delivered", settlement_ref="delivery:settle:1").get(
            "status"
        )
        == "delivered"
    )

    assert store.unsettled("feed") == ()
    assert store.ack("feed", "delivery:settle:1") == {
        "settled": False,
        "reason": "settlement_missing",
    }
    assert [row["settlement_ref"] for row in store.unsettled("fitbit")] == [
        "delivery:settle:1"
    ]
    assert store.ack("fitbit", "delivery:settle:1") == {
        "settled": True,
        "duplicate": False,
    }
    assert store.ack("fitbit", "delivery:settle:1") == {
        "settled": True,
        "duplicate": True,
    }
    assert store.state_counts() == {"settled": 1}


def test_context_without_provider_ack_settles_at_delivery(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit(
        "steam",
        "poll:1",
        [_item("context", not_before=now, requires_ack=False)],
    )
    token = _select(store, now, "context")
    _ = store.transition(token, "ready_for_delivery")

    delivered = store.transition(
        token, "delivered", settlement_ref="delivery:context:1"
    )

    assert delivered.get("status") == "settled"
    assert store.unsettled("steam") == ()
    assert store.state_counts() == {"settled": 1}


@pytest.mark.parametrize(
    ("requires_ack", "expected_status"),
    ((True, "delivered"), (False, "settled")),
)
def test_delivery_capability_is_body_free_and_replays_stable_receipt(
    tmp_path,
    requires_ack: bool,
    expected_status: str,
) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit(
        "fitbit",
        "poll:delivery",
        [_item("delivery", not_before=now, requires_ack=requires_ack)],
    )
    token = _select(store, now, "delivery")
    _ = store.transition(token, "ready_for_delivery")
    accepted = {"session_id": "wake:fixture", "turn_id": "turn:delivery"}
    delivery = content_plugin._DeliveryServices(store)

    assert delivery.pending() == (
        {"selection_token": token, "accepted_turn": accepted},
    )
    first = delivery.settle(token, "wake:logical-delivery")
    recovered = delivery.lookup(accepted)
    duplicate = delivery.settle(token, "wake:logical-delivery")

    assert first["status"] == expected_status
    assert first["receipt"] == duplicate["receipt"]
    assert recovered == {
        "selection_token": token,
        "accepted_turn": accepted,
        "status": expected_status,
        "settlement_ref": "wake:logical-delivery",
        "receipt": first["receipt"],
    }
    assert delivery.pending() == ()
    assert recovered is not None
    assert set(recovered) == {
        "selection_token",
        "accepted_turn",
        "status",
        "settlement_ref",
        "receipt",
    }

    if requires_ack:
        assert store.ack("fitbit", "wake:logical-delivery")["settled"] is True
        after_ack = delivery.lookup(accepted)
        assert after_ack is not None
        assert after_ack["status"] == "settled"
        assert after_ack["receipt"] == first["receipt"]


def test_delivery_capability_rejects_conflicting_settlement_identity(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit("fitbit", "poll:1", [_item("one", not_before=now)])
    token = _select(store, now)
    _ = store.transition(token, "ready_for_delivery")
    delivery = content_plugin._DeliveryServices(store)
    _ = delivery.settle(token, "wake:one")

    with pytest.raises(RuntimeError, match="settlement identity conflict"):
        delivery.settle(token, "wake:two")


def test_wake_capability_cannot_commit_delivery_but_can_abandon_ready_item(
    tmp_path,
) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit("feed", "poll:1", [_item("one", not_before=now)])
    token = _select(store, now)
    wake = content_plugin._WakeServices(store)
    assert wake.transition(token, "ready_for_delivery")["changed"] is True

    with pytest.raises(ValueError, match="不拥有 transition: delivered"):
        wake.transition(token, "delivered")
    assert "settlement_ref" not in inspect.signature(wake.transition).parameters
    assert wake.transition(token, "abandoned")["status"] == "abandoned"


@pytest.mark.parametrize(
    "action",
    (
        "ready_for_delivery",
        "defer",
        "await_change",
        "invalidated",
        "abandoned",
        "expired",
    ),
)
def test_non_delivery_transition_cannot_write_settlement_ref(
    tmp_path, action: str
) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    store = ContentStore(tmp_path / "content.sqlite3")
    _ = store.submit("feed", "poll:1", [_item("one", not_before=now)])
    token = _select(store, now)
    not_before = now + timedelta(hours=1) if action == "defer" else None

    with pytest.raises(ValueError, match="只有 delivered"):
        store.transition(
            token,
            action,
            not_before=not_before,
            settlement_ref="delivery:forbidden",
        )

    connection = sqlite3.connect(store.path)
    persisted = connection.execute(
        "SELECT status, settlement_ref FROM items WHERE selection_token = ?",
        (token,),
    ).fetchone()
    connection.close()
    assert persisted == ("selected", None)


def test_source_id_has_one_bound_owner_per_root(tmp_path) -> None:
    services = content_plugin._SourceServices(
        ContentStore(tmp_path / "content.sqlite3"),
        lambda: None,
    )
    first = services.bind("fitbit")

    with pytest.raises(RuntimeError, match="已有 owner"):
        services.bind("fitbit")

    assert first.unsettled() == ()


def test_source_bound_exact_reads_do_not_write_or_emit_change(tmp_path) -> None:
    path = tmp_path / "content.sqlite3"
    changed = 0

    def record_changed() -> None:
        nonlocal changed
        changed += 1

    bound = content_plugin._SourceServices(ContentStore(path), record_changed).bind(
        "feed-subscriptions"
    )
    receipt = bound.submit("legacy:event-1", [_item("event-1", revision="rev-1")])
    assert changed == 1
    before = {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in tmp_path.iterdir()
        if entry.is_file()
    }

    assert bound.read_submission("legacy:event-1") == receipt
    assert bound.read_revision("event-1", "rev-1")["ref"] == {
        "source_id": "feed-subscriptions",
        "item_id": "event-1",
        "revision": "rev-1",
    }

    after = {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in tmp_path.iterdir()
        if entry.is_file()
    }
    assert changed == 1
    assert after == before


def test_exact_read_rejects_uncheckpointed_wal_instead_of_missing_row(tmp_path) -> None:
    path = tmp_path / "content.sqlite3"
    store = ContentStore(path)
    _ = store.submit("feed-subscriptions", "one", [_item("one")])
    writer = sqlite3.connect(path)
    _ = writer.execute("PRAGMA journal_mode = WAL")
    _ = writer.execute(
        "UPDATE content_state SET state_version = state_version + 1 WHERE singleton=1"
    )
    writer.commit()
    assert path.with_name(path.name + "-wal").stat().st_size > 0

    with pytest.raises(RuntimeError, match="checkpointed offline store"):
        store.read_submission("feed-subscriptions", "one")

    writer.close()
    assert store.read_submission("feed-subscriptions", "one") is not None


def test_read_only_store_reads_formal_state_and_rejects_every_write(tmp_path) -> None:
    now = datetime(2026, 8, 23, 5, tzinfo=UTC)
    path = tmp_path / "content.sqlite3"
    formal = ContentStore(path)
    _ = formal.submit("feed", "poll:1", [_item("one", not_before=now)])
    snapshot = formal.snapshot(now)
    token = _select(formal, now)
    candidate = ContentStore(path, data_access="read_only")

    candidate.initialize()
    assert candidate.snapshot(now)["snapshot_seq"] == snapshot["snapshot_seq"]
    assert candidate.selection(
        {"session_id": "wake:fixture", "turn_id": "turn:one"}
    )["selection_token"] == token
    assert candidate.unsettled("feed") == ()
    assert candidate.state_counts() == {"selected": 1}

    with pytest.raises(PermissionError, match="read-only candidate"):
        candidate.submit("feed", "poll:2", [_item("two", not_before=now)])
    with pytest.raises(PermissionError, match="read-only candidate"):
        candidate.select(
            snapshot["items"][0]["ref"],
            snapshot["snapshot_seq"],
            {"session_id": "wake:candidate", "turn_id": "turn:write"},
            now,
        )
    with pytest.raises(PermissionError, match="read-only candidate"):
        candidate.transition(token, "ready_for_delivery")
    with pytest.raises(PermissionError, match="read-only candidate"):
        candidate.ack("feed", "delivery:missing")


def test_read_only_initialize_does_not_create_database_or_parent(tmp_path) -> None:
    path = tmp_path / "missing" / "content.sqlite3"

    with pytest.raises(sqlite3.OperationalError, match="open database"):
        ContentStore(path, data_access="read_only").initialize()

    assert not path.parent.exists()


def test_initialize_rejects_unknown_or_malformed_schema(tmp_path) -> None:
    unknown = tmp_path / "unknown.sqlite3"
    connection = sqlite3.connect(unknown)
    connection.execute("PRAGMA user_version = 99")
    connection.close()
    with pytest.raises(RuntimeError, match="schema version: 99"):
        ContentStore(unknown).initialize()

    malformed = tmp_path / "malformed.sqlite3"
    connection = sqlite3.connect(malformed)
    connection.execute("CREATE TABLE items(wrong TEXT)")
    connection.execute("PRAGMA user_version = 1")
    connection.close()
    with pytest.raises(RuntimeError, match="schema mismatch"):
        ContentStore(malformed).initialize()


def test_initialize_rejects_constraint_free_schema_with_same_columns(tmp_path) -> None:
    path = tmp_path / "lookalike.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE content_state(
            singleton INTEGER,
            next_seq INTEGER NOT NULL,
            state_version INTEGER NOT NULL,
            wake_needed INTEGER NOT NULL,
            earliest_not_before TEXT
        );
        INSERT INTO content_state VALUES(1, 0, 0, 0, NULL);
        CREATE TABLE items(
            source_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            snapshot_seq INTEGER NOT NULL,
            status TEXT NOT NULL,
            not_before TEXT NOT NULL,
            requires_ack INTEGER NOT NULL,
            item_state_version INTEGER NOT NULL,
            selection_token TEXT,
            selected_session_id TEXT,
            selected_turn_id TEXT,
            settlement_ref TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE submissions(
            source_id TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            submitted_at TEXT NOT NULL
        );
        PRAGMA user_version = 1;
        """)
    connection.close()

    with pytest.raises(RuntimeError, match="schema mismatch"):
        ContentStore(path).initialize()


def test_initialize_rejects_missing_index_and_singleton_row(tmp_path) -> None:
    missing_index = ContentStore(tmp_path / "missing-index.sqlite3")
    missing_index.initialize()
    connection = sqlite3.connect(missing_index.path)
    connection.execute("DROP INDEX items_wake_idx")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="items indexes"):
        missing_index.initialize()

    missing_state = ContentStore(tmp_path / "missing-state.sqlite3")
    missing_state.initialize()
    connection = sqlite3.connect(missing_state.path)
    connection.execute("DELETE FROM content_state")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="content_state singleton row"):
        missing_state.initialize()


def test_initialize_rejects_physically_corrupt_sqlite(tmp_path) -> None:
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"not a sqlite database")

    with pytest.raises(sqlite3.DatabaseError, match="database|encrypted"):
        ContentStore(path).initialize()

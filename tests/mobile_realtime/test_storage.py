from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import infra.mobile_realtime.storage as storage_module

from infra.mobile_realtime.storage import (
    AckOverflowError,
    AckRollbackError,
    CommandConflictError,
    CommandReceiptCapacityError,
    DeviceRecord,
    MobileRealtimeStorage,
    PairingExpiredError,
    PairingSessionRecord,
    PairingStateError,
    SentCursorError,
    ServerIdentityConflictError,
    ServerIdentityReference,
)


NOW = datetime(2026, 7, 14, 9, 30, tzinfo=timezone.utc)


def _device(device_id: str = "device-1") -> DeviceRecord:
    return DeviceRecord(
        device_id=device_id,
        public_key=f"public-key:{device_id}",
        display_name=f"Phone {device_id}",
        created_at=NOW,
        revoked_at=None,
        capabilities=("chat", "attachments"),
    )


def _event_json(event_id: str) -> str:
    return json.dumps(
        {
            "v": 1,
            "kind": "event",
            "type": "message.final",
            "id": event_id,
            "payload": {"text": event_id},
        },
        separators=(",", ":"),
    )


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[MobileRealtimeStorage]:
    value = MobileRealtimeStorage(tmp_path / "mobile.db")
    try:
        yield value
    finally:
        value.close()


def test_database_uses_wal_and_keeps_server_identity_stable(
    storage: MobileRealtimeStorage,
) -> None:
    reference = ServerIdentityReference(
        server_id="server-1",
        keyset_manifest_path="keys/keyset-v1/manifest.json",
        public_key_fingerprint="sha256:server-1",
    )
    storage.write_server_identity(reference)
    storage.write_server_identity(
        ServerIdentityReference(
            server_id="server-1",
            keyset_manifest_path="keys/keyset-v2/manifest.json",
            public_key_fingerprint="sha256:server-1",
        )
    )

    assert storage.read_server_identity() == ServerIdentityReference(
        server_id="server-1",
        keyset_manifest_path="keys/keyset-v2/manifest.json",
        public_key_fingerprint="sha256:server-1",
    )
    with closing(sqlite3.connect(storage.db_path)) as db, db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)

    with pytest.raises(ServerIdentityConflictError):
        storage.write_server_identity(
            ServerIdentityReference(
                server_id="server-2",
                keyset_manifest_path="keys/keyset-v3/manifest.json",
                public_key_fingerprint="sha256:server-2",
            )
        )


def test_durable_event_range_detects_missing_sequence(
    storage: MobileRealtimeStorage,
) -> None:
    storage.register_device(_device())
    for index in range(3):
        storage.append_durable_event(
            device_id="device-1",
            event_id=f"event-{index + 1}",
            envelope_json=_event_json(f"event-{index + 1}"),
            created_at=NOW,
        )

    assert storage.durable_event_range_is_contiguous(
        "device-1",
        after_event_seq=0,
        through_event_seq=3,
    )
    with closing(sqlite3.connect(storage.db_path)) as db, db:
        db.execute(
            "DELETE FROM mobile_device_inbox WHERE device_id = ? AND event_seq = ?",
            ("device-1", 2),
        )
    assert not storage.durable_event_range_is_contiguous(
        "device-1",
        after_event_seq=0,
        through_event_seq=3,
    )


def test_pairing_confirmation_and_consumption_are_one_time(
    storage: MobileRealtimeStorage,
) -> None:
    pairing = PairingSessionRecord(
        pairing_id="pairing-1",
        secret_hash="sha256:secret",
        expires_at=NOW + timedelta(minutes=5),
        status="pending",
    )
    storage.create_pairing_session(pairing)

    with pytest.raises(PairingStateError):
        storage.consume_pairing("pairing-1", _device(), now=NOW)

    confirmed = storage.confirm_pairing("pairing-1", now=NOW)
    assert confirmed.status == "confirmed"
    storage.consume_pairing("pairing-1", _device(), now=NOW)

    consumed = storage.read_pairing_session("pairing-1")
    assert consumed is not None
    assert consumed.status == "consumed"
    assert consumed.secret_hash is None
    assert storage.read_device("device-1") == _device()
    assert storage.read_cursor("device-1").next_event_seq == 1

    with pytest.raises(PairingStateError):
        storage.consume_pairing("pairing-1", _device("device-2"), now=NOW)


def test_pairing_expiry_and_failed_device_insert_leave_secret_usable(
    storage: MobileRealtimeStorage,
) -> None:
    storage.register_device(_device())
    storage.create_pairing_session(
        PairingSessionRecord(
            pairing_id="expired",
            secret_hash="hash-expired",
            expires_at=NOW,
            status="pending",
        )
    )
    with pytest.raises(PairingExpiredError):
        storage.confirm_pairing("expired", now=NOW)

    storage.create_pairing_session(
        PairingSessionRecord(
            pairing_id="pairing-duplicate-device",
            secret_hash="hash-active",
            expires_at=NOW + timedelta(minutes=5),
            status="pending",
        )
    )
    storage.confirm_pairing("pairing-duplicate-device", now=NOW)
    with pytest.raises(sqlite3.IntegrityError):
        storage.consume_pairing(
            "pairing-duplicate-device",
            replace(_device(), public_key="different-public-key"),
            now=NOW,
        )

    unchanged = storage.read_pairing_session("pairing-duplicate-device")
    assert unchanged is not None
    assert unchanged.status == "confirmed"
    assert unchanged.secret_hash == "hash-active"


def test_repairing_same_public_key_preserves_device_cursor_and_sessions(
    storage: MobileRealtimeStorage,
) -> None:
    original = _device()
    storage.register_device(original)
    storage.claim_session(
        device_id=original.device_id,
        session_id="mobile:session-1",
        created_at=NOW,
    )
    storage.create_pairing_session(
        PairingSessionRecord(
            pairing_id="pairing-repair",
            secret_hash="hash-repair",
            expires_at=NOW + timedelta(minutes=5),
            status="pending",
        )
    )
    storage.confirm_pairing("pairing-repair", now=NOW)

    repaired = storage.consume_pairing(
        "pairing-repair",
        replace(
            original,
            device_id="new-proposed-device-id",
            display_name="Renamed Phone",
        ),
        now=NOW,
    )

    assert repaired.device_id == original.device_id
    assert repaired.display_name == "Renamed Phone"
    assert storage.read_cursor(original.device_id).next_event_seq == 1
    assert storage.list_device_sessions(original.device_id) == ("mobile:session-1",)


def test_event_sequence_and_insert_rollback_together(
    storage: MobileRealtimeStorage,
) -> None:
    storage.register_device(_device())
    first = storage.append_durable_event(
        device_id="device-1",
        event_id="event-1",
        envelope_json=_event_json("event-1"),
        created_at=NOW,
    )
    assert first.event_seq == 1

    with closing(sqlite3.connect(storage.db_path)) as db, db:
        db.execute(
            """
            CREATE TRIGGER fail_mobile_cursor_advance
            BEFORE UPDATE OF next_event_seq ON mobile_device_cursors
            BEGIN
                SELECT RAISE(ABORT, 'cursor advance failed');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="cursor advance failed"):
        storage.append_durable_event(
            device_id="device-1",
            event_id="event-2",
            envelope_json=_event_json("event-2"),
            created_at=NOW,
        )

    assert storage.read_cursor("device-1").next_event_seq == 2
    assert storage.count_durable_events("device-1") == 1


def test_same_logical_event_can_be_enqueued_for_each_device(
    storage: MobileRealtimeStorage,
) -> None:
    storage.register_device(_device("device-1"))
    storage.register_device(_device("device-2"))

    for device_id in ("device-1", "device-2"):
        event = storage.append_durable_event(
            device_id=device_id,
            event_id="broadcast-event",
            envelope_json=_event_json("broadcast-event"),
            created_at=NOW,
        )
        assert event.event_seq == 1


def test_ack_advances_and_deletes_in_one_transaction(
    storage: MobileRealtimeStorage,
) -> None:
    storage.register_device(_device())
    for index in range(1, 4):
        storage.append_durable_event(
            device_id="device-1",
            event_id=f"event-{index}",
            envelope_json=_event_json(f"event-{index}"),
            created_at=NOW,
        )
    storage.mark_events_sent("device-1", through_event_seq=2)

    result = storage.acknowledge_durable_events(
        "device-1",
        through_event_seq=1,
    )
    assert result.previous_event_seq == 0
    assert result.acknowledged_event_seq == 1
    assert result.deleted_events == 1
    assert [event.event_seq for event in storage.read_durable_events(
        "device-1", after_event_seq=0, limit=10
    )] == [2, 3]

    duplicate = storage.acknowledge_durable_events(
        "device-1",
        through_event_seq=1,
    )
    assert duplicate.deleted_events == 0
    with pytest.raises(AckRollbackError):
        storage.acknowledge_durable_events("device-1", through_event_seq=0)
    with pytest.raises(AckOverflowError):
        storage.acknowledge_durable_events("device-1", through_event_seq=3)
    with pytest.raises(SentCursorError):
        storage.mark_events_sent("device-1", through_event_seq=4)


def test_ack_delete_failure_rolls_back_cursor_advance(
    storage: MobileRealtimeStorage,
) -> None:
    storage.register_device(_device())
    storage.append_durable_event(
        device_id="device-1",
        event_id="event-1",
        envelope_json=_event_json("event-1"),
        created_at=NOW,
    )
    storage.mark_events_sent("device-1", through_event_seq=1)
    with closing(sqlite3.connect(storage.db_path)) as db, db:
        db.execute(
            """
            CREATE TRIGGER fail_mobile_inbox_delete
            BEFORE DELETE ON mobile_device_inbox
            BEGIN
                SELECT RAISE(ABORT, 'inbox delete failed');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="inbox delete failed"):
        storage.acknowledge_durable_events("device-1", through_event_seq=1)

    assert storage.read_cursor("device-1").acknowledged_event_seq == 0
    assert storage.count_durable_events("device-1") == 1


def test_corrupt_sqlite_event_payload_fails_at_row_boundary(
    storage: MobileRealtimeStorage,
) -> None:
    storage.register_device(_device())
    storage.append_durable_event(
        device_id="device-1",
        event_id="event-1",
        envelope_json=_event_json("event-1"),
        created_at=NOW,
    )
    with closing(sqlite3.connect(storage.db_path)) as db, db:
        db.execute(
            """
            UPDATE mobile_device_inbox
            SET envelope_json = '[]'
            WHERE device_id = 'device-1' AND event_seq = 1
            """
        )

    with pytest.raises(TypeError, match="JSON object"):
        storage.read_durable_events("device-1", after_event_seq=0, limit=10)


def test_command_receipt_replay_conflict_and_retention(
    storage: MobileRealtimeStorage,
) -> None:
    storage.register_device(_device())
    old = NOW - timedelta(days=8)
    receipt, created = storage.reserve_command(
        device_id="device-1",
        command_id="command-1",
        command_type="ping",
        request_hash="hash-1",
        created_at=NOW,
    )
    assert created and receipt.status == "processing"
    completed = storage.complete_command(
        device_id="device-1",
        command_id="command-1",
        reply_type="ping.ok",
        reply_payload_json='{"ok":true}',
        session_id=None,
        turn_id=None,
        completed_at=NOW,
    )
    replay, created = storage.reserve_command(
        device_id="device-1",
        command_id="command-1",
        command_type="ping",
        request_hash="hash-1",
        created_at=NOW + timedelta(days=1),
    )
    assert not created and replay == completed
    with pytest.raises(CommandConflictError):
        storage.reserve_command(
            device_id="device-1",
            command_id="command-1",
            command_type="ping",
            request_hash="different",
            created_at=NOW,
        )

    _, created = storage.reserve_command(
        device_id="device-1",
        command_id="command-2",
        command_type="ping",
        request_hash="hash-2",
        created_at=old,
    )
    assert created
    storage.complete_command(
        device_id="device-1",
        command_id="command-2",
        reply_type="ping.ok",
        reply_payload_json='{"ok":true}',
        session_id=None,
        turn_id=None,
        completed_at=old,
    )
    replacement, created = storage.reserve_command(
        device_id="device-1",
        command_id="command-2",
        command_type="ping",
        request_hash="hash-2",
        created_at=NOW,
    )
    assert created and replacement.status == "processing"


def test_command_receipt_capacity_cleanup_and_insert_are_bounded(
    storage: MobileRealtimeStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage.register_device(_device())
    monkeypatch.setattr(storage_module, "_COMMAND_RECEIPT_MAX_COUNT", 1)
    storage.reserve_command(
        device_id="device-1",
        command_id="command-1",
        command_type="ping",
        request_hash="hash-1",
        created_at=NOW,
    )
    with pytest.raises(CommandReceiptCapacityError):
        storage.reserve_command(
            device_id="device-1",
            command_id="command-2",
            command_type="ping",
            request_hash="hash-2",
            created_at=NOW,
        )
    assert storage.read_cursor("device-1").next_event_seq == 1


def test_command_receipt_completion_rejects_byte_overflow_without_losing_processing(
    storage: MobileRealtimeStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage.register_device(_device())
    storage.reserve_command(
        device_id="device-1",
        command_id="command-1",
        command_type="ping",
        request_hash="hash-1",
        created_at=NOW,
    )
    monkeypatch.setattr(storage_module, "_COMMAND_RECEIPT_MAX_BYTES", 50)
    with pytest.raises(CommandReceiptCapacityError):
        storage.complete_command(
            device_id="device-1",
            command_id="command-1",
            reply_type="ping.ok",
            reply_payload_json='{"result":"' + ("x" * 128) + '"}',
            session_id=None,
            turn_id=None,
            completed_at=NOW,
        )
    row = storage._db.execute(
        "SELECT status FROM mobile_command_receipts "
        "WHERE device_id = ? AND command_id = ?",
        ("device-1", "command-1"),
    ).fetchone()
    assert row["status"] == "processing"

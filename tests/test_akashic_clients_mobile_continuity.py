from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest

from plugins.akashic_clients.mobile_realtime.storage import (
    DeviceRecord,
    MobileRealtimeStorage,
)


NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)


def _device(device_id: str) -> DeviceRecord:
    return DeviceRecord(
        device_id=device_id,
        public_key=f"public:{device_id}",
        display_name=device_id,
        created_at=NOW,
        revoked_at=None,
        capabilities=(),
    )


def _event(event_id: str) -> str:
    return json.dumps(
        {"id": event_id, "type": "message.final", "payload": {"id": event_id}},
        separators=(",", ":"),
        sort_keys=True,
    )


def test_transport_boot_appends_ordered_reset_without_rebasing_state(tmp_path: Path) -> None:
    db_path = tmp_path / "mobile.db"
    storage = MobileRealtimeStorage(db_path)
    try:
        storage.register_device(_device("active"))
        storage.register_device(_device("revoked"))
        storage.revoke_device("revoked", revoked_at=NOW)
        _ = storage.reserve_command(
            device_id="active",
            command_id="command-1",
            command_type="message.send",
            request_hash="request-hash",
            created_at=NOW,
        )

        assert storage.mark_transport_boot("boot-a", created_at=NOW) == ()
        old = storage.append_durable_event(
            device_id="active",
            event_id="event-1",
            envelope_json=_event("event-1"),
            created_at=NOW,
        )
        reset = storage.mark_transport_boot("boot-b", created_at=NOW)

        assert len(reset) == 1
        assert reset[0].device_id == "active"
        assert reset[0].event_seq == old.event_seq + 1
        assert json.loads(reset[0].envelope_json) == {
            "id": reset[0].event_id,
            "payload": {"reason": "transport_continuity_lost"},
            "type": "sync.reset_required",
        }
        assert [item.event_seq for item in storage.read_durable_events(
            "active", after_event_seq=0, limit=10
        )] == [old.event_seq, reset[0].event_seq]
        assert storage.count_durable_events("revoked") == 0
        assert storage.read_command(device_id="active", command_id="command-1") is not None
        assert storage.read_transport_boot_id() == "boot-b"
        assert storage.mark_transport_boot("boot-b", created_at=NOW) == ()
        assert storage.count_durable_events("active") == 2
    finally:
        storage.close()

    reopened = MobileRealtimeStorage(db_path)
    try:
        assert reopened.mark_transport_boot("boot-b", created_at=NOW) == ()
        reset = reopened.mark_transport_boot("boot-c", created_at=NOW)
        assert len(reset) == 1
        assert reset[0].event_seq == 3
        assert [item.event_seq for item in reopened.read_durable_events(
            "active", after_event_seq=0, limit=10
        )] == [1, 2, 3]
    finally:
        reopened.close()


def test_transport_boot_marker_and_reset_share_transaction(tmp_path: Path) -> None:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    try:
        storage.register_device(_device("active"))
        assert storage.mark_transport_boot("boot-a", created_at=NOW) == ()
        with closing(sqlite3.connect(storage.db_path)) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TRIGGER reject_transport_reset
                    BEFORE INSERT ON mobile_device_inbox
                    WHEN json_extract(NEW.envelope_json, '$.type') = 'sync.reset_required'
                    BEGIN
                        SELECT RAISE(ABORT, 'injected reset failure');
                    END
                    """
                )

        with pytest.raises(sqlite3.IntegrityError, match="injected reset failure"):
            storage.mark_transport_boot("boot-b", created_at=NOW)

        assert storage.read_transport_boot_id() == "boot-a"
        assert storage.count_durable_events("active") == 0

        with closing(sqlite3.connect(storage.db_path)) as connection:
            with connection:
                connection.execute("DROP TRIGGER reject_transport_reset")
        reset = storage.mark_transport_boot("boot-b", created_at=NOW)
        assert len(reset) == 1
        assert storage.read_transport_boot_id() == "boot-b"
    finally:
        storage.close()

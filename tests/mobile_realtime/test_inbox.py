from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from infra.mobile_realtime.inbox import DurableInboxManager, InboxResetRequired
from infra.mobile_realtime.storage import DeviceRecord, MobileRealtimeStorage


NOW = datetime(2026, 7, 14, 9, 30, tzinfo=timezone.utc)


def _build_storage(tmp_path: Path) -> MobileRealtimeStorage:
    storage = MobileRealtimeStorage(tmp_path / "mobile.db")
    storage.register_device(
        DeviceRecord(
            device_id="device-1",
            public_key="public-key",
            display_name="Phone",
            created_at=NOW,
            revoked_at=None,
            capabilities=("chat",),
        )
    )
    return storage


def _envelope(event_id: str) -> str:
    return json.dumps(
        {
            "v": 1,
            "kind": "event",
            "type": "message.final",
            "id": event_id,
            "payload": {},
        }
    )


def test_inbox_replays_in_sequence_and_acknowledges_p0(tmp_path: Path) -> None:
    storage = _build_storage(tmp_path)
    try:
        manager = DurableInboxManager(storage, clock=lambda: NOW)
        first = manager.enqueue(
            device_id="device-1",
            event_id="event-1",
            envelope_json=_envelope("event-1"),
        )
        second = manager.enqueue(
            device_id="device-1",
            event_id="event-2",
            envelope_json=_envelope("event-2"),
        )

        replay = manager.replay("device-1", after_event_seq=0, limit=10)
        assert replay.events == (first, second)
        assert replay.cursor.next_event_seq == 3

        manager.mark_sent("device-1", through_event_seq=2)
        ack = manager.acknowledge("device-1", through_event_seq=2)
        assert ack.deleted_events == 2
        assert storage.count_durable_events("device-1") == 0
    finally:
        storage.close()


def test_expired_p0_requires_reset_without_silent_deletion(tmp_path: Path) -> None:
    storage = _build_storage(tmp_path)
    try:
        current_time = NOW
        manager = DurableInboxManager(storage, clock=lambda: current_time)
        manager.enqueue(
            device_id="device-1",
            event_id="event-1",
            envelope_json=_envelope("event-1"),
        )
        current_time = NOW + timedelta(days=8)

        with pytest.raises(InboxResetRequired, match="恢复窗口"):
            manager.replay("device-1", after_event_seq=0, limit=10)
        assert storage.count_durable_events("device-1") == 1
    finally:
        storage.close()


def test_default_retention_allows_exactly_seven_days_then_requires_reset(
    tmp_path: Path,
) -> None:
    storage = _build_storage(tmp_path)
    try:
        current_time = NOW
        manager = DurableInboxManager(storage, clock=lambda: current_time)
        manager.enqueue(
            device_id="device-1",
            event_id="event-1",
            envelope_json=_envelope("event-1"),
        )

        current_time = NOW + timedelta(days=7)
        assert len(manager.replay("device-1", after_event_seq=0, limit=10).events) == 1

        current_time += timedelta(microseconds=1)
        with pytest.raises(InboxResetRequired, match="恢复窗口"):
            manager.replay("device-1", after_event_seq=0, limit=10)
    finally:
        storage.close()


def test_inbox_rejects_naive_clock(tmp_path: Path) -> None:
    storage = _build_storage(tmp_path)
    try:
        manager = DurableInboxManager(
            storage,
            clock=lambda: datetime(2026, 7, 14, 9, 30),
        )
        with pytest.raises(ValueError, match="带时区"):
            manager.enqueue(
                device_id="device-1",
                event_id="event-1",
                envelope_json=_envelope("event-1"),
            )
    finally:
        storage.close()

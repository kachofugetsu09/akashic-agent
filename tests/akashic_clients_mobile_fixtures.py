"""Small fixtures shared by tests of the installed clients plugin."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from plugins.akashic_clients.config import MobileRealtimeConfig
from plugins.akashic_clients.mobile_realtime.protocol import GenericCommand, parse_frame
from plugins.akashic_clients.mobile_realtime.storage import DeviceRecord, MobileRealtimeStorage


class _Runtime:
    """Provide only the event sink required by MobileRealtimeChannel tests."""

    def __init__(self, storage: MobileRealtimeStorage) -> None:
        self.storage = storage
        self.config = MobileRealtimeConfig(max_attachment_mb=50)
        self.events: list[dict[str, object]] = []

    def _recipient_count(self) -> int:
        active_devices = self.storage.list_active_devices()
        if not isinstance(active_devices, tuple):
            raise TypeError("list_active_devices 必须返回 tuple")
        return len(active_devices)

    async def publish_event(self, **event: object) -> int:
        self.events.append(dict(event))
        return self._recipient_count()

    async def publish_connection_control(self, **control: object) -> None:
        self.events.append(dict(control))

    async def refresh_device_capabilities(
        self,
        *,
        device_id: str,
        capabilities: tuple[str, ...],
    ) -> None:
        self.storage.update_device_capabilities(device_id, capabilities)


def _register_device(storage: MobileRealtimeStorage, device_id: str) -> None:
    storage.register_device(
        DeviceRecord(
            device_id=device_id,
            public_key=f"test-public-key:{device_id}",
            display_name=device_id,
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
            capabilities=("stream-v1",),
        )
    )


def _generic_frame(
    *,
    frame_id: str,
    command_type: str,
    session_id: str | None = None,
    turn_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> GenericCommand:
    """Build a strict command envelope through the plugin protocol parser."""

    raw: dict[str, object] = {
        "v": 1,
        "kind": "command",
        "type": command_type,
        "id": frame_id,
        "connection_epoch": 1,
        "payload": payload or {},
    }
    if session_id is not None:
        raw["session_id"] = session_id
    if turn_id is not None:
        raw["turn_id"] = turn_id
    frame = parse_frame(json.dumps(raw))
    if not isinstance(frame, GenericCommand):
        raise TypeError("expected GenericCommand")
    return frame

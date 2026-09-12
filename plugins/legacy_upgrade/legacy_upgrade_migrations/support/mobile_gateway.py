"""Frozen scalar helpers used by historical mobile migration records."""
from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime


def _new_ulid() -> str:
    """Return a migration-only sortable identifier with the old alphabet."""
    return f"{int(datetime.now(UTC).timestamp() * 1000):012x}{secrets.token_hex(10)}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _encode_stored_event(
    *,
    event_id: str,
    event_type: str,
    payload: dict[str, object],
    session_id: str | None = None,
    turn_id: str | None = None,
) -> str:
    """Encode the historical durable envelope without connection state."""

    body: dict[str, object] = {
        "id": event_id,
        "type": event_type,
        "payload": payload,
    }
    if session_id is not None:
        body["session_id"] = session_id
    if turn_id is not None:
        body["turn_id"] = turn_id
    return json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

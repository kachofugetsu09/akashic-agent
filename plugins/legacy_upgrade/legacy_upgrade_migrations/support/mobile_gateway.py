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


def _encode_stored_event(event: object) -> str:
    """Encode one already validated stored event as canonical JSON."""
    if isinstance(event, str):
        return event
    return json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

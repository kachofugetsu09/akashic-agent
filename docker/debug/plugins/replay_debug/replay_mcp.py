#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import threading
from importlib import import_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

_ACK_LOCK = threading.Lock()


def fetch_replay_events(offset: int = 0, limit: int = 50) -> str:
    events = _available_events()
    return json.dumps(events[max(0, offset):max(0, offset) + max(1, limit)], ensure_ascii=False)


def _available_events() -> list[dict[str, Any]]:
    now = _clock_now()
    events_path = _required_path("AKASHIC_REPLAY_EVENTS_FILE")
    acked = _read_acked(_acks_path(events_path))
    available: list[dict[str, Any]] = []
    for event in _read_events(events_path):
        event_id = str(event.get("event_id") or "").strip()
        if not event_id or event_id in acked:
            continue
        available_at = _parse_time(event.get("available_at") or event.get("published_at"))
        if available_at > now:
            continue
        kind = str(event.get("kind") or "content").strip()
        if kind not in {"alert", "content", "context"}:
            continue
        available.append(dict(event))
    return available


def acknowledge_replay_events(
    event_ids: list[str], feedback: str | None = None
) -> str:
    del feedback
    events_path = _required_path("AKASHIC_REPLAY_EVENTS_FILE")
    acks_path = _acks_path(events_path)
    clean_ids = list(dict.fromkeys(str(item).strip() for item in event_ids if str(item).strip()))
    if not clean_ids:
        return json.dumps({"acked": 0})
    with _ACK_LOCK:
        payload = _read_ack_payload(acks_path)
        acked = payload.setdefault("acked", {})
        now = _clock_now().isoformat()
        for event_id in clean_ids:
            acked[event_id] = {"acked_at": now}
        _atomic_write_json(acks_path, payload)
    return json.dumps({"acked": len(clean_ids)})


def _required_path(name: str) -> Path:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return Path(value)


def _clock_now() -> datetime:
    path = _required_path("AKASHIC_REPLAY_CLOCK_FILE")
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid replay clock file: {path}")
    current_time = cast(dict[str, object], payload).get("current_time")
    return _parse_time(current_time)


def _acks_path(events_path: Path) -> Path:
    configured = str(os.environ.get("AKASHIC_REPLAY_ACKS_FILE") or "").strip()
    return Path(configured) if configured else events_path.parent / "acks.json"


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        payload = cast(object, json.loads(text))
        if isinstance(payload, list):
            values = cast(list[object], payload)
        elif isinstance(payload, dict):
            raw_values = cast(dict[str, object], payload).get("events", [])
            values = cast(list[object], raw_values) if isinstance(raw_values, list) else []
        else:
            values = []
        return [dict(cast(dict[str, Any], item)) for item in values if isinstance(item, dict)]
    result: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        item = cast(object, json.loads(line))
        if not isinstance(item, dict):
            raise ValueError(f"replay event must be object: {path}")
        result.append(dict(cast(dict[str, Any], item)))
    return result


def _read_ack_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"acked": {}}
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid replay ack file: {path}")
    typed = cast(dict[str, Any], payload)
    if not isinstance(typed.get("acked"), dict):
        raise ValueError(f"invalid replay ack file: {path}")
    return typed


def _read_acked(path: Path) -> set[str]:
    return set(_read_ack_payload(path)["acked"])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _ = temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _ = temporary.replace(path)


def _parse_time(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("replay event time must include timezone")
    return parsed.astimezone(timezone.utc)


def create_server() -> Any:
    FastMCP = getattr(import_module("mcp.server.fastmcp"), "FastMCP")
    server = FastMCP("replay-debug")
    server.tool()(fetch_replay_events)
    server.tool()(acknowledge_replay_events)
    return server


if __name__ == "__main__":
    create_server().run(transport="stdio")

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.clock import ReplayClock


_PROFILE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


@dataclass(frozen=True)
class ReplayLayout:
    profile_root: Path
    replay_root: Path
    clock_path: Path
    events_path: Path
    outbox_path: Path

    @classmethod
    def for_profile(cls, profile: str) -> "ReplayLayout":
        if not _PROFILE_RE.fullmatch(profile):
            raise ValueError(f"无效 profile 名称: {profile!r}")
        profile_root = Path(__file__).resolve().parent / "profiles" / profile
        replay_root = profile_root / "workspace" / "replay"
        return cls(
            profile_root=profile_root,
            replay_root=replay_root,
            clock_path=replay_root / "clock.json",
            events_path=replay_root / "events.jsonl",
            outbox_path=replay_root / "outbox.jsonl",
        )


def initialize(layout: ReplayLayout, start_at: datetime) -> dict[str, Any]:
    layout.replay_root.mkdir(parents=True, exist_ok=True)
    clock = ReplayClock(layout.clock_path)
    now = clock.set(start_at)
    layout.events_path.touch(exist_ok=True)
    layout.outbox_path.touch(exist_ok=True)
    return {"profile": layout.profile_root.name, "current_time": now.isoformat()}


def append_event(layout: ReplayLayout, event: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_event(event)
    layout.replay_root.mkdir(parents=True, exist_ok=True)
    with layout.events_path.open("a", encoding="utf-8") as handle:
        _ = handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    return normalized


def import_events(layout: ReplayLayout, input_path: Path) -> int:
    count = 0
    for event in _read_event_input(input_path):
        _ = append_event(layout, event)
        count += 1
    return count


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    event_id = str(event.get("event_id") or "").strip()
    kind = str(event.get("kind") or "content").strip()
    source_id = str(event.get("source_id") or "").strip()
    if not event_id or not source_id:
        raise ValueError("event_id 和 source_id 不能为空")
    if kind not in {"alert", "content", "context"}:
        raise ValueError(f"未知事件类型: {kind}")
    available_at = _parse_time(event.get("available_at") or event.get("published_at"))
    raw_published_at = event.get("published_at")
    published_at = _parse_time(raw_published_at) if raw_published_at else None
    return {
        "event_id": event_id,
        "kind": kind,
        "source_id": source_id,
        "source_name": str(event.get("source_name") or source_id),
        "title": str(event.get("title") or ""),
        "content": str(event.get("content") or ""),
        "url": str(event.get("url") or ""),
        "published_at": published_at.isoformat() if published_at else None,
        "first_seen_at": _parse_time(
            event.get("first_seen_at") or available_at
        ).isoformat(),
        "available_at": available_at.isoformat(),
        "preprocess_score": float(event.get("preprocess_score") or 0.0),
        "preprocess_features": (
            event.get("preprocess_features")
            if isinstance(event.get("preprocess_features"), dict)
            else {}
        ),
        "wake_eligible": event.get("wake_eligible") is not False,
        "payload": event.get("payload") if isinstance(event.get("payload"), dict) else {},
    }


def status(layout: ReplayLayout) -> dict[str, Any]:
    clock = ReplayClock(layout.clock_path)
    now = clock.now()
    events = list(_read_jsonl(layout.events_path))
    outbox = list(_read_jsonl(layout.outbox_path))
    available = sum(
        _parse_time(event.get("available_at")) <= now
        for event in events
    )
    return {
        "profile": layout.profile_root.name,
        "current_time": now.isoformat(),
        "events": len(events),
        "available_events": available,
        "future_events": len(events) - available,
        "outbox_messages": len(outbox),
        "latest_outbound": outbox[-1] if outbox else None,
    }


def _read_event_input(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from _read_jsonl(path)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("events", [])
    if not isinstance(items, list):
        raise ValueError("历史事件文件必须是 JSON 数组、events 数组或 JSONL")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("历史事件必须是对象")
        yield item


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL 行必须是对象: {path}")
        yield payload


def _parse_time(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("事件时间不能为空")
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("事件时间必须包含时区")
    return parsed.astimezone(UTC)


def _event_from_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.payload) if args.payload else {}
    return {
        "event_id": args.event_id,
        "kind": args.kind,
        "source_id": args.source_id,
        "source_name": args.source_name,
        "title": args.title,
        "content": args.content,
        "url": args.url,
        "published_at": args.published_at,
        "first_seen_at": args.first_seen_at,
        "available_at": args.available_at,
        "preprocess_score": args.preprocess_score,
        "payload": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="真实 Runtime 的时间回放控制器")
    parser.add_argument("--profile", default="wake-replay")
    commands = parser.add_subparsers(dest="command", required=True)

    init_parser = commands.add_parser("init")
    init_parser.add_argument("--start-at", required=True)

    set_parser = commands.add_parser("set-time")
    set_parser.add_argument("--to", required=True)

    advance_parser = commands.add_parser("advance")
    advance_parser.add_argument("--seconds", type=float, required=True)

    inject_parser = commands.add_parser("inject")
    inject_parser.add_argument("--event-id", required=True)
    inject_parser.add_argument("--kind", choices=("alert", "content", "context"), default="content")
    inject_parser.add_argument("--source-id", required=True)
    inject_parser.add_argument("--source-name")
    inject_parser.add_argument("--title", default="")
    inject_parser.add_argument("--content", default="")
    inject_parser.add_argument("--url", default="")
    inject_parser.add_argument("--published-at", required=True)
    inject_parser.add_argument("--first-seen-at")
    inject_parser.add_argument("--available-at")
    inject_parser.add_argument("--preprocess-score", type=float, default=0.0)
    inject_parser.add_argument("--payload", default="{}")

    import_parser = commands.add_parser("import-events")
    import_parser.add_argument("input", type=Path)

    commands.add_parser("status")
    commands.add_parser("clear-outbox")

    args = parser.parse_args()
    layout = ReplayLayout.for_profile(args.profile)
    if args.command == "init":
        result: Any = initialize(layout, _parse_time(args.start_at))
    elif args.command == "set-time":
        result = {"current_time": ReplayClock(layout.clock_path).set(_parse_time(args.to)).isoformat()}
    elif args.command == "advance":
        current = ReplayClock(layout.clock_path).advance(timedelta(seconds=args.seconds))
        result = {"current_time": current.isoformat()}
    elif args.command == "inject":
        result = append_event(layout, _event_from_args(args))
    elif args.command == "import-events":
        result = {"imported": import_events(layout, args.input)}
    elif args.command == "clear-outbox":
        layout.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        layout.outbox_path.write_text("", encoding="utf-8")
        result = {"cleared": str(layout.outbox_path)}
    else:
        result = status(layout)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

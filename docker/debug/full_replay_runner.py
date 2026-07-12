#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, cast

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.plugins.manifest import load_plugin_manifest, write_plugin_manifest
from core.clock import ReplayClock
from docker.debug.replay_controller import ReplayLayout, append_event, initialize


DEBUG_ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = DEBUG_ROOT / "docker-compose.yml"
ALLOWED_PLUGINS = {"akasha", "default_memory", "replay_debug", "wake_proactive"}


@dataclass(frozen=True)
class FeedItem:
    event_id: str
    source_id: str
    source_name: str
    source_type: str
    title: str
    content: str
    url: str
    author: str
    published_at: datetime | None
    first_seen_at: datetime
    last_seen_at: datetime
    content_hash: str
    published_at_recovered: bool = False


_X_STATUS_RE = re.compile(r"/status/(\d+)")
_X_EPOCH_MS = 1_288_834_974_657
_FRESHNESS_HALF_LIFE_HOURS = 36.0
_MISSING_PUBLICATION_CONFIDENCE = 0.03
_WAKE_ADMISSION_FLOOR = 0.02


def load_feed_union(paths: list[Path], end_at: datetime) -> list[FeedItem]:
    latest: dict[str, FeedItem] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(items)")
            }
            required = {
                "event_id", "source_id", "source_name", "source_type", "title",
                "content", "url", "published_at", "first_seen_at", "last_seen_at",
                "content_hash",
            }
            missing = required - columns
            if missing:
                raise ValueError(f"Feed snapshot 缺少字段 {sorted(missing)}: {path}")
            author_expr = "author" if "author" in columns else "''"
            rows = connection.execute(
                f"""
                SELECT event_id, source_id, source_name, source_type, title,
                       content, url, {author_expr}, published_at, first_seen_at,
                       last_seen_at, content_hash
                FROM items
                WHERE first_seen_at IS NOT NULL
                """
            )
            for row in rows:
                item = _feed_item(row)
                if item.first_seen_at > end_at:
                    continue
                identity = _event_identity(item)
                previous = latest.get(identity)
                if previous is None or item.last_seen_at > previous.last_seen_at:
                    latest[identity] = replace(
                        item,
                        first_seen_at=min(
                            item.first_seen_at,
                            previous.first_seen_at if previous is not None else item.first_seen_at,
                        ),
                    )
                elif item.first_seen_at < previous.first_seen_at:
                    latest[identity] = replace(
                        previous, first_seen_at=item.first_seen_at
                    )
        finally:
            connection.close()
    return sorted(latest.values(), key=lambda item: (item.first_seen_at, item.event_id))


def _feed_item(row: tuple[Any, ...]) -> FeedItem:
    first_seen_at = _parse_time(row[9])
    raw_published_at = _parse_time(row[8]) if row[8] else None
    recovered = raw_published_at is None
    published_at = raw_published_at or _x_published_at(str(row[6] or ""))
    return FeedItem(
        event_id=str(row[0]),
        source_id=str(row[1]),
        source_name=str(row[2]),
        source_type=str(row[3]),
        title=str(row[4] or ""),
        content=str(row[5] or ""),
        url=str(row[6] or ""),
        author=str(row[7] or ""),
        published_at=published_at,
        first_seen_at=first_seen_at,
        last_seen_at=_parse_time(row[10] or first_seen_at),
        content_hash=str(row[11] or ""),
        published_at_recovered=recovered and published_at is not None,
    )


def _x_published_at(url: str) -> datetime | None:
    match = _X_STATUS_RE.search(url)
    if match is None:
        return None
    timestamp_ms = (int(match.group(1)) >> 22) + _X_EPOCH_MS
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)


def _event_identity(item: FeedItem) -> str:
    match = _X_STATUS_RE.search(item.url)
    if match is not None:
        return f"x:{item.source_id}:{match.group(1)}"
    return f"event:{item.event_id}"


def occupied_hour_steps(items: list[FeedItem], end_at: datetime) -> list[tuple[datetime, int]]:
    counts: dict[datetime, int] = defaultdict(int)
    for item in items:
        hour = item.first_seen_at.replace(minute=0, second=0, microsecond=0)
        counts[hour] += 1
    if not counts:
        return []
    target = min(counts) + timedelta(hours=1)
    steps: list[tuple[datetime, int]] = []
    while target <= end_at:
        steps.append((target, counts[target - timedelta(hours=1)]))
        target += timedelta(hours=1)
    return steps


def prepare_profile(
    layout: ReplayLayout,
    *,
    template_profile: str,
    reset: bool,
    sessions_db: Path,
    akasha_db: Path,
    memory_md: Path,
    proactive_context: Path,
) -> None:
    template = ReplayLayout.for_profile(template_profile).profile_root
    if layout.profile_root == template:
        raise ValueError("回放 profile 必须与模板 profile 不同")
    if not (template / "config.toml").is_file():
        raise FileNotFoundError(template / "config.toml")
    if layout.profile_root.exists():
        if not reset:
            raise FileExistsError(f"profile 已存在，使用 --reset-profile 重建: {layout.profile_root}")
        shutil.rmtree(layout.profile_root)

    workspace = layout.profile_root / "workspace"
    _ = workspace.mkdir(parents=True)
    _ = shutil.copy2(template / "config.toml", layout.profile_root / "config.toml")
    copy_context_snapshot(
        workspace,
        sessions_db=sessions_db,
        akasha_db=akasha_db,
        memory_md=memory_md,
        proactive_context=proactive_context,
    )

    source_plugins_home = template / "home" / ".akashic-plugin"
    entries = load_plugin_manifest(source_plugins_home)
    isolated = {plugin_id: plugin_id in ALLOWED_PLUGINS for plugin_id in entries}
    for plugin_id in ALLOWED_PLUGINS:
        isolated[plugin_id] = True
    _ = write_plugin_manifest(
        isolated,
        plugins_home=layout.profile_root / "home" / ".akashic-plugin",
    )
    _patch_config(layout.profile_root / "config.toml")


def copy_context_snapshot(
    workspace: Path,
    *,
    sessions_db: Path,
    akasha_db: Path,
    memory_md: Path,
    proactive_context: Path,
) -> None:
    sources = (sessions_db, akasha_db, memory_md, proactive_context)
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少上下文快照: {missing}")
    memory_root = workspace / "memory"
    _ = memory_root.mkdir(parents=True, exist_ok=True)
    _ = shutil.copy2(sessions_db, workspace / "sessions.db")
    _ = shutil.copy2(akasha_db, memory_root / "akasha.db")
    _ = shutil.copy2(memory_md, memory_root / "MEMORY.md")
    _ = shutil.copy2(proactive_context, workspace / "PROACTIVE_CONTEXT.md")


def _patch_config(path: Path) -> None:
    config = cast(dict[str, Any], tomllib.loads(path.read_text(encoding="utf-8")))
    channels = config.setdefault("channels", {})
    for value in channels.values():
        if isinstance(value, dict) and "enabled" in value:
            value["enabled"] = False
    proactive = config.setdefault("proactive", {})
    proactive["enabled"] = True
    proactive["lifecycle"] = "wake"
    raw_target = proactive.get("target")
    target = cast(dict[str, Any], raw_target) if isinstance(raw_target, dict) else {}
    if not str(target.get("channel") or "").strip():
        raise ValueError("模板 config 缺少 proactive.target.channel")
    trigger = proactive.setdefault("overrides", {}).setdefault("trigger", {})
    trigger.update({"tick_interval_s0": 1, "tick_interval_s1": 1, "tick_jitter": 0.0})
    _ = path.write_text(toml.dumps(config), encoding="utf-8")


def load_score_map(
    path: Path | None,
) -> dict[str, tuple[float, dict[str, Any], dict[str, Any]]]:
    if path is None:
        return {}
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError("score map 必须是 event_id 到 score 的 JSON 对象")
    result: dict[str, tuple[float, dict[str, Any], dict[str, Any]]] = {}
    for event_id, raw in cast(dict[object, object], payload).items():
        if isinstance(raw, (int, float)):
            result[str(event_id)] = (float(raw), {}, {})
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"无效 score map 条目: {event_id}")
        typed_raw = cast(dict[object, object], raw)
        score = typed_raw.get("score")
        if not isinstance(score, (int, float)):
            raise ValueError(f"无效 score map 条目: {event_id}")
        features = typed_raw.get("features")
        if features is not None and not isinstance(features, dict):
            raise ValueError(f"features 必须是对象: {event_id}")
        typed_features = cast(dict[str, Any], features) if isinstance(features, dict) else {}
        metadata = {
            key: typed_raw[key]
            for key in (
                "published_at", "wake_eligible", "freshness_reason",
                "published_at_override",
            )
            if key in typed_raw
        }
        result[str(event_id)] = (float(score), dict(typed_features), metadata)
    return result


def write_replay_events(
    layout: ReplayLayout,
    items: list[FeedItem],
    score_map: dict[
        str, tuple[float, dict[str, Any], dict[str, Any]]
    ] | None = None,
) -> int:
    scores = score_map or {}
    missing = 0
    _ = layout.events_path.write_text("", encoding="utf-8")
    _ = layout.outbox_path.write_text("", encoding="utf-8")
    acks_path = layout.replay_root / "acks.json"
    if acks_path.exists():
        acks_path.unlink()
    for item in items:
        score_entry = scores.get(item.event_id)
        if score_entry is None:
            score, features, metadata = 0.0, {}, {}
            missing += 1
        else:
            score, features, metadata = score_entry
        scored_published_at = (
            _parse_time(metadata["published_at"])
            if metadata.get("published_at")
            else item.published_at
        )
        wake_eligible = metadata.get("wake_eligible")
        if not isinstance(wake_eligible, bool):
            wake_eligible = bool(
                scored_published_at is not None
                and timedelta(0)
                <= item.first_seen_at - scored_published_at
                <= timedelta(hours=72)
            )
        _ = append_event(
            layout,
            {
                "event_id": item.event_id,
                "kind": "content",
                "source_id": item.source_id,
                "source_name": item.source_name,
                "title": item.title,
                "content": item.content,
                "url": item.url,
                "published_at": scored_published_at,
                "first_seen_at": item.first_seen_at,
                "available_at": item.first_seen_at,
                "wake_eligible": wake_eligible,
                "preprocess_score": score,
                "payload": {
                    "author": item.author,
                    "source_type": item.source_type,
                    "content_hash": item.content_hash,
                    "last_seen_at": item.last_seen_at.isoformat(),
                    "published_at_missing": item.published_at is None,
                    "published_at_recovered": item.published_at_recovered,
                    "published_at_override": metadata.get("published_at_override"),
                    "freshness_reason": metadata.get("freshness_reason"),
                    "features": features,
                },
                "preprocess_features": features,
            },
        )
    return missing


def admit_replay_items(
    items: list[FeedItem],
    score_map: dict[str, tuple[float, dict[str, Any], dict[str, Any]]],
) -> list[FeedItem]:
    admitted: list[FeedItem] = []
    for item in items:
        score_entry = score_map.get(item.event_id)
        if score_entry is None:
            continue
        _score, features, metadata = score_entry
        raw_interest = features.get("interest")
        if not isinstance(raw_interest, (int, float)):
            continue
        published_at = (
            _parse_time(metadata["published_at"])
            if metadata.get("published_at")
            else item.published_at
        )
        age_hours = (
            max(0.0, (item.first_seen_at - published_at).total_seconds() / 3600)
            if published_at is not None
            else 0.0
        )
        confidence = 1.0 if published_at is not None else _MISSING_PUBLICATION_CONFIDENCE
        freshness = confidence * math.exp(
            -math.log(2.0) * age_hours / _FRESHNESS_HALF_LIFE_HOURS
        )
        interest = min(0.999, max(0.0, float(raw_interest)))
        if -math.log1p(-interest) * freshness >= _WAKE_ADMISSION_FLOOR:
            admitted.append(item)
    return admitted


def refresh_replay_eligibility(
    layout: ReplayLayout, items: list[FeedItem]
) -> None:
    feed_items = {item.event_id: item for item in items}
    events = [
        event
        for event in _read_jsonl(layout.events_path)
        if str(event.get("event_id") or "") in feed_items
    ]
    for event in events:
        item = feed_items.get(str(event.get("event_id") or ""))
        assert item is not None
        event["wake_eligible"] = bool(
            item.published_at is not None
            and timedelta(0)
            <= item.first_seen_at - item.published_at
            <= timedelta(hours=72)
        )
    temporary = layout.events_path.with_suffix(".jsonl.tmp")
    _ = temporary.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )
    _ = temporary.replace(layout.events_path)


def start_runtime(profile: str, *, build: bool) -> str:
    env = {**os.environ, "AKASHIC_DEBUG_PROFILE": profile}
    env["AKASHIC_REPLAY_CHANNEL"] = _target_channel(
        ReplayLayout.for_profile(profile).profile_root / "config.toml"
    )
    if build:
        _ = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "build", "akashic-debug"],
            cwd=DEBUG_ROOT.parent.parent,
            env=env,
            check=True,
        )
    result = subprocess.run(
        [
            "docker", "compose", "-f", str(COMPOSE_FILE), "run", "-d",
            "--no-deps", "akashic-debug",
        ],
        cwd=DEBUG_ROOT.parent.parent,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[-1]


def _target_channel(path: Path) -> str:
    config = cast(dict[str, Any], tomllib.loads(path.read_text(encoding="utf-8")))
    raw_proactive = config.get("proactive")
    proactive = (
        cast(dict[str, Any], raw_proactive) if isinstance(raw_proactive, dict) else {}
    )
    raw_target = proactive.get("target")
    target = cast(dict[str, Any], raw_target) if isinstance(raw_target, dict) else {}
    channel = str(target.get("channel") or "").strip()
    if not channel:
        raise ValueError("config 缺少 proactive.target.channel")
    return channel


def stop_runtime(container_id: str) -> None:
    _ = subprocess.run(
        ["docker", "stop", container_id],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def runtime_error_summary(container_id: str) -> list[str]:
    result = subprocess.run(
        ["docker", "logs", "--tail", "200", container_id],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = (result.stdout + "\n" + result.stderr).splitlines()
    markers = ("error", "exception", "traceback", "失败", "异常")
    matched = [line for line in lines if any(marker in line.lower() for marker in markers)]
    return matched[-20:]


def wait_until_stable(
    layout: ReplayLayout,
    *,
    target: datetime,
    expected_events: int,
    timeout: float,
    quiet_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    stable_since: float | None = None
    previous_signature: str | None = None
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            latest = observe(layout)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            time.sleep(0.25)
            continue
        incomplete_at_target = int(
            latest["active_incomplete_wake_times"].get(target.isoformat(), 0)
        )
        if incomplete_at_target >= 3:
            raise RuntimeError(
                f"同一模拟时刻已有 {incomplete_at_target} 个未终止 wake: {target.isoformat()}"
            )
        ready = (
            latest["reservoir_total"] >= expected_events
            and latest["unfinished_ticks"] == 0
            and not latest["active_incomplete_wake_times"]
            and target.isoformat() in latest["processed_times"]
        )
        signature = json.dumps(latest["stable_state"], ensure_ascii=False, sort_keys=True)
        now = time.monotonic()
        if ready and signature == previous_signature:
            stable_since = stable_since or now
            if now - stable_since >= quiet_seconds:
                return latest
        else:
            stable_since = None
        previous_signature = signature
        time.sleep(0.25)
    raise TimeoutError(
        f"runtime 未在 {timeout:g}s 内稳定: target={target.isoformat()} state={latest}"
    )


def observe(layout: ReplayLayout) -> dict[str, Any]:
    wake_db = layout.profile_root / "workspace" / "wake_proactive.db"
    proactive_db = layout.profile_root / "workspace" / "proactive.db"
    reservoir_total = 0
    reservoir_status: dict[str, int] = {}
    wakes: list[dict[str, Any]] = []
    hazards: list[dict[str, Any]] = []
    processed_times: set[str] = set()
    incomplete_wake_times: dict[str, int] = {}
    incomplete_seqs: dict[str, list[int]] = {}
    latest_terminal_seq = 0
    if wake_db.exists():
        connection = _open_readonly(wake_db)
        try:
            if _table_exists(connection, "reservoir_events"):
                reservoir_total = int(connection.execute("SELECT count(*) FROM reservoir_events").fetchone()[0])
                reservoir_status = {
                    str(status): int(count)
                    for status, count in connection.execute(
                        "SELECT status, count(*) FROM reservoir_events GROUP BY status"
                    )
                }
            if _table_exists(connection, "wake_runs"):
                wakes = [
                    dict(row) for row in connection.execute(
                        "SELECT rowid AS run_seq, wake_id, now_utc, terminal_action, final_message FROM wake_runs ORDER BY rowid"
                    )
                ]
                processed_times.update(str(row["now_utc"]) for row in wakes)
                for row in wakes:
                    if row["terminal_action"] is None:
                        wake_time = str(row["now_utc"])
                        incomplete_wake_times[wake_time] = (
                            incomplete_wake_times.get(wake_time, 0) + 1
                        )
                        incomplete_seqs.setdefault(wake_time, []).append(
                            int(row["run_seq"])
                        )
                    else:
                        latest_terminal_seq = max(
                            latest_terminal_seq, int(row["run_seq"])
                        )
            if not _table_exists(connection, "wake_runs") and _table_exists(connection, "wake_observations"):
                observations = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT id AS run_seq, wake_id, now_utc, kind,
                               'observe' AS terminal_action,
                               '' AS final_message
                        FROM wake_observations
                        ORDER BY id
                        """
                    )
                ]
                if observations:
                    wakes = observations
                    incomplete_wake_times = {}
                    incomplete_seqs = {}
                    latest_terminal_seq = int(observations[-1]["run_seq"])
                    processed_times.update(
                        str(row["now_utc"]) for row in observations
                    )
            if _table_exists(connection, "hazard_state"):
                hazards = [dict(row) for row in connection.execute("SELECT * FROM hazard_state ORDER BY session_key")]
                processed_times.update(str(row["updated_at"]) for row in hazards)
            if _table_exists(connection, "drift_state"):
                processed_times.update(
                    str(row["updated_at"])
                    for row in connection.execute(
                        "SELECT updated_at FROM drift_state"
                    )
                )
        finally:
            connection.close()

    tick_count = 0
    unfinished_ticks = 0
    latest_tick: dict[str, Any] | None = None
    if proactive_db.exists():
        connection = _open_readonly(proactive_db)
        try:
            if _table_exists(connection, "tick_log"):
                tick_count, unfinished_ticks = map(
                    int,
                    connection.execute(
                        "SELECT count(*), coalesce(sum(CASE WHEN finished_at IS NULL THEN 1 ELSE 0 END), 0) FROM tick_log"
                    ).fetchone(),
                )
                row = connection.execute(
                    "SELECT tick_id, terminal_action, skip_reason, content_count, final_message FROM tick_log ORDER BY id DESC LIMIT 1"
                ).fetchone()
                latest_tick = dict(row) if row is not None else None
        finally:
            connection.close()

    outbox = list(_read_jsonl(layout.outbox_path))
    active_incomplete_wake_times = {
        wake_time: sum(
            seq > latest_terminal_seq for seq in sequences
        )
        for wake_time, sequences in incomplete_seqs.items()
        if any(seq > latest_terminal_seq for seq in sequences)
    }
    stable_state: dict[str, Any] = {
        "reservoir_total": reservoir_total,
        "reservoir_status": reservoir_status,
        "wake_count": len(wakes),
        "incomplete_wake_times": incomplete_wake_times,
        "active_incomplete_wake_times": active_incomplete_wake_times,
        "hazards": hazards,
        "outbox_count": len(outbox),
        "unfinished_ticks": unfinished_ticks,
    }
    return {
        **stable_state,
        "tick_count": tick_count,
        "latest_tick": latest_tick,
        "latest_wake": wakes[-1] if wakes else None,
        "latest_outbox": outbox[-1] if outbox else None,
        "wake_records": wakes,
        "outbox_records": outbox,
        "processed_times": sorted(processed_times),
        "stable_state": stable_state,
    }


def _open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
    connection.row_factory = sqlite3.Row
    _ = connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    return (
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    end_at = _parse_time(args.end_at)
    layout = ReplayLayout.for_profile(args.profile)
    items = load_feed_union(args.feed_db, end_at)
    if args.start_at:
        start_at = _parse_time(args.start_at)
        items = [item for item in items if item.first_seen_at >= start_at]
    source_event_count = len(items)
    score_map = load_score_map(args.score_map)
    items = admit_replay_items(items, score_map)
    if not items:
        raise ValueError("指定时间范围内没有通过衰减准入的 Feed items")
    report_path = layout.replay_root / "full_replay_steps.jsonl"
    completed_steps = 0
    if args.resume_profile:
        if not report_path.is_file():
            raise FileNotFoundError(f"续跑报告不存在: {report_path}")
        completed_steps = max(
            (
                int(record.get("step") or 0)
                for record in _read_jsonl(report_path)
                if record.get("type") != "abort"
            ),
            default=0,
        )
        missing_scores = sum(
            1 for item in items if item.event_id not in score_map
        )
        refresh_replay_eligibility(layout, items)
    else:
        prepare_profile(
            layout,
            template_profile=args.template_profile,
            reset=args.reset_profile,
            sessions_db=args.sessions_db,
            akasha_db=args.akasha_db,
            memory_md=args.memory_md,
            proactive_context=args.proactive_context,
        )
        _ = initialize(layout, items[0].first_seen_at - timedelta(seconds=1))
        missing_scores = write_replay_events(layout, items, score_map)
        _ = report_path.write_text("", encoding="utf-8")
    scored_events = len(items) - missing_scores
    steps = occupied_hour_steps(items, end_at)

    container_id = start_runtime(args.profile, build=args.build)
    cumulative = sum(count for _, count in steps[:completed_steps])
    initial = observe(layout)
    wake_cursor = len(initial["wake_records"])
    outbox_cursor = len(initial["outbox_records"])
    try:
        for index, (target, event_count) in enumerate(
            steps[completed_steps:], completed_steps + 1
        ):
            _ = ReplayClock(layout.clock_path).set(target)
            cumulative += event_count
            snapshot = wait_until_stable(
                layout,
                target=target,
                expected_events=cumulative,
                timeout=args.step_timeout,
                quiet_seconds=args.quiet_seconds,
            )
            record: dict[str, Any] = {
                "step": index,
                "at": target.isoformat(),
                "events_in_hour": event_count,
                "cumulative_events": cumulative,
                "scored_events": scored_events,
                "missing_scores": missing_scores,
                **{
                    key: value
                    for key, value in snapshot.items()
                    if key not in {
                        "processed_times", "stable_state", "wake_records", "outbox_records"
                    }
                },
                "new_wakes": snapshot["wake_records"][wake_cursor:],
                "new_outbox": snapshot["outbox_records"][outbox_cursor:],
            }
            wake_cursor = len(snapshot["wake_records"])
            outbox_cursor = len(snapshot["outbox_records"])
            with report_path.open("a", encoding="utf-8") as handle:
                _ = handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if args.max_steps and index >= args.max_steps:
                break
    except Exception as exc:
        failure: dict[str, Any] = {
            "type": "abort",
            "at": ReplayClock(layout.clock_path).now().isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_errors": runtime_error_summary(container_id),
        }
        with report_path.open("a", encoding="utf-8") as handle:
            _ = handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        raise
    finally:
        if not args.leave_running:
            stop_runtime(container_id)
    return {
        "profile": args.profile,
        "source_events": source_event_count,
        "unique_events": len(items),
        "decayed_before_reservoir": source_event_count - len(items),
        "hours": len(steps),
        "scored_events": scored_events,
        "missing_scores": missing_scores,
        "report": str(report_path),
    }


def _parse_time(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"时间必须包含时区: {value}")
    return parsed.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Feed 全量快照的真实 Runtime 小时回放")
    _ = parser.add_argument("--profile", default="wake-full-replay")
    _ = parser.add_argument("--template-profile", default="wake-history-replay")
    _ = parser.add_argument("--feed-db", type=Path, action="append", required=True)
    _ = parser.add_argument("--start-at")
    _ = parser.add_argument("--end-at", required=True)
    _ = parser.add_argument("--score-map", type=Path)
    _ = parser.add_argument("--sessions-db", type=Path, required=True)
    _ = parser.add_argument("--akasha-db", type=Path, required=True)
    _ = parser.add_argument("--memory-md", type=Path, required=True)
    _ = parser.add_argument("--proactive-context", type=Path, required=True)
    _ = parser.add_argument("--step-timeout", type=float, default=60)
    _ = parser.add_argument("--quiet-seconds", type=float, default=2)
    _ = parser.add_argument("--max-steps", type=int, default=0)
    _ = parser.add_argument("--reset-profile", action="store_true")
    _ = parser.add_argument("--resume-profile", action="store_true")
    _ = parser.add_argument("--build", action="store_true")
    _ = parser.add_argument("--leave-running", action="store_true")
    args = parser.parse_args()
    if args.reset_profile and args.resume_profile:
        parser.error("--reset-profile 与 --resume-profile 不能同时使用")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

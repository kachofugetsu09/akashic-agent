"""只读聚合当前运行版本与宿主自部署任务状态。"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import cast

_DEFAULT_RUNTIME_INFO = Path("/opt/akashic/runtime-info.json")
_STALE_ACTIVE_SECONDS = 75 * 60
_VALID_STATUSES = frozenset({"submitting", "accepted", "running", "active", "failed"})
_IN_PROGRESS_STATUSES = frozenset({"submitting", "accepted", "running"})


def _clean_commit(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip().lower()
    if len(trimmed) == 40 and all(ch in "0123456789abcdef" for ch in trimmed):
        return trimmed
    return trimmed or None


def read_runtime_commit(*, runtime_info_path: Path = _DEFAULT_RUNTIME_INFO) -> str | None:
    try:
        raw = json.loads(runtime_info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = None
    if isinstance(raw, Mapping):
        commit = _clean_commit(cast(Mapping[str, object], raw).get("sourceCommit"))
        if commit is not None:
            return commit
    return _clean_commit(os.environ.get("AKASHIC_RUNTIME_COMMIT"))


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _clean_text(value: object, *, max_len: int) -> str | None:
    if not isinstance(value, str):
        return None
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return None
    summary = lines[0]
    return summary if len(summary) <= max_len else summary[: max_len - 1] + "…"


def _parse_deploy_record(
    path: Path,
    *,
    now: datetime,
) -> tuple[float, dict[str, object]] | None:
    try:
        stat = path.stat()
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, Mapping):
        return None
    record = cast(Mapping[str, object], raw)
    status = record.get("status")
    if not isinstance(status, str) or status not in _VALID_STATUSES:
        return None
    request = record.get("request")
    if not isinstance(request, Mapping):
        return None
    target_commit = _clean_commit(cast(Mapping[str, object], request).get("targetCommit"))
    if target_commit is None:
        return None

    submitted_dt = _parse_iso(record.get("submittedAt"))
    started_dt = _parse_iso(record.get("startedAt"))
    finished_dt = _parse_iso(record.get("finishedAt"))
    anchor_dt = finished_dt or started_dt or submitted_dt
    anchor_ts = anchor_dt.timestamp() if anchor_dt is not None else stat.st_mtime

    phase_raw = record.get("phase")
    if isinstance(phase_raw, str) and phase_raw.strip():
        phase = phase_raw.strip()
    elif status == "running":
        phase = "preparing"
    else:
        phase = status

    stale = False
    if status in _IN_PROGRESS_STATUSES:
        age_seconds = now.timestamp() - anchor_ts
        stale = age_seconds > _STALE_ACTIVE_SECONDS

    result_raw = record.get("result")
    degraded: list[str] = []
    if isinstance(result_raw, Mapping):
        degraded_raw = cast(Mapping[str, object], result_raw).get("degradedExternal")
        if isinstance(degraded_raw, Mapping):
            degraded = sorted(str(key) for key in cast(Mapping[object, object], degraded_raw).keys())

    previous_commit = _clean_commit(record.get("previousCommit"))
    commit_subject = _clean_text(record.get("commitSubject"), max_len=160)
    error_summary = _clean_text(record.get("error"), max_len=240) if status == "failed" else None

    entry: dict[str, object] = {
        "requestId": path.stem,
        "status": status,
        "phase": phase,
        "targetCommit": target_commit,
        "previousCommit": previous_commit,
        "commitSubject": commit_subject,
        "submittedAt": submitted_dt.isoformat() if submitted_dt is not None else None,
        "startedAt": started_dt.isoformat() if started_dt is not None else None,
        "finishedAt": finished_dt.isoformat() if finished_dt is not None else None,
        "recordedAt": datetime.fromtimestamp(anchor_ts, tz=UTC).isoformat(),
        "stale": stale,
        "degradedExternal": degraded,
        "error": error_summary,
    }
    return anchor_ts, entry


def read_deploy_snapshot(
    *,
    started_at: str,
    socket_path: str | None = None,
    runtime_info_path: Path = _DEFAULT_RUNTIME_INFO,
    now: datetime | None = None,
    history_limit: int = 1,
) -> dict[str, object]:
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    commit = read_runtime_commit(runtime_info_path=runtime_info_path)
    resolved_socket = os.environ.get("AKASHIC_HOST_BRIDGE_SOCKET") if socket_path is None else socket_path

    parsed_entries: list[tuple[float, str, dict[str, object]]] = []
    if resolved_socket:
        directory = Path(resolved_socket).parent / "self-deploy"
        try:
            paths = tuple(directory.glob("*.json")) if directory.is_dir() else ()
        except OSError:
            paths = ()
        for item in paths:
            if not item.is_file():
                continue
            parsed = _parse_deploy_record(item, now=current_time)
            if parsed is None:
                continue
            ts, entry = parsed
            parsed_entries.append((ts, item.stem, entry))

    parsed_entries.sort(key=lambda row: (row[0], row[1]), reverse=True)
    limit = max(1, history_limit)
    history = [row[2] for row in parsed_entries[:limit]]
    latest_deploy = history[0] if history else None

    snapshot: dict[str, object] = {
        "runtime": {
            "commit": commit,
            "startedAt": started_at,
        },
        "deploy": latest_deploy,
    }
    if history_limit > 1:
        last_success = next(
            (
                row[2]
                for row in parsed_entries
                if row[2].get("status") == "active"
                and (commit is None or row[2].get("targetCommit") == commit)
            ),
            None,
        ) or next((row[2] for row in parsed_entries if row[2].get("status") == "active"), None)
        snapshot["lastSuccess"] = last_success
        snapshot["history"] = history
    return snapshot

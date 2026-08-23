"""Read retained proactive history without restoring its runtime or effects."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class LegacyProactiveHistory:
    """Project retained Wake, Drift, job, and document history read-only."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def wake_runs(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        path = self.workspace / "wake_proactive.db"
        if not path.is_file():
            return ()
        with closing(_connect(path)) as connection:
            if "wake_runs" not in _tables(connection):
                return ()
            rows = connection.execute(
                "SELECT * FROM wake_runs ORDER BY now_utc DESC LIMIT ?", (limit,)
            ).fetchall()
        return tuple(_decode_wake(dict(row)) for row in rows)

    def drift_runs(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        path = self.workspace / "drift" / "drift.db"
        if not path.is_file():
            return ()
        with closing(_connect(path)) as connection:
            if "runs" not in _tables(connection):
                return ()
            rows = connection.execute(
                "SELECT id, event_id, run_at, skill_name, status, briefing, "
                "message_result FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def job_outcomes(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        path = self.workspace / "runtime" / "plugin-jobs" / "outcomes.sqlite"
        if not path.is_file():
            return ()
        with closing(_connect(path)) as connection:
            if "job_outcomes" not in _tables(connection):
                return ()
            rows = connection.execute(
                "SELECT * FROM job_outcomes ORDER BY created_at DESC, invocation_id "
                "DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(_decode_job(dict(row)) for row in rows)

    def document_manifests(self) -> tuple[dict[str, Any], ...]:
        root = self.workspace / "runtime" / "proactive-documents"
        result: list[dict[str, Any]] = []
        for family in ("intents", "receipts"):
            directory = root / family
            if not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir(), key=lambda item: item.name):
                manifest = entry / "intent.json" if entry.is_dir() else entry
                if manifest.suffix != ".json" and manifest.name != "intent.json":
                    continue
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        f"legacy document manifest is not object: {manifest}"
                    )
                result.append(
                    {"family": family, "name": entry.name, "manifest": payload}
                )
        return tuple(result)

    def snapshot(self, *, limit: int = 100) -> dict[str, object]:
        """Return one dashboard-ready projection with no write-capable objects."""

        return {
            "wake_runs": self.wake_runs(limit=limit),
            "drift_runs": self.drift_runs(limit=limit),
            "job_outcomes": self.job_outcomes(limit=limit),
            "document_manifests": self.document_manifests(),
        }


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    _ = connection.execute("PRAGMA query_only = ON")
    return connection


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _decode_wake(row: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "scratchpad_json",
        "investigations_json",
        "cited_ids_json",
        "display_event_map_json",
        "source_refs_json",
    ):
        if key in row:
            row[key.removesuffix("_json")] = json.loads(row.pop(key) or "null")
    return row


def _decode_job(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.pop("event_payload_json", None)
    row["event_payload"] = None if raw is None else json.loads(raw)
    return row


__all__ = ["LegacyProactiveHistory"]

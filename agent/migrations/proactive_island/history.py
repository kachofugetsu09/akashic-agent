"""Read retained proactive history without restoring its runtime or effects."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from agent.migrations.proactive_island.reader import open_legacy_sqlite

_PROACTIVE_HISTORY_TABLES = (
    "deliveries",
    "session_state",
    "context_only_timestamps",
    "tick_log",
    "tick_step_log",
    "rejection_cooldown",
    "seen_items",
    "semantic_items",
    "kv_state",
)


class LegacyProactiveHistory:
    """Project retained Wake, Drift, job, and document history read-only."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def wake_runs(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        path = self.workspace / "wake_proactive.db"
        if not path.is_file():
            return ()
        with open_legacy_sqlite(path) as connection:
            if "wake_runs" not in _tables(connection):
                return ()
            rows = connection.execute(
                "SELECT * FROM wake_runs ORDER BY now_utc DESC LIMIT ?", (limit,)
            ).fetchall()
        return tuple(_decode_wake(dict(row)) for row in rows)

    def wake_observations(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        path = self.workspace / "wake_proactive.db"
        if not path.is_file():
            return ()
        with open_legacy_sqlite(path) as connection:
            if "wake_observations" not in _tables(connection):
                return ()
            rows = connection.execute(
                "SELECT * FROM wake_observations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return tuple(_decode_wake_observation(dict(row)) for row in rows)

    def wake_hazard_monitor(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        """Project dashboard-only hazard observations, never decision continuity."""

        path = self.workspace / "wake_proactive.db"
        if not path.is_file():
            return ()
        with open_legacy_sqlite(path) as connection:
            if "hazard_monitor" not in _tables(connection):
                return ()
            rows = connection.execute(
                "SELECT * FROM hazard_monitor ORDER BY evaluated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def drift_runs(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        path = self.workspace / "drift" / "drift.db"
        if not path.is_file():
            return ()
        with open_legacy_sqlite(path) as connection:
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
        with open_legacy_sqlite(path) as connection:
            if "job_outcomes" not in _tables(connection):
                return ()
            rows = connection.execute(
                "SELECT * FROM job_outcomes ORDER BY created_at DESC, invocation_id "
                "DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(_decode_job(dict(row)) for row in rows)

    def proactive_tables(
        self, *, limit: int = 100
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        """Project every known default-proactive table without owning its rows."""

        path = self.workspace / "proactive.db"
        if not path.is_file():
            return {}
        result: dict[str, tuple[dict[str, Any], ...]] = {}
        with open_legacy_sqlite(path) as connection:
            tables = _tables(connection)
            for table in _PROACTIVE_HISTORY_TABLES:
                if table not in tables:
                    continue
                rows = connection.execute(
                    f'SELECT * FROM "{table}" ORDER BY rowid DESC LIMIT ?',
                    (limit,),
                ).fetchall()
                result[table] = tuple(_decode_sqlite(dict(row)) for row in rows)
        return result

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
            "proactive_tables": self.proactive_tables(limit=limit),
            "wake_runs": self.wake_runs(limit=limit),
            "wake_observations": self.wake_observations(limit=limit),
            "wake_hazard_monitor": self.wake_hazard_monitor(limit=limit),
            "drift_runs": self.drift_runs(limit=limit),
            "job_outcomes": self.job_outcomes(limit=limit),
            "document_manifests": self.document_manifests(),
        }


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
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


def _decode_wake_observation(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("trigger_json", "candidates_json", "llm_input_json"):
        if key in row:
            row[key.removesuffix("_json")] = json.loads(row.pop(key))
    return row


def _decode_job(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.pop("event_payload_json", None)
    row["event_payload"] = None if raw is None else json.loads(raw)
    return row


def _decode_sqlite(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"sqlite_blob_hex": value.hex()} if isinstance(value, bytes) else value
        for key, value in row.items()
    }


__all__ = ["LegacyProactiveHistory"]

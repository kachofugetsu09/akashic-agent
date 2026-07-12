from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from plugins.wake_proactive.state import WakeStateStore


class WakeDashboardReader:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def count(self) -> int:
        row = self._db().execute("SELECT count(*) FROM wake_runs").fetchone()
        return int(row[0])

    def page(self, page: int, page_size: int) -> list[dict[str, Any]]:
        rows = self._db().execute(
            "SELECT * FROM wake_runs ORDER BY now_utc DESC LIMIT ? OFFSET ?",
            (page_size, (page - 1) * page_size),
        ).fetchall()
        return [self._decode(dict(row)) for row in rows]

    def get(self, wake_id: str) -> dict[str, Any] | None:
        row = self._db().execute(
            "SELECT * FROM wake_runs WHERE wake_id = ?", (wake_id,)
        ).fetchone()
        if row is None:
            return None
        item = self._decode(dict(row))
        observations = self._db().execute(
            """
            SELECT kind, now_utc, trigger_json, candidates_json, llm_input_json
            FROM wake_observations
            WHERE wake_id = ?
            ORDER BY id
            """,
            (wake_id,),
        ).fetchall()
        item["observations"] = [
            {
                "kind": observation["kind"],
                "now_utc": observation["now_utc"],
                "trigger": json.loads(observation["trigger_json"]),
                "candidates": json.loads(observation["candidates_json"]),
                "llm_input": json.loads(observation["llm_input_json"]),
            }
            for observation in observations
        ]
        return item

    def meter(self) -> dict[str, Any]:
        monitor = self._db().execute(
            "SELECT * FROM hazard_monitor ORDER BY evaluated_at DESC LIMIT 1"
        ).fetchone()
        state = self._db().execute(
            "SELECT * FROM hazard_state ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        unread = self._db().execute(
            "SELECT count(*) FROM reservoir_events WHERE kind = 'content' AND status = 'unread'"
        ).fetchone()
        latest_run = self._db().execute(
            "SELECT terminal_action, now_utc FROM wake_runs ORDER BY now_utc DESC LIMIT 1"
        ).fetchone()
        if monitor is not None:
            result = dict(monitor)
        elif state is not None:
            result = {
                "session_key": state["session_key"],
                "hazard_before": state["hazard"],
                "hazard_after": state["hazard"],
                "preference_pressure": 0.0,
                "threshold": state["threshold"],
                "evidence": 0.0,
                "rate": 0.0,
                "driver_item_id": "",
                "candidate_count": 0,
                "should_wake": 0,
                "evaluated_at": state["updated_at"],
            }
        else:
            result = {
                "session_key": "",
                "hazard_before": 0.0,
                "hazard_after": 0.0,
                "preference_pressure": 0.0,
                "threshold": 0.0,
                "evidence": 0.0,
                "rate": 0.0,
                "driver_item_id": "",
                "candidate_count": 0,
                "should_wake": 0,
                "evaluated_at": None,
            }
        result["unread_count"] = int(unread[0]) if unread is not None else 0
        result["last_action"] = latest_run["terminal_action"] if latest_run else None
        result["last_action_at"] = latest_run["now_utc"] if latest_run else None
        return result

    @staticmethod
    def _decode(item: dict[str, Any]) -> dict[str, Any]:
        for key in (
            "scratchpad_json", "investigations_json", "cited_ids_json",
            "display_event_map_json", "source_refs_json",
        ):
            item[key.removesuffix("_json")] = json.loads(item.pop(key) or "null")
        return item


def register(app: FastAPI, plugin_dir: Path, workspace: Path) -> WakeDashboardReader:
    WakeStateStore(workspace / "wake_proactive.db").close()
    reader = WakeDashboardReader(workspace / "wake_proactive.db")

    @app.get("/api/dashboard/wake-proactive/runs")
    def runs(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        return {"items": reader.page(page, page_size), "total": reader.count()}

    @app.get("/api/dashboard/wake-proactive/meter")
    def meter() -> dict[str, Any]:
        return reader.meter()

    @app.get("/api/dashboard/wake-proactive/runs/{wake_id}")
    def run(wake_id: str) -> dict[str, Any]:
        item = reader.get(wake_id)
        if item is None:
            raise HTTPException(status_code=404, detail="wake run not found")
        return item

    return reader

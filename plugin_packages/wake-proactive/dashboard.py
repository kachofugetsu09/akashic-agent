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
        return self._decode(dict(row)) if row else None

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

    @app.get("/api/dashboard/wake-proactive/runs/{wake_id}")
    def run(wake_id: str) -> dict[str, Any]:
        item = reader.get(wake_id)
        if item is None:
            raise HTTPException(status_code=404, detail="wake run not found")
        return item

    return reader

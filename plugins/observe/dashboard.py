from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
import sqlite3
import threading

from fastapi import FastAPI

# Observe monitoring dashboard: aggregates the agent-loop telemetry written to
# observe.db (turns table) into Grafana-style metrics — token & KV cache usage,
# ReAct iteration health, and error aggregation. Read-only.

# Range presets -> lookback hours (None = all history).
_RANGES: dict[str, int | None] = {
    "24h": 24,
    "7d": 24 * 7,
    "30d": 24 * 30,
    "all": None,
}


# Resolve a range token to (cutoff_iso, bucket_len). bucket_len is the substring
# length of the ISO ts used to group time buckets: 13 = hour (YYYY-MM-DDTHH),
# 10 = day (YYYY-MM-DD).
def _resolve_range(range_token: str) -> tuple[str | None, int]:
    hours = _RANGES.get(range_token, 24)
    bucket_len = 13 if (hours is not None and hours <= 24) else 10
    if hours is None:
        return None, bucket_len
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return cutoff.isoformat(), bucket_len


class ObserveDashboardReader:
    def __init__(self, workspace: Path) -> None:
        self.db_path = workspace / "observe" / "observe.db"
        self._lock = threading.RLock()

    # Aggregate the metric-card figures over the selected window.
    def get_overview(self, range_token: str) -> dict[str, Any]:
        cutoff, _ = _resolve_range(range_token)
        if not self.db_path.exists():
            return _empty_overview(range_token)
        where, params = _agent_window(cutoff)
        with self._lock, _connect(self.db_path) as db:
            row = db.execute(
                f"""
                SELECT
                    COUNT(*) AS turns,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                    COALESCE(SUM(COALESCE(react_input_sum_tokens, prompt_tokens, 0)), 0) AS input_tokens,
                    COALESCE(SUM(react_cache_prompt_tokens), 0) AS cache_prompt_tokens,
                    COALESCE(SUM(react_cache_hit_tokens), 0) AS cache_hit_tokens,
                    AVG(react_iteration_count) AS avg_iteration,
                    MAX(react_iteration_count) AS max_iteration,
                    MAX(ts) AS last_ts
                FROM turns
                WHERE {where}
                """,
                params,
            ).fetchone()
        return _overview_from_row(row, range_token)

    # Bucketed time series for the trend charts.
    def get_timeseries(self, range_token: str) -> dict[str, Any]:
        cutoff, bucket_len = _resolve_range(range_token)
        if not self.db_path.exists():
            return {"range": range_token, "bucket": _bucket_name(bucket_len), "points": []}
        where, params = _agent_window(cutoff)
        with self._lock, _connect(self.db_path) as db:
            rows = db.execute(
                f"""
                SELECT
                    substr(ts, 1, ?) AS bucket,
                    COUNT(*) AS turns,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                    COALESCE(SUM(COALESCE(react_input_sum_tokens, prompt_tokens, 0)), 0) AS input_tokens,
                    COALESCE(SUM(react_cache_prompt_tokens), 0) AS cache_prompt_tokens,
                    COALESCE(SUM(react_cache_hit_tokens), 0) AS cache_hit_tokens,
                    AVG(react_iteration_count) AS avg_iteration
                FROM turns
                WHERE {where}
                GROUP BY bucket
                ORDER BY bucket ASC
                """,
                (bucket_len, *params),
            ).fetchall()
        return {
            "range": range_token,
            "bucket": _bucket_name(bucket_len),
            "points": [_point_from_row(r) for r in rows],
        }

    # Error rows plus a top-N aggregation by normalized error signature.
    def get_errors(self, range_token: str, *, page: int, page_size: int) -> dict[str, Any]:
        cutoff, _ = _resolve_range(range_token)
        if not self.db_path.exists():
            return {"range": range_token, "items": [], "total": 0, "page": 1, "page_size": page_size, "groups": []}
        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 100))
        offset = (safe_page - 1) * safe_size
        where, params = _agent_window(cutoff)
        err_where = f"{where} AND error IS NOT NULL"
        with self._lock, _connect(self.db_path) as db:
            total = int(
                (db.execute(f"SELECT COUNT(*) AS c FROM turns WHERE {err_where}", params).fetchone() or {"c": 0})["c"]
                or 0
            )
            rows = db.execute(
                f"""
                SELECT id, ts, session_key, user_msg, error
                FROM turns
                WHERE {err_where}
                ORDER BY ts DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, safe_size, offset),
            ).fetchall()
            group_rows = db.execute(
                f"""
                SELECT error, COUNT(*) AS count, MAX(ts) AS last_ts
                FROM turns
                WHERE {err_where}
                GROUP BY substr(error, 1, 80)
                ORDER BY count DESC, last_ts DESC
                LIMIT 8
                """,
                params,
            ).fetchall()
        return {
            "range": range_token,
            "items": [_error_row(r) for r in rows],
            "total": total,
            "page": safe_page,
            "page_size": safe_size,
            "groups": [_error_group(r) for r in group_rows],
        }


def register(app: FastAPI, plugin_dir: Path, workspace: Path) -> None:
    reader = ObserveDashboardReader(workspace)

    @app.get("/api/dashboard/observe/overview")
    def observe_overview(range: str = "24h") -> dict[str, Any]:
        return reader.get_overview(range)

    @app.get("/api/dashboard/observe/timeseries")
    def observe_timeseries(range: str = "24h") -> dict[str, Any]:
        return reader.get_timeseries(range)

    @app.get("/api/dashboard/observe/errors")
    def observe_errors(range: str = "24h", page: int = 1, page_size: int = 25) -> dict[str, Any]:
        return reader.get_errors(range, page=page, page_size=page_size)


# Build the shared WHERE clause: agent turns, optionally bounded by cutoff.
def _agent_window(cutoff: str | None) -> tuple[str, tuple[Any, ...]]:
    if cutoff is None:
        return "source = 'agent'", ()
    return "source = 'agent' AND ts >= ?", (cutoff,)


def _bucket_name(bucket_len: int) -> str:
    return "hour" if bucket_len == 13 else "day"


def _rate(hit: int, total: int) -> float | None:
    return (hit / total) if total > 0 else None


def _overview_from_row(row: sqlite3.Row | None, range_token: str) -> dict[str, Any]:
    if row is None:
        return _empty_overview(range_token)
    turns = int(row["turns"] or 0)
    errors = int(row["errors"] or 0)
    cache_prompt = int(row["cache_prompt_tokens"] or 0)
    cache_hit = int(row["cache_hit_tokens"] or 0)
    return {
        "range": range_token,
        "turns": turns,
        "errors": errors,
        "error_rate": _rate(errors, turns),
        "input_tokens": int(row["input_tokens"] or 0),
        "cache_prompt_tokens": cache_prompt,
        "cache_hit_tokens": cache_hit,
        "cache_hit_rate": _rate(cache_hit, cache_prompt),
        "avg_iteration": float(row["avg_iteration"]) if row["avg_iteration"] is not None else None,
        "max_iteration": int(row["max_iteration"] or 0),
        "last_ts": row["last_ts"],
    }


def _empty_overview(range_token: str) -> dict[str, Any]:
    return {
        "range": range_token,
        "turns": 0,
        "errors": 0,
        "error_rate": None,
        "input_tokens": 0,
        "cache_prompt_tokens": 0,
        "cache_hit_tokens": 0,
        "cache_hit_rate": None,
        "avg_iteration": None,
        "max_iteration": 0,
        "last_ts": None,
    }


def _point_from_row(row: sqlite3.Row) -> dict[str, Any]:
    cache_prompt = int(row["cache_prompt_tokens"] or 0)
    cache_hit = int(row["cache_hit_tokens"] or 0)
    return {
        "bucket": row["bucket"],
        "turns": int(row["turns"] or 0),
        "errors": int(row["errors"] or 0),
        "input_tokens": int(row["input_tokens"] or 0),
        "cache_hit_rate": _rate(cache_hit, cache_prompt),
        "avg_iteration": float(row["avg_iteration"]) if row["avg_iteration"] is not None else None,
    }


def _error_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "ts": row["ts"],
        "session_key": row["session_key"],
        "user_preview": _preview(row["user_msg"], 80),
        "error": _preview(row["error"], 200),
    }


def _error_group(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "signature": _preview(row["error"], 80),
        "count": int(row["count"] or 0),
        "last_ts": row["last_ts"],
    }


def _preview(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

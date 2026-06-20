from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
import json
import sqlite3
import threading

from fastapi import FastAPI

from plugins.observe.db import open_db

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


    # ── 全局错误（global_errors 表）─────────────────────────────────

    # KPI + 排障台头部：总数、错误种类、新类型、爆发类型、最近时间、整体 spark。
    def get_global_overview(self, range_token: str) -> dict[str, Any]:
        cutoff, _ = _resolve_range(range_token)
        groups = self._global_groups(cutoff)
        total = sum(g["count"] for g in groups)
        last_ts = max((g["last_ts"] for g in groups), default=None)
        spark = _merge_buckets(groups)
        return {
            "range": range_token,
            "total": total,
            "types": len(groups),
            "new_types": sum(1 for g in groups if g["is_new"]),
            "spiking_types": sum(1 for g in groups if g["is_spiking"]),
            "last_ts": last_ts,
            "spark": [p["value"] for p in spark],
        }

    # 排障台左栏：按指纹聚合的群组，可按 facet 分组、按 q 过滤。
    def get_global_list(self, range_token: str, *, facet: str, q: str) -> dict[str, Any]:
        cutoff, _ = _resolve_range(range_token)
        groups = self._global_groups(cutoff, include_ignored=False)
        if q:
            needle = q.lower()
            groups = [
                g for g in groups
                if needle in g["error_type"].lower()
                or needle in g["message"].lower()
                or needle in g["logger_name"].lower()
            ]
        groups.sort(key=lambda g: g["count"], reverse=True)
        sections = _facet_sections(groups, facet)
        for g in groups:
            g.pop("_buckets", None)
        return {
            "range": range_token,
            "facet": facet,
            "total": sum(g["count"] for g in groups),
            "sections": sections,
        }

    # 排障台右栏详情：完整 message、趋势、变体（同类型其他指纹）、现场 occurrences。
    def get_global_detail(self, fingerprint: str, range_token: str) -> dict[str, Any]:
        if not self.db_path.exists():
            return {}
        with self._lock, _connect(self.db_path) as db:
            rows = db.execute(
                "SELECT * FROM global_errors WHERE fingerprint = ? ORDER BY bucket ASC",
                (fingerprint,),
            ).fetchall()
            if not rows:
                return {}
            agg = _aggregate_fingerprint(rows)
            siblings = db.execute(
                """
                SELECT fingerprint, traceback_text, SUM(count) AS count, MAX(last_ts) AS last_ts
                FROM global_errors WHERE error_type = ? GROUP BY fingerprint ORDER BY count DESC
                """,
                (agg["error_type"],),
            ).fetchall()
            occurrences = self._global_occurrences(db, agg["session_keys"])
        agg["trend"] = [{"bucket": b, "count": c} for b, c in sorted(agg["_buckets"].items())]
        agg["variants"] = [
            {
                "fingerprint": s["fingerprint"],
                "count": int(s["count"] or 0),
                "traceback_text": s["traceback_text"] or "",
            }
            for s in siblings
        ]
        agg["occurrences"] = occurrences
        agg.pop("_buckets", None)
        return agg

    def set_global_status(self, fingerprint: str, status: str) -> dict[str, Any]:
        if status not in ("active", "acknowledged", "ignored") or not self.db_path.exists():
            return {"ok": False}
        with self._lock, _connect(self.db_path) as db:
            with db:
                db.execute(
                    "UPDATE global_errors SET status = ? WHERE fingerprint = ?",
                    (status, fingerprint),
                )
        return {"ok": True, "fingerprint": fingerprint, "status": status}

    # 取窗口内 global_errors 全部行，按指纹在 Python 侧聚合成群组列表。
    def _global_groups(
        self, cutoff: str | None, *, include_ignored: bool = True
    ) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        where = "1=1" if cutoff is None else "last_ts >= ?"
        params: tuple[Any, ...] = () if cutoff is None else (cutoff,)
        with self._lock, _connect(self.db_path) as db:
            rows = db.execute(
                f"SELECT * FROM global_errors WHERE {where} ORDER BY fingerprint, bucket ASC",
                params,
            ).fetchall()
        by_fp: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_fp.setdefault(row["fingerprint"], []).append(row)
        groups = [_aggregate_fingerprint(rows) for rows in by_fp.values()]
        if not include_ignored:
            groups = [g for g in groups if g["status"] != "ignored"]
        return groups

    # 现场：对每个 session_key 反查 turns 最近一条，取 user_msg 作触发上下文。
    def _global_occurrences(
        self, db: sqlite3.Connection, session_keys: list[str]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key in session_keys[:8]:
            row = db.execute(
                "SELECT ts, user_msg FROM turns WHERE session_key = ? ORDER BY ts DESC LIMIT 1",
                (key,),
            ).fetchone()
            out.append({
                "session_key": key,
                "ts": row["ts"] if row else None,
                "user_preview": _preview(row["user_msg"], 80) if row else "",
            })
        return out


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

    @app.get("/api/dashboard/observe/global_errors/overview")
    def global_errors_overview(range: str = "24h") -> dict[str, Any]:
        return reader.get_global_overview(range)

    @app.get("/api/dashboard/observe/global_errors")
    def global_errors_list(range: str = "24h", facet: str = "type", q: str = "") -> dict[str, Any]:
        return reader.get_global_list(range, facet=facet, q=q)

    @app.get("/api/dashboard/observe/global_errors/{fingerprint}")
    def global_errors_detail(fingerprint: str, range: str = "7d") -> dict[str, Any]:
        return reader.get_global_detail(fingerprint, range)

    @app.post("/api/dashboard/observe/global_errors/{fingerprint}/status")
    def global_errors_status(fingerprint: str, value: str = "acknowledged") -> dict[str, Any]:
        return reader.set_global_status(fingerprint, value)


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


# ── 全局错误聚合辅助 ──────────────────────────────────────────────────────────


def _channel_of(session_keys: list[str]) -> str:
    for key in session_keys:
        head = key.split(":", 1)[0] if ":" in key else ""
        if head:
            return head
    return "—"


# 把同一指纹的多个小时桶行聚合成一个群组 dict（含派生的 spark / is_new / is_spiking）。
def _aggregate_fingerprint(rows: list[sqlite3.Row]) -> dict[str, Any]:
    rep = max(rows, key=lambda r: str(r["last_ts"] or ""))
    buckets: dict[str, int] = {}
    sessions: list[str] = []
    for row in rows:
        buckets[row["bucket"]] = buckets.get(row["bucket"], 0) + int(row["count"] or 0)
        for key in _parse_keys(row["session_keys"]):
            if key not in sessions and len(sessions) < 20:
                sessions.append(key)
    count = sum(buckets.values())
    first_ts = min(str(r["first_ts"] or "") for r in rows)
    last_ts = max(str(r["last_ts"] or "") for r in rows)
    spark = [buckets[b] for b in sorted(buckets)]
    return {
        "fingerprint": rep["fingerprint"],
        "error_type": rep["error_type"] or "Error",
        "logger_name": rep["logger_name"] or "",
        "source": rep["source"] or "log",
        "level": rep["level"] or "ERROR",
        "status": rep["status"] or "active",
        "message": _preview(rep["message"], 200),
        "traceback_text": rep["traceback_text"] or "",
        "count": count,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "session_keys": sessions,
        "sessions": len(sessions),
        "channel": _channel_of(sessions),
        "is_new": _is_new(first_ts),
        "is_spiking": _is_spiking(spark),
        "_buckets": buckets,
    }


def _parse_keys(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(str(raw))
    except (ValueError, TypeError):
        return []
    return [str(x) for x in data] if isinstance(data, list) else []


# NEW = 指纹首次出现在最近 24h 内。
def _is_new(first_ts: str) -> bool:
    if not first_ts:
        return False
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    return first_ts >= cutoff


# 爆发 = 最近一个桶的计数显著高于此前桶的均值。
def _is_spiking(spark: list[int]) -> bool:
    if len(spark) < 3:
        return False
    last = spark[-1]
    prev = spark[:-1]
    avg = sum(prev) / len(prev) if prev else 0
    return last >= 3 and last >= 2 * max(avg, 1)


# 按 facet 把群组切成带小标题的 section；type 走单段平铺。
def _facet_sections(groups: list[dict[str, Any]], facet: str) -> list[dict[str, Any]]:
    if facet not in ("source", "channel"):
        return [{"key": "all", "label": "", "count": sum(g["count"] for g in groups), "items": groups}]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for g in groups:
        buckets.setdefault(str(g[facet]), []).append(g)
    sections = [
        {
            "key": key,
            "label": _SOURCE_LABEL.get(key, key) if facet == "source" else key,
            "count": sum(g["count"] for g in items),
            "items": items,
        }
        for key, items in buckets.items()
    ]
    sections.sort(key=lambda s: s["count"], reverse=True)
    return sections


# 合并多个群组的小时桶 → 整体 spark 点序列。
def _merge_buckets(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, int] = {}
    for g in groups:
        for bucket, count in g.get("_buckets", {}).items():
            merged[bucket] = merged.get(bucket, 0) + count
    return [{"bucket": b, "value": merged[b]} for b in sorted(merged)]


_SOURCE_LABEL: dict[str, str] = {
    "log": "主动日志",
    "uncaught": "未捕获异常",
    "asyncio": "asyncio 任务",
    "thread": "子线程",
}


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = open_db(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

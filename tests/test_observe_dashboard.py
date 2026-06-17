from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from plugins.observe.db import open_db
from plugins.observe.dashboard import ObserveDashboardReader


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _insert_turn(conn: Any, **fields: Any) -> None:
    cols = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    conn.execute(
        f"INSERT INTO turns ({cols}) VALUES ({placeholders})",
        tuple(fields.values()),
    )


def _seed(workspace: Path) -> None:
    db_path = workspace / "observe" / "observe.db"
    conn = open_db(db_path)
    now = datetime.now(timezone.utc)
    # Two recent agent turns: one healthy with cache, one errored.
    _insert_turn(
        conn,
        ts=_iso(now - timedelta(hours=1)),
        source="agent",
        session_key="telegram:1",
        user_msg="hello",
        react_input_sum_tokens=1000,
        react_cache_prompt_tokens=800,
        react_cache_hit_tokens=600,
        react_iteration_count=2,
    )
    _insert_turn(
        conn,
        ts=_iso(now - timedelta(hours=2)),
        source="agent",
        session_key="cli:local",
        user_msg="boom",
        react_input_sum_tokens=500,
        react_iteration_count=5,
        error="ValueError: bad provider response",
    )
    # An old turn outside the 24h window — must be excluded from 24h aggregates.
    _insert_turn(
        conn,
        ts=_iso(now - timedelta(days=5)),
        source="agent",
        session_key="telegram:1",
        user_msg="ancient",
        react_input_sum_tokens=9999,
        react_iteration_count=1,
    )
    # A non-agent row — must never be counted.
    _insert_turn(
        conn,
        ts=_iso(now - timedelta(hours=1)),
        source="proactive",
        session_key="telegram:1",
        react_input_sum_tokens=123,
    )
    conn.commit()
    conn.close()


def test_overview_aggregates_within_window(tmp_path) -> None:
    _seed(tmp_path)
    reader = ObserveDashboardReader(tmp_path)
    ov = reader.get_overview("24h")
    # Only the two recent agent turns count.
    assert ov["turns"] == 2
    assert ov["errors"] == 1
    assert ov["error_rate"] == 0.5
    assert ov["input_tokens"] == 1500
    assert ov["cache_prompt_tokens"] == 800
    assert ov["cache_hit_tokens"] == 600
    assert ov["cache_hit_rate"] == 0.75
    assert ov["avg_iteration"] == 3.5
    assert ov["max_iteration"] == 5


def test_overview_all_range_includes_old(tmp_path) -> None:
    _seed(tmp_path)
    reader = ObserveDashboardReader(tmp_path)
    ov = reader.get_overview("all")
    # All three agent turns (recent two + ancient one); proactive still excluded.
    assert ov["turns"] == 3
    assert ov["input_tokens"] == 1500 + 9999


def test_timeseries_buckets_by_hour(tmp_path) -> None:
    _seed(tmp_path)
    reader = ObserveDashboardReader(tmp_path)
    ts = reader.get_timeseries("24h")
    assert ts["bucket"] == "hour"
    # Two recent turns fall in two distinct hour buckets.
    assert len(ts["points"]) == 2
    assert sum(p["turns"] for p in ts["points"]) == 2
    assert sum(p["errors"] for p in ts["points"]) == 1


def test_errors_list_and_groups(tmp_path) -> None:
    _seed(tmp_path)
    reader = ObserveDashboardReader(tmp_path)
    res = reader.get_errors("24h", page=1, page_size=25)
    assert res["total"] == 1
    assert len(res["items"]) == 1
    assert res["items"][0]["error"].startswith("ValueError")
    assert res["items"][0]["session_key"] == "cli:local"
    assert len(res["groups"]) == 1
    assert res["groups"][0]["count"] == 1


def test_missing_db_returns_empty(tmp_path) -> None:
    reader = ObserveDashboardReader(tmp_path)
    ov = reader.get_overview("24h")
    assert ov["turns"] == 0
    assert ov["cache_hit_rate"] is None
    assert reader.get_timeseries("24h")["points"] == []
    assert reader.get_errors("24h", page=1, page_size=25)["total"] == 0

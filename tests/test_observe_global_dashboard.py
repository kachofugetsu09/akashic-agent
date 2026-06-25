import importlib
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import cast

_db = importlib.import_module("plugins.observe.db")
_writer = importlib.import_module("plugins.observe.writer")
_dash = importlib.import_module("plugins.observe.dashboard")
_events = importlib.import_module("plugins.observe.events")

open_db = cast(Callable[[Path], sqlite3.Connection], getattr(_db, "open_db"))
_write_global_error = getattr(_writer, "_write_global_error")
GlobalErrorTrace = getattr(_events, "GlobalErrorTrace")
ObserveDashboardReader = getattr(_dash, "ObserveDashboardReader")


def _trace(fp: str, bucket: str, *, count: int, etype: str, keys, frame_msg="boom"):
    return GlobalErrorTrace(
        fingerprint=fp,
        bucket=bucket,
        source="log",
        logger_name="agent.looping.core",
        error_type=etype,
        message=frame_msg,
        traceback_text=f"Traceback...\n{etype}: {frame_msg}",
        level="ERROR",
        first_ts=f"{bucket}:00:00+00:00",
        last_ts=f"{bucket}:30:00+00:00",
        count=count,
        session_keys=keys,
    )


def _seed(tmp_path: Path) -> Path:
    ws = tmp_path
    db_path = ws / "observe" / "observe.db"
    conn = open_db(db_path)
    try:
        _write_global_error(conn, _trace("fpA", "2026-06-18T10", count=5, etype="TimeoutError", keys=["telegram:1"]))
        _write_global_error(conn, _trace("fpA", "2026-06-18T11", count=8, etype="TimeoutError", keys=["telegram:2"]))
        _write_global_error(conn, _trace("fpB", "2026-06-18T11", count=2, etype="KeyError", keys=["qq:9"]))
        with conn:
            conn.execute(
                "INSERT INTO turns (ts, source, session_key, user_msg, llm_output) VALUES (?, 'agent', ?, ?, '')",
                ("2026-06-18T11:31:00+00:00", "telegram:1", "帮我查天气"),
            )
    finally:
        conn.close()
    return ws


def test_global_overview(tmp_path):
    reader = ObserveDashboardReader(_seed(tmp_path))
    ov = reader.get_global_overview("all")
    assert ov["total"] == 15
    assert ov["types"] == 2
    assert sum(ov["spark"]) == 15


def test_global_list_type_facet(tmp_path):
    reader = ObserveDashboardReader(_seed(tmp_path))
    data = reader.get_global_list("all", facet="type", q="")
    assert len(data["sections"]) == 1
    items = data["sections"][0]["items"]
    assert items[0]["error_type"] == "TimeoutError"
    assert items[0]["count"] == 13
    assert items[0]["sessions"] == 2
    assert items[0]["spark"] == [5, 8]
    assert "_buckets" not in items[0]


def test_global_list_q_filter(tmp_path):
    reader = ObserveDashboardReader(_seed(tmp_path))
    data = reader.get_global_list("all", facet="type", q="keyerror")
    items = data["sections"][0]["items"]
    assert len(items) == 1
    assert items[0]["error_type"] == "KeyError"


def test_global_detail_with_occurrence_join(tmp_path):
    reader = ObserveDashboardReader(_seed(tmp_path))
    detail = reader.get_global_detail("fpA", "all")
    assert detail["error_type"] == "TimeoutError"
    assert detail["count"] == 13
    assert len(detail["trend"]) == 2
    occ = {o["session_key"]: o for o in detail["occurrences"]}
    assert occ["telegram:1"]["user_preview"] == "帮我查天气"


def test_global_status_ignore_hides_from_list(tmp_path):
    reader = ObserveDashboardReader(_seed(tmp_path))
    assert reader.set_global_status("fpB", "ignored")["ok"] is True
    data = reader.get_global_list("all", facet="type", q="")
    types = {i["error_type"] for i in data["sections"][0]["items"]}
    assert "KeyError" not in types
    assert "TimeoutError" in types

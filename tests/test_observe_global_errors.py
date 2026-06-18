import asyncio
import importlib
import json
import logging
import sqlite3
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

_db = importlib.import_module("plugins.observe.db")
_events = importlib.import_module("plugins.observe.events")
_writer = importlib.import_module("plugins.observe.collector")
_collector = importlib.import_module("plugins.observe.collector")
_writermod = importlib.import_module("plugins.observe.writer")

open_db = cast(Callable[[Path], sqlite3.Connection], getattr(_db, "open_db"))
GlobalErrorTrace = getattr(_events, "GlobalErrorTrace")
GlobalErrorCollector = getattr(_collector, "GlobalErrorCollector")
current_session_key = getattr(_collector, "current_session_key")
_fingerprint = getattr(_collector, "_fingerprint")
_write_global_error = getattr(_writermod, "_write_global_error")


class _RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[object] = []

    def emit(self, event: object) -> None:
        self.events.append(event)


def _trace(**kw) -> object:
    base = dict(
        fingerprint="fp1",
        bucket="2026-06-18T14",
        source="log",
        logger_name="agent.looping.core",
        error_type="TimeoutError",
        message="timed out",
        traceback_text="Traceback...",
        level="ERROR",
        first_ts="2026-06-18T14:00:00+00:00",
        last_ts="2026-06-18T14:00:00+00:00",
        count=1,
        session_keys=[],
    )
    base.update(kw)
    return GlobalErrorTrace(**base)


# ── 指纹 ──────────────────────────────────────────


def test_fingerprint_is_stable_across_numbers():
    a = _fingerprint("TimeoutError", "timed out after 60.0s id=12345", "agent/core.py:10")
    b = _fingerprint("TimeoutError", "timed out after 42.5s id=99887", "agent/core.py:10")
    assert a == b


def test_fingerprint_differs_by_frame_and_type():
    a = _fingerprint("TimeoutError", "x", "agent/core.py:10")
    b = _fingerprint("TimeoutError", "x", "agent/other.py:10")
    c = _fingerprint("KeyError", "x", "agent/core.py:10")
    assert len({a, b, c}) == 3


# ── writer UPSERT ─────────────────────────────────


def test_write_global_error_upserts_count_and_session_keys(tmp_path):
    conn = open_db(tmp_path / "observe.db")
    try:
        _write_global_error(conn, _trace(count=2, session_keys=["tg:1"]))
        _write_global_error(
            conn,
            _trace(
                count=3,
                last_ts="2026-06-18T14:30:00+00:00",
                session_keys=["tg:1", "tg:2"],
            ),
        )
        row = conn.execute(
            "select count, last_ts, session_keys from global_errors where fingerprint='fp1' and bucket='2026-06-18T14'"
        ).fetchone()
        n_rows = conn.execute("select count(*) from global_errors").fetchone()[0]
    finally:
        conn.close()

    assert n_rows == 1
    assert row[0] == 5
    assert row[1] == "2026-06-18T14:30:00+00:00"
    assert set(json.loads(row[2])) == {"tg:1", "tg:2"}


def test_write_global_error_separate_bucket_is_new_row(tmp_path):
    conn = open_db(tmp_path / "observe.db")
    try:
        _write_global_error(conn, _trace(bucket="2026-06-18T14"))
        _write_global_error(conn, _trace(bucket="2026-06-18T15"))
        n_rows = conn.execute("select count(*) from global_errors").fetchone()[0]
    finally:
        conn.close()
    assert n_rows == 2


# ── collector 去重计数 + flush ─────────────────────


def test_collector_dedups_and_counts():
    emitter = _RecordingEmitter()
    col = GlobalErrorCollector(emitter)
    for _ in range(5):
        col.capture(
            source="log",
            logger_name="agent.looping.core",
            error_type="TimeoutError",
            message="timed out after 60s",
            traceback_text="tb",
            level="ERROR",
            top_frame="agent/looping/core.py:188",
            session_key="tg:1",
        )
    col._flush()
    assert len(emitter.events) == 1
    ev = emitter.events[0]
    assert ev.count == 5
    assert ev.session_keys == ["tg:1"]


def test_collector_flush_clears_so_next_flush_emits_delta():
    emitter = _RecordingEmitter()
    col = GlobalErrorCollector(emitter)
    col.capture(source="log", logger_name="x", error_type="E", message="m",
                traceback_text="t", level="ERROR", top_frame="a:1", session_key=None)
    col._flush()
    col._flush()  # 无新增 → 不再 emit
    assert len(emitter.events) == 1


# ── 钩子安装/还原 + 防自噬 ──────────────────────────


@pytest.mark.asyncio
async def test_install_captures_logger_error_and_skips_observe():
    emitter = _RecordingEmitter()
    col = GlobalErrorCollector(emitter)
    prev_excepthook = sys.excepthook
    prev_threadhook = threading.excepthook
    col.install()
    try:
        logging.getLogger("agent.looping.core").error("boom")
        logging.getLogger("observe.writer").error("self failure")  # 应被跳过
        col._flush()
    finally:
        await col.uninstall()

    assert sys.excepthook is prev_excepthook
    assert threading.excepthook is prev_threadhook
    loggers = {ev.logger_name for ev in emitter.events}
    assert "agent.looping.core" in loggers
    assert "observe.writer" not in loggers


@pytest.mark.asyncio
async def test_install_captures_exception_with_traceback():
    emitter = _RecordingEmitter()
    col = GlobalErrorCollector(emitter)
    col.install()
    try:
        try:
            raise ValueError("bad value 42")
        except ValueError:
            logging.getLogger("agent.x").exception("caught")
        col._flush()
    finally:
        await col.uninstall()

    assert len(emitter.events) == 1
    ev = emitter.events[0]
    assert ev.error_type == "ValueError"
    assert "ValueError" in ev.traceback_text

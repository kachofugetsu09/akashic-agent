import asyncio
from contextlib import suppress
import importlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

_observe_db = importlib.import_module("plugins.00_observe.db")
_observe_events = importlib.import_module("plugins.00_observe.events")
_observe_migration = importlib.import_module("plugins.00_observe.migrate_legacy_rag")
_observe_retention = importlib.import_module("plugins.00_observe.retention")
_observe_writer = importlib.import_module("plugins.00_observe.writer")

open_db = cast(Callable[[Path], sqlite3.Connection], getattr(_observe_db, "open_db"))
RagHitLog = getattr(_observe_events, "RagHitLog")
RagQueryLog = getattr(_observe_events, "RagQueryLog")
TurnTrace = getattr(_observe_events, "TurnTrace")
migrate_legacy_rag_tables = getattr(_observe_migration, "migrate_legacy_rag_tables")
_run_cleanup = cast(Callable[[Path], None], getattr(_observe_retention, "_run_cleanup"))
_write_turn = getattr(_observe_writer, "_write_turn")
TraceWriter = getattr(_observe_writer, "TraceWriter")


def test_write_turn_persists_raw_output_and_meme_fields(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = open_db(db_path)
    try:
        _write_turn(
            conn,
            TurnTrace(
                source="agent",
                session_key="telegram:1",
                user_msg="我好喜欢你",
                llm_output="我也喜欢你。",
                raw_llm_output="我也喜欢你。 <meme:shy>",
                meme_tag="shy",
                meme_media_count=1,
            ),
            "2026-03-27T00:00:00+00:00",
        )
        row = conn.execute(
            """
            select llm_output, raw_llm_output, meme_tag, meme_media_count
            from turns
            where session_key = ?
            """,
            ("telegram:1",),
        ).fetchone()
    finally:
        conn.close()

    assert row[0] == "我也喜欢你。"
    assert row[1] == "我也喜欢你。 <meme:shy>"
    assert row[2] == "shy"
    assert row[3] == 1


def test_write_turn_persists_context_budget_fields(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = open_db(db_path)
    try:
        _write_turn(
            conn,
            TurnTrace(
                source="agent",
                session_key="telegram:1",
                user_msg="你好",
                llm_output="收到",
                history_window=40,
                history_messages=27,
                history_chars=18234,
                history_tokens=6078,
                prompt_tokens=6607,
                next_turn_baseline_tokens=12685,
            ),
            "2026-04-12T00:00:00+00:00",
        )
        row = conn.execute(
            """
            select history_window, history_messages, history_chars,
                   history_tokens, prompt_tokens, next_turn_baseline_tokens
            from turns
            where session_key = ?
            """,
            ("telegram:1",),
        ).fetchone()
    finally:
        conn.close()

    assert row[0] == 40
    assert row[1] == 27
    assert row[2] == 18234
    assert row[3] == 6078
    assert row[4] == 6607
    assert row[5] == 12685


def test_write_turn_persists_react_budget_fields(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = open_db(db_path)
    try:
        _write_turn(
            conn,
            TurnTrace(
                source="agent",
                session_key="telegram:1",
                user_msg="你好",
                llm_output="收到",
                react_iteration_count=3,
                react_input_sum_tokens=42100,
                react_input_peak_tokens=18800,
                react_final_input_tokens=17500,
                react_cache_prompt_tokens=32000,
                react_cache_hit_tokens=18000,
            ),
            "2026-04-12T00:00:00+00:00",
        )
        row = conn.execute(
            """
            select react_iteration_count, react_input_sum_tokens,
                   react_input_peak_tokens, react_final_input_tokens,
                   react_cache_prompt_tokens, react_cache_hit_tokens
            from turns
            where session_key = ?
            """,
            ("telegram:1",),
        ).fetchone()
    finally:
        conn.close()

    assert row[0] == 3
    assert row[1] == 42100
    assert row[2] == 18800
    assert row[3] == 17500
    assert row[4] == 32000
    assert row[5] == 18000


def test_open_db_creates_react_budget_columns(tmp_path):
    conn = open_db(tmp_path / "observe.db")
    try:
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(turns)").fetchall()
        }
    finally:
        conn.close()

    assert "react_iteration_count" in cols
    assert "react_input_sum_tokens" in cols
    assert "react_input_peak_tokens" in cols
    assert "react_final_input_tokens" in cols
    assert "react_cache_prompt_tokens" in cols
    assert "react_cache_hit_tokens" in cols


@pytest.mark.asyncio
async def test_trace_writer_drain_waits_for_rag_query(tmp_path):
    db_path = tmp_path / "observe.db"
    writer = TraceWriter(db_path)
    task = asyncio.create_task(writer.run())
    row = None
    try:
        writer.emit(
            RagQueryLog(
                caller="passive",
                session_key="telegram:1",
                query="改写问题",
                orig_query="原问题",
                aux_queries=[],
                hits=[
                    RagHitLog(
                        item_id="m1",
                        memory_type="event",
                        score=0.9,
                        summary="记忆",
                        injected=True,
                    )
                ],
                injected_count=1,
                route_decision="RETRIEVE",
            )
        )
        await writer.drain()
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                """
                select caller, session_key, query, orig_query, injected_count,
                       route_decision, hits_json
                from rag_queries
                """
            ).fetchone()
        finally:
            conn.close()
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert row is not None
    assert row[0] == "passive"
    assert row[1] == "telegram:1"
    assert row[2] == "改写问题"
    assert row[3] == "原问题"
    assert row[4] == 1
    assert row[5] == "RETRIEVE"
    assert '"id": "m1"' in row[6]


def test_open_db_does_not_create_legacy_rag_tables(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = open_db(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "select name from sqlite_master where type = 'table'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert "rag_queries" in tables
    assert "rag_events" not in tables
    assert "rag_items" not in tables


def test_open_db_removes_legacy_proactive_observe_data(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.executescript(
                """
                create table turns (
                    id integer primary key autoincrement,
                    ts text not null,
                    source text not null,
                    session_key text not null,
                    user_msg text,
                    llm_output text not null default '',
                    error text
                );
                create table proactive_decisions (
                    id integer primary key autoincrement,
                    tick_id text unique,
                    ts text not null,
                    session_key text not null,
                    stage text not null
                );
                insert into turns(ts, source, session_key, user_msg, llm_output)
                values('2026-04-01T00:00:00+00:00', 'agent', 'cli:1', 'hi', 'ok');
                insert into turns(ts, source, session_key, user_msg, llm_output)
                values('2026-04-01T00:01:00+00:00', 'proactive', 'cli:1', '', 'push');
                insert into proactive_decisions(tick_id, ts, session_key, stage)
                values('tick-1', '2026-04-01T00:01:00+00:00', 'cli:1', 'gate');
                """
            )
    finally:
        conn.close()

    conn = open_db(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "select name from sqlite_master where type = 'table'"
            ).fetchall()
        }
        rows = conn.execute("select source, llm_output from turns").fetchall()
    finally:
        conn.close()

    assert "proactive_decisions" not in tables
    assert rows == [("agent", "ok")]


def test_migrate_legacy_rag_tables_moves_events_into_rag_queries(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.executescript(
                """
                create table rag_events (
                    id integer primary key autoincrement,
                    ts text not null,
                    source text not null,
                    session_key text not null,
                    original_query text not null,
                    query text not null,
                    route_decision text,
                    hyde_hypothesis text,
                    error text
                );
                create table rag_items (
                    id integer primary key autoincrement,
                    rag_event_id integer not null references rag_events (id),
                    item_id text not null,
                    memory_type text not null,
                    score real not null,
                    summary text not null,
                    retrieval_path text not null,
                    injected integer not null default 0
                );
                """
            )
            event_id = conn.execute(
                """
                insert into rag_events (
                    ts, source, session_key, original_query, query,
                    route_decision, hyde_hypothesis
                ) values (
                    '2026-04-01T00:00:00+00:00', 'agent', 'cli:1',
                    '原问题', '改写问题', 'RETRIEVE', '假想答案'
                )
                """
            ).lastrowid
            conn.execute(
                """
                insert into rag_items (
                    rag_event_id, item_id, memory_type, score, summary,
                    retrieval_path, injected
                ) values (?, 'm1', 'event', 0.8, '旧记忆', 'history_raw', 1)
                """,
                (event_id,),
            )
    finally:
        conn.close()

    result = migrate_legacy_rag_tables(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            select ts, caller, session_key, query, orig_query,
                   aux_queries, hits_json, injected_count, route_decision
            from rag_queries
            """
        ).fetchone()
        tables = {
            r[0]
            for r in conn.execute(
                "select name from sqlite_master where type = 'table'"
            ).fetchall()
        }
    finally:
        conn.close()

    assert result.migrated_events == 1
    assert result.migrated_hits == 1
    assert row[0] == "2026-04-01T00:00:00+00:00"
    assert row[1] == "passive"
    assert row[2] == "cli:1"
    assert row[3] == "改写问题"
    assert row[4] == "原问题"
    assert json.loads(row[5]) == ["假想答案"]
    assert json.loads(row[6]) == [
        {
            "id": "m1",
            "type": "event",
            "score": 0.8,
            "summary": "旧记忆",
            "injected": True,
        }
    ]
    assert row[7] == 1
    assert row[8] == "RETRIEVE"
    assert "rag_events" not in tables
    assert "rag_items" not in tables


def test_migrate_legacy_rag_tables_is_noop_without_legacy_tables(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = open_db(db_path)
    conn.close()

    result = migrate_legacy_rag_tables(db_path)

    assert result.migrated_events == 0
    assert result.migrated_hits == 0
    assert result.dropped_tables == ()


def test_retention_cleans_rag_queries(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = open_db(db_path)
    try:
        with conn:
            conn.execute(
                """
                insert into rag_queries (
                    ts, caller, session_key, query
                ) values (
                    datetime('now', '-91 days'), 'passive', 'cli:1', '旧问题'
                )
                """
            )
            conn.execute(
                """
                insert into rag_queries (
                    ts, caller, session_key, query, error
                ) values (
                    datetime('now', '-91 days'), 'passive', 'cli:1', '错误问题', 'failed'
                )
                """
            )
    finally:
        conn.close()

    _run_cleanup(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "select query from rag_queries order by query"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [("错误问题",)]

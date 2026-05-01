import asyncio
from contextlib import suppress
import sqlite3

import pytest

from core.observe.db import open_db
from core.observe.events import ProactiveDecisionTrace, RagHitLog, RagQueryLog, TurnTrace
from core.observe.retention import _run_cleanup
from core.observe.writer import _write_proactive_decision, _write_turn
from core.observe.writer import TraceWriter


def test_write_proactive_decision_backfills_legacy_columns_for_gate_and_sense(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = open_db(db_path)
    try:
        # 1. 使用新阶段名写入一条 proactive trace。
        _write_proactive_decision(
            conn,
            ProactiveDecisionTrace(
                tick_id="tick-1",
                session_key="telegram:1",
                stage="gate_and_sense",
                stage_result_json='{"sleep_state":"awake","pre_score":0.4}',
            ),
            "2026-03-18T00:00:00+00:00",
        )

        # 2. 校验旧读侧依赖的列仍能拿到同一份 JSON。
        row = conn.execute(
            """
            select stage, gate_result_json, sense_result_json, pre_score_result_json
            from proactive_decisions
            where tick_id = ?
            """,
            ("tick-1",),
        ).fetchone()
    finally:
        conn.close()

    assert row[0] == "gate_and_sense"
    assert row[1] == '{"sleep_state":"awake","pre_score":0.4}'
    assert row[2] == '{"sleep_state":"awake","pre_score":0.4}'
    assert row[3] == '{"sleep_state":"awake","pre_score":0.4}'


def test_write_proactive_decision_backfills_legacy_columns_for_evaluate_and_judge(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = open_db(db_path)
    try:
        # 1. evaluate 兼容旧 fetch_filter/score 列。
        _write_proactive_decision(
            conn,
            ProactiveDecisionTrace(
                tick_id="tick-2",
                session_key="telegram:1",
                stage="evaluate",
                stage_result_json='{"base_score":0.7,"draw_score":0.6}',
            ),
            "2026-03-18T00:00:01+00:00",
        )
        evaluate_row = conn.execute(
            """
            select stage, fetch_filter_result_json, score_result_json
            from proactive_decisions
            where tick_id = ?
            """,
            ("tick-2",),
        ).fetchone()

        # 2. judge_and_send 兼容旧 decide/act 列。
        _write_proactive_decision(
            conn,
            ProactiveDecisionTrace(
                tick_id="tick-3",
                session_key="telegram:1",
                stage="judge_and_send",
                stage_result_json='{"reason_code":"sent_ready"}',
            ),
            "2026-03-18T00:00:02+00:00",
        )
        judge_row = conn.execute(
            """
            select stage, decide_result_json, act_result_json
            from proactive_decisions
            where tick_id = ?
            """,
            ("tick-3",),
        ).fetchone()
    finally:
        conn.close()

    assert evaluate_row[0] == "evaluate"
    assert evaluate_row[1] == '{"base_score":0.7,"draw_score":0.6}'
    assert evaluate_row[2] == '{"base_score":0.7,"draw_score":0.6}'
    assert judge_row[0] == "judge_and_send"
    assert judge_row[1] == '{"reason_code":"sent_ready"}'
    assert judge_row[2] == '{"reason_code":"sent_ready"}'


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


def test_retention_cleans_legacy_rag_events_and_orphan_items(tmp_path):
    db_path = tmp_path / "observe.db"
    conn = open_db(db_path)
    try:
        with conn:
            old_event = conn.execute(
                """
                insert into rag_events (
                    ts, source, session_key, original_query, query
                ) values (
                    datetime('now', '-91 days'), 'agent', 'cli:1', '旧问题', '旧问题'
                )
                """
            ).lastrowid
            kept_event = conn.execute(
                """
                insert into rag_events (
                    ts, source, session_key, original_query, query, error
                ) values (
                    datetime('now', '-91 days'), 'agent', 'cli:1', '错误问题', '错误问题', 'failed'
                )
                """
            ).lastrowid
            conn.execute(
                """
                insert into rag_items (
                    rag_event_id, item_id, memory_type, score, summary, retrieval_path
                ) values (?, 'old-item', 'event', 0.8, '旧记忆', 'history_raw')
                """,
                (old_event,),
            )
            conn.execute(
                """
                insert into rag_items (
                    rag_event_id, item_id, memory_type, score, summary, retrieval_path
                ) values (?, 'kept-item', 'event', 0.8, '保留记忆', 'history_raw')
                """,
                (kept_event,),
            )
    finally:
        conn.close()

    _run_cleanup(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        event_count = conn.execute("select count(*) from rag_events").fetchone()[0]
        item_ids = [
            row[0]
            for row in conn.execute(
                "select item_id from rag_items order by item_id"
            ).fetchall()
        ]
    finally:
        conn.close()

    assert event_count == 1
    assert item_ids == ["kept-item"]


def test_retention_initializes_missing_rag_queries_for_old_db(tmp_path):
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
                    llm_output text not null default '',
                    error text
                );
                create table rag_events (
                    id integer primary key autoincrement,
                    ts text not null,
                    source text not null,
                    session_key text not null,
                    original_query text not null,
                    query text not null,
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
                create table proactive_decisions (
                    id integer primary key autoincrement,
                    ts text not null,
                    session_key text not null,
                    stage text not null,
                    error text
                );
                insert into rag_events (
                    ts, source, session_key, original_query, query
                ) values (
                    datetime('now', '-91 days'), 'agent', 'cli:1', '旧问题', '旧问题'
                );
                """
            )
    finally:
        conn.close()

    _run_cleanup(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        rag_query_cols = {
            row[1]
            for row in conn.execute("pragma table_info(rag_queries)").fetchall()
        }
        legacy_count = conn.execute("select count(*) from rag_events").fetchone()[0]
    finally:
        conn.close()

    assert "hits_json" in rag_query_cols
    assert legacy_count == 0

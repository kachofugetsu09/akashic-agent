import sqlite3

from core.observe.db import open_db
from core.observe.events import ProactiveDecisionTrace, TurnTrace
from core.observe.writer import _write_proactive_decision, _write_turn


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
            ),
            "2026-04-12T00:00:00+00:00",
        )
        row = conn.execute(
            """
            select react_iteration_count, react_input_sum_tokens,
                   react_input_peak_tokens, react_final_input_tokens
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

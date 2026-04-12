"""SQLite 连接管理与 schema 初始化。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# schema 与 schema/observe.sql 保持同步，在代码里内嵌一份避免运行时文件依赖
_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    session_key TEXT    NOT NULL,
    user_msg    TEXT,
    llm_output  TEXT    NOT NULL DEFAULT '',
    raw_llm_output TEXT,
    meme_tag    TEXT,
    meme_media_count INTEGER,
    tool_calls  TEXT,                       -- JSON: [{name, args, result}]（每次 tool 调用）
    tool_chain_json TEXT,                   -- JSON: [{text, calls:[{name,args,result}]}] 完整迭代链路
    history_window INTEGER,
    history_messages INTEGER,
    history_chars INTEGER,
    history_tokens INTEGER,
    prompt_tokens INTEGER,
    next_turn_baseline_tokens INTEGER,
    react_iteration_count INTEGER,
    react_input_sum_tokens INTEGER,
    react_input_peak_tokens INTEGER,
    react_final_input_tokens INTEGER,
    error       TEXT                        -- NULL = 正常
);
CREATE INDEX IF NOT EXISTS ix_turns_sk_ts  ON turns (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_turns_source ON turns (source, ts);

CREATE TABLE IF NOT EXISTS rag_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL,
    source              TEXT    NOT NULL,
    session_key         TEXT    NOT NULL,
    tick_id             TEXT,           -- proactive: 关联 proactive_decisions.tick_id
    original_query      TEXT    NOT NULL,
    query               TEXT    NOT NULL,
    gate_type           TEXT,
    route_decision      TEXT,
    route_latency_ms    INTEGER,
    hyde_hypothesis     TEXT,
    history_scope_mode  TEXT,
    history_gate_reason TEXT,
    injected_block      TEXT,
    preference_block    TEXT,
    preference_query    TEXT,
    sufficiency_check_json TEXT,
    fallback_reason     TEXT,
    error               TEXT
);
CREATE INDEX IF NOT EXISTS ix_re_sk_ts   ON rag_events (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_re_source  ON rag_events (source, ts);

CREATE TABLE IF NOT EXISTS rag_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    rag_event_id    INTEGER NOT NULL REFERENCES rag_events (id),
    item_id         TEXT    NOT NULL,
    memory_type     TEXT    NOT NULL,
    score           REAL    NOT NULL,
    summary         TEXT    NOT NULL,
    happened_at     TEXT,
    extra_json      TEXT,
    retrieval_path  TEXT    NOT NULL,
    injected        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_ri_event ON rag_items (rag_event_id);
CREATE INDEX IF NOT EXISTS ix_ri_item  ON rag_items (item_id);

-- ─────────────────────────────────────────────
-- 4. memory_writes  post-response 记忆写入记录
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_writes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    session_key     TEXT    NOT NULL,
    source_ref      TEXT,
    action          TEXT    NOT NULL,   -- 'write' | 'supersede'
    memory_type     TEXT,               -- write 时填写
    item_id         TEXT,               -- write: 'new:xxx' or 'reinforced:xxx'
    summary         TEXT,               -- write 时填写
    superseded_ids  TEXT,               -- supersede: JSON 数组
    error           TEXT
);
CREATE INDEX IF NOT EXISTS ix_mw_sk_ts ON memory_writes (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_mw_action ON memory_writes (action, ts);

CREATE TABLE IF NOT EXISTS proactive_decisions (
    id                                INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_id                           TEXT    UNIQUE,
    ts                                TEXT    NOT NULL,
    updated_ts                        TEXT    NOT NULL,
    session_key                       TEXT    NOT NULL,
    stage                             TEXT    NOT NULL,
    reason_code                       TEXT,
    should_send                       INTEGER,
    action                            TEXT,
    gate_reason                       TEXT,
    pre_score                         REAL,
    base_score                        REAL,
    draw_score                        REAL,
    decision_score                    REAL,
    send_threshold                    REAL,
    interruptibility                  REAL,
    candidate_count                   INTEGER,
    candidate_item_ids                TEXT,
    sleep_state                       TEXT,
    sleep_prob                        REAL,
    sleep_available                   INTEGER,
    sleep_data_lag_min                INTEGER,
    user_replied_after_last_proactive INTEGER,
    proactive_sent_24h                INTEGER,
    fresh_items_24h                   INTEGER,
    delivery_key                      TEXT,
    is_delivery_duplicate             INTEGER,
    is_message_duplicate              INTEGER,
    delivery_attempted                INTEGER,
    delivery_result                   TEXT,
    reasoning_preview                 TEXT,
    reasoning                         TEXT,       -- LLM 完整推理过程
    evidence_item_ids                 TEXT,       -- LLM 实际引用的条目 IDs（JSON 数组）
    source_refs_json                  TEXT,       -- 引用条目元数据 JSON
    fetched_urls                      TEXT,       -- reflect 阶段 web_fetch 调用的 URL（JSON 数组）
    gate_result_json                  TEXT,
    sense_result_json                 TEXT,
    pre_score_result_json             TEXT,
    fetch_filter_result_json          TEXT,
    score_result_json                 TEXT,
    decide_result_json                TEXT,
    act_result_json                   TEXT,
    decision_signals_json             TEXT,
    sent_message                      TEXT,       -- 实际发送的消息正文（act 阶段填充）
    candidates_json                   TEXT,       -- 候选内容 JSON（fetch_filter/decide 阶段填充）
    error                             TEXT
);
CREATE INDEX IF NOT EXISTS ix_pd_sk_ts ON proactive_decisions (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_pd_stage ON proactive_decisions (stage, ts);
"""


_PROACTIVE_DECISION_COLUMNS: dict[str, str] = {
    "tick_id": "TEXT",
    "updated_ts": "TEXT NOT NULL DEFAULT ''",
    "sleep_state": "TEXT",
    "sleep_prob": "REAL",
    "sleep_available": "INTEGER",
    "sleep_data_lag_min": "INTEGER",
    "gate_result_json": "TEXT",
    "sense_result_json": "TEXT",
    "pre_score_result_json": "TEXT",
    "fetch_filter_result_json": "TEXT",
    "score_result_json": "TEXT",
    "decide_result_json": "TEXT",
    "act_result_json": "TEXT",
    "decision_signals_json": "TEXT",
    "sent_message": "TEXT",
    "candidates_json": "TEXT",
    "reasoning": "TEXT",
    "evidence_item_ids": "TEXT",
    "source_refs_json": "TEXT",
    "fetched_urls": "TEXT",
    "research_status": "TEXT",
    "research_rounds_used": "INTEGER",
    "research_tools_called": "TEXT",
    "research_evidence_count": "INTEGER",
    "research_reason": "TEXT",
    "fact_claims_count": "INTEGER",
}

_RAG_EVENT_COLUMNS: dict[str, str] = {
    "tick_id": "TEXT",
    "gate_type": "TEXT",
    "sufficiency_check_json": "TEXT",
}

_TURNS_COLUMNS: dict[str, str] = {
    "tool_chain_json": "TEXT",
    "raw_llm_output": "TEXT",
    "meme_tag": "TEXT",
    "meme_media_count": "INTEGER",
    "history_window": "INTEGER",
    "history_messages": "INTEGER",
    "history_chars": "INTEGER",
    "history_tokens": "INTEGER",
    "prompt_tokens": "INTEGER",
    "next_turn_baseline_tokens": "INTEGER",
    "react_iteration_count": "INTEGER",
    "react_input_sum_tokens": "INTEGER",
    "react_input_peak_tokens": "INTEGER",
    "react_final_input_tokens": "INTEGER",
}


def _ensure_turns_columns(conn: sqlite3.Connection) -> None:
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(turns)").fetchall()
    }
    for col, ddl in _TURNS_COLUMNS.items():
        if col in cols:
            continue
        conn.execute(f"ALTER TABLE turns ADD COLUMN {col} {ddl}")


def _ensure_proactive_decision_columns(conn: sqlite3.Connection) -> None:
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(proactive_decisions)").fetchall()
    }
    for col, ddl in _PROACTIVE_DECISION_COLUMNS.items():
        if col in cols:
            continue
        conn.execute(f"ALTER TABLE proactive_decisions ADD COLUMN {col} {ddl}")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_pd_tick_id ON proactive_decisions (tick_id)"
    )
    conn.execute(
        "UPDATE proactive_decisions SET updated_ts = ts WHERE updated_ts IS NULL OR updated_ts = ''"
    )


def _ensure_rag_event_columns(conn: sqlite3.Connection) -> None:
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(rag_events)").fetchall()
    }
    for col, ddl in _RAG_EVENT_COLUMNS.items():
        if col in cols:
            continue
        conn.execute(f"ALTER TABLE rag_events ADD COLUMN {col} {ddl}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_re_tick_id ON rag_events (tick_id)"
    )


def open_db(db_path: Path) -> sqlite3.Connection:
    """打开（或新建）observe.db，初始化 schema，返回连接。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.executescript(_SCHEMA_SQL)
    _ensure_proactive_decision_columns(conn)
    _ensure_rag_event_columns(conn)
    _ensure_turns_columns(conn)
    conn.commit()
    return conn

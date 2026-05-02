"""SQLite 连接管理与 schema 初始化。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# schema 与 observe.sql 保持同步，在代码里内嵌一份避免运行时文件依赖
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
    react_cache_prompt_tokens INTEGER,
    react_cache_hit_tokens INTEGER,
    error       TEXT                        -- NULL = 正常
);
CREATE INDEX IF NOT EXISTS ix_turns_sk_ts  ON turns (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_turns_source ON turns (source, ts);

CREATE TABLE IF NOT EXISTS rag_queries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT    NOT NULL,
    caller         TEXT    NOT NULL,    -- passive | proactive | explicit
    session_key    TEXT    NOT NULL,
    query          TEXT    NOT NULL,    -- rewrite 后的检索 query
    orig_query     TEXT,               -- 改写前原文，NULL = 未改写
    aux_queries    TEXT,               -- JSON: ["hypothesis1", ...]  HyDE 假想条目
    hits_json      TEXT,               -- JSON: [{id, type, score, summary, injected}]
    injected_count INTEGER NOT NULL DEFAULT 0,
    route_decision TEXT,               -- "RETRIEVE" | "NO_RETRIEVE" | NULL
    error          TEXT
);
CREATE INDEX IF NOT EXISTS ix_rq_sk_ts  ON rag_queries (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_rq_caller ON rag_queries (caller, ts);

-- ─────────────────────────────────────────────
-- 3. memory_writes  post-response 记忆写入记录
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

"""


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
    "react_cache_prompt_tokens": "INTEGER",
    "react_cache_hit_tokens": "INTEGER",
}


def _ensure_turns_columns(conn: sqlite3.Connection) -> None:
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(turns)").fetchall()
    }
    for col, ddl in _TURNS_COLUMNS.items():
        if col in cols:
            continue
        _ = conn.execute(f"ALTER TABLE turns ADD COLUMN {col} {ddl}")


def _migrate_removed_proactive_observe(conn: sqlite3.Connection) -> None:
    _ = conn.execute("DELETE FROM turns WHERE source = 'proactive'")
    _ = conn.execute("DROP TABLE IF EXISTS proactive_decisions")


def open_db(db_path: Path) -> sqlite3.Connection:
    """打开（或新建）observe.db，初始化 schema，返回连接。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    _ = conn.executescript(_SCHEMA_SQL)
    _ensure_turns_columns(conn)
    _migrate_removed_proactive_observe(conn)
    conn.commit()
    return conn

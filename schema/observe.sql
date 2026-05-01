-- Observe DB schema
-- Agent Loop + Proactive Loop 可观测性数据库
-- 版本：4 (2026-05-01)

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

-- ─────────────────────────────────────────────
-- 1. turns  每轮 agent 对话 / proactive tick
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,           -- ISO8601 UTC
    source      TEXT    NOT NULL,           -- 'agent' | 'proactive'
    session_key TEXT    NOT NULL,
    user_msg    TEXT,                       -- agent: 用户原文; proactive: NULL
    llm_output  TEXT    NOT NULL DEFAULT '', -- LLM 最终输出完整文本
    raw_llm_output TEXT,                    -- 装饰/清洗前的原始模型输出
    meme_tag    TEXT,                       -- 命中的 <meme:tag>
    meme_media_count INTEGER,               -- 命中的媒体数量
    tool_calls  TEXT,                       -- JSON: [{name, args, result}]（每次 tool 调用）
    tool_chain_json TEXT,                   -- JSON: [{text, calls:[{name,args,result}]}] 完整迭代链路
    history_window INTEGER,                 -- 下一轮实际保留的 history 窗口
    history_messages INTEGER,               -- 保留 history 的消息条数（展开后）
    history_chars INTEGER,                  -- 保留 history 序列化后的字符数
    history_tokens INTEGER,                 -- 保留 history 的估算 token
    prompt_tokens INTEGER,                  -- 当前 prompt 底座估算 token
    next_turn_baseline_tokens INTEGER,      -- 下一轮基线占用估算 token
    react_iteration_count INTEGER,          -- 本轮 ReAct 调用 LLM 的次数
    react_input_sum_tokens INTEGER,         -- 本轮所有 LLM 输入估算 token 累计
    react_input_peak_tokens INTEGER,        -- 本轮最大一次 LLM 输入估算 token
    react_final_input_tokens INTEGER,       -- 最后一次 LLM 输入估算 token
    react_cache_prompt_tokens INTEGER,      -- DeepSeek KV cache: 本轮输入中 hit+miss tokens
    react_cache_hit_tokens INTEGER,         -- DeepSeek KV cache: 本轮缓存命中 tokens
    error       TEXT                        -- NULL = 正常
);

CREATE INDEX IF NOT EXISTS ix_turns_sk_ts ON turns (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_turns_source ON turns (source, ts);

-- ─────────────────────────────────────────────
-- 2. rag_queries  当前 memory 检索记录
-- ─────────────────────────────────────────────
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

-- ─────────────────────────────────────────────
-- 4. proactive_decisions  主动链路关键决策
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS proactive_decisions (
    id                               INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_id                          TEXT    UNIQUE,
    ts                               TEXT    NOT NULL,
    updated_ts                       TEXT    NOT NULL,
    session_key                      TEXT    NOT NULL,
    stage                            TEXT    NOT NULL,
    reason_code                      TEXT,
    should_send                      INTEGER,
    action                           TEXT,
    gate_reason                      TEXT,
    pre_score                        REAL,
    base_score                       REAL,
    draw_score                       REAL,
    decision_score                   REAL,
    send_threshold                   REAL,
    interruptibility                 REAL,
    candidate_count                  INTEGER,
    candidate_item_ids               TEXT,
    sleep_state                      TEXT,
    sleep_prob                       REAL,
    sleep_available                  INTEGER,
    sleep_data_lag_min               INTEGER,
    user_replied_after_last_proactive INTEGER,
    proactive_sent_24h               INTEGER,
    fresh_items_24h                  INTEGER,
    delivery_key                     TEXT,
    is_delivery_duplicate            INTEGER,
    is_message_duplicate             INTEGER,
    delivery_attempted               INTEGER,
    delivery_result                  TEXT,
    reasoning_preview                TEXT,
    sent_message                     TEXT,       -- 实际发送的消息正文（act 阶段填充）
    candidates_json                  TEXT,       -- 候选内容 JSON: [{kind, source_type, source_name, title, content, url, severity?}]
    gate_result_json                 TEXT,
    sense_result_json                TEXT,
    pre_score_result_json            TEXT,
    fetch_filter_result_json         TEXT,
    score_result_json                TEXT,
    decide_result_json               TEXT,
    act_result_json                  TEXT,
    decision_signals_json            TEXT,
    error                            TEXT
);

CREATE INDEX IF NOT EXISTS ix_pd_sk_ts   ON proactive_decisions (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_pd_stage   ON proactive_decisions (stage, ts);
CREATE UNIQUE INDEX IF NOT EXISTS ux_pd_tick_id ON proactive_decisions (tick_id);

-- ─────────────────────────────────────────────
-- 淘汰策略（由 retention.py 执行，不在 schema 里 enforce）
-- turns:               180 天
-- rag_queries:          90 天
-- proactive_decisions:  90 天
-- 例外：error IS NOT NULL 的行永久保留
-- ─────────────────────────────────────────────

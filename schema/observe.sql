-- Observe DB schema
-- Agent Loop + Proactive Loop 可观测性数据库
-- 版本：3 (2026-03-14)

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
    error       TEXT                        -- NULL = 正常
);

CREATE INDEX IF NOT EXISTS ix_turns_sk_ts ON turns (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_turns_source ON turns (source, ts);

-- ─────────────────────────────────────────────
-- 2. rag_events  每次 memory 检索事件
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rag_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL,
    source              TEXT    NOT NULL,   -- 'agent' | 'proactive'
    session_key         TEXT    NOT NULL,
    tick_id             TEXT,               -- proactive: 关联 proactive_decisions.tick_id
    -- query 链路
    original_query      TEXT    NOT NULL,   -- agent: user_msg; proactive: build_proactive_memory_query 输出前的原始 query
    query               TEXT    NOT NULL,   -- 实际用于检索的 query（route decision 改写后）
    gate_type           TEXT,
    route_decision      TEXT,               -- 'RETRIEVE' | 'NO_RETRIEVE'（仅 agent）
    route_latency_ms    INTEGER,
    -- HyDE
    hyde_hypothesis     TEXT,               -- HyDE 生成的假设文本（NULL = 未使用 HyDE）
    -- history 检索元信息
    history_scope_mode  TEXT,               -- scoped / global / global-fallback / disabled
    history_gate_reason TEXT,               -- 仅 proactive
    -- 最终注入内容（完整，不截断）
    injected_block      TEXT,
    preference_block    TEXT,
    preference_query    TEXT,
    sufficiency_check_json TEXT,
    fallback_reason     TEXT,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS ix_re_sk_ts   ON rag_events (session_key, ts);
CREATE INDEX IF NOT EXISTS ix_re_source  ON rag_events (source, ts);
CREATE INDEX IF NOT EXISTS ix_re_tick_id ON rag_events (tick_id);

-- ─────────────────────────────────────────────
-- 3. rag_items  每个检索到的 item 展开一行（raw data）
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rag_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    rag_event_id    INTEGER NOT NULL REFERENCES rag_events (id),
    -- vector_search 原始返回字段，不加工
    item_id         TEXT    NOT NULL,
    memory_type     TEXT    NOT NULL,       -- event | profile | procedure | preference
    score           REAL    NOT NULL,
    summary         TEXT    NOT NULL,       -- 完整 summary
    happened_at     TEXT,
    extra_json      TEXT,                   -- 原始 extra_json 序列化为 JSON string
    -- 检索路径（区分来源）
    retrieval_path  TEXT    NOT NULL,
    -- procedure | history_raw | history_hyde | preference
    injected        INTEGER NOT NULL DEFAULT 0  -- 1 = 最终注入到 context
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

-- ─────────────────────────────────────────────
-- 5. proactive_decisions  主动链路关键决策
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
-- turns:      180 天
-- rag_events:  90 天
-- rag_items:   90 天（随 rag_events 级联）
-- 例外：error IS NOT NULL 的行永久保留
-- ─────────────────────────────────────────────

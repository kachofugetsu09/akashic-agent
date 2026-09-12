"""SQLite schema for the derived sparse turn index."""

INDEX_VERSION = "10"
TOOL_CHAIN_PROJECTION_VERSION = "1"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sparse_turns (
    turn_id              TEXT PRIMARY KEY,
    session_key          TEXT NOT NULL,
    user_seq             INTEGER NOT NULL,
    user_message_id      TEXT NOT NULL UNIQUE,
    assistant_message_id TEXT NOT NULL UNIQUE,
    started_at           TEXT NOT NULL,
    committed_at         TEXT NOT NULL,
    user_text            TEXT NOT NULL,
    assistant_text       TEXT NOT NULL,
    assistant_tool_chain_json TEXT,
    remember_targets_json TEXT NOT NULL,
    forget_targets_json   TEXT NOT NULL,
    remember_boost        REAL NOT NULL CHECK (
        remember_boost >= 1.0 AND remember_boost <= 3.0
    ),
    source_digest        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sparse_turns_committed
    ON sparse_turns(committed_at, session_key, user_seq);

CREATE TABLE IF NOT EXISTS sparse_features (
    turn_id      TEXT NOT NULL REFERENCES sparse_turns(turn_id) ON DELETE CASCADE,
    family       TEXT NOT NULL,
    feature_id   TEXT NOT NULL,
    value        REAL NOT NULL,
    rank         INTEGER NOT NULL,
    evidence_json TEXT NOT NULL,
    PRIMARY KEY (turn_id, family, feature_id)
);

CREATE INDEX IF NOT EXISTS idx_sparse_features_lookup
    ON sparse_features(family, feature_id, value DESC);

CREATE TABLE IF NOT EXISTS turn_terms (
    turn_id TEXT NOT NULL REFERENCES sparse_turns(turn_id) ON DELETE CASCADE,
    field   TEXT NOT NULL,
    term    TEXT NOT NULL,
    tf      INTEGER NOT NULL CHECK (tf > 0),
    PRIMARY KEY (turn_id, field, term)
);

CREATE INDEX IF NOT EXISTS idx_turn_terms_lookup
    ON turn_terms(field, term, turn_id);

CREATE TABLE IF NOT EXISTS lexical_corpora (
    field        TEXT PRIMARY KEY,
    doc_count    INTEGER NOT NULL,
    total_length INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS lexical_stats (
    field TEXT NOT NULL,
    term  TEXT NOT NULL,
    df    INTEGER NOT NULL,
    PRIMARY KEY (field, term)
);

CREATE TABLE IF NOT EXISTS turn_dense (
    turn_id   TEXT NOT NULL REFERENCES sparse_turns(turn_id) ON DELETE CASCADE,
    field     TEXT NOT NULL,
    source_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim       INTEGER NOT NULL,
    PRIMARY KEY (turn_id, field)
);

CREATE TABLE IF NOT EXISTS time_observations (
    turn_id                  TEXT PRIMARY KEY REFERENCES sparse_turns(turn_id) ON DELETE CASCADE,
    channel                  TEXT NOT NULL,
    previous_turn_id         TEXT,
    session_turn_index       INTEGER NOT NULL,
    start_gap_seconds        REAL,
    log_start_gap            REAL,
    response_delta_seconds   REAL,
    idle_gap_seconds         REAL,
    log_idle_gap             REAL,
    overlap_seconds          REAL,
    log_overlap              REAL,
    persisted_message_span_seconds REAL NOT NULL,
    log_persisted_message_span     REAL NOT NULL,
    local_hour               REAL NOT NULL,
    hour_sin                 REAL NOT NULL,
    hour_cos                 REAL NOT NULL,
    weekday                  INTEGER NOT NULL,
    weekday_sin              REAL NOT NULL,
    weekday_cos              REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_time_observations_channel_idle_gap
    ON time_observations(channel, log_idle_gap, turn_id);

CREATE TABLE IF NOT EXISTS time_stats (
    channel                 TEXT PRIMARY KEY,
    idle_gap_count          INTEGER NOT NULL,
    mean_log_idle_gap       REAL NOT NULL,
    m2_log_idle_gap         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS stream_state (
    session_key       TEXT PRIMARY KEY,
    last_started_at   TEXT NOT NULL,
    last_committed_at TEXT NOT NULL,
    last_turn_id      TEXT NOT NULL,
    turn_count        INTEGER NOT NULL
);
"""

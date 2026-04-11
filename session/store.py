from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionStore:
    """SQLite-backed store for session metadata and messages."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._has_fts = False
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    key               TEXT PRIMARY KEY,
                    created_at        TEXT NOT NULL,
                    updated_at        TEXT NOT NULL,
                    last_consolidated INTEGER NOT NULL DEFAULT 0,
                    metadata          TEXT
                )
                """
            )
            self._ensure_session_columns()
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id          TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    seq         INTEGER NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT,
                    tool_chain  TEXT,
                    extra       TEXT,
                    ts          TEXT NOT NULL,
                    UNIQUE (session_key, seq)
                )
                """
            )
            self._ensure_fts()
            self._conn.commit()

    def _ensure_session_columns(self) -> None:
        rows = self._conn.execute("PRAGMA table_info(sessions)").fetchall()
        existing = {str(row["name"]) for row in rows}
        if "last_user_at" not in existing:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN last_user_at TEXT"
            )
        if "last_proactive_at" not in existing:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN last_proactive_at TEXT"
            )

    def _ensure_fts(self) -> None:
        try:
            # Migrate to trigram tokenizer if the table exists without it.
            # trigram supports CJK substring matching; the old unicode61 default does not.
            existing = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            if existing:
                try:
                    cfg = dict(
                        self._conn.execute("SELECT * FROM messages_fts_config").fetchall()
                    )
                    is_trigram = "trigram" in cfg.get("tokenize", "")
                except sqlite3.OperationalError:
                    is_trigram = False
                if not is_trigram:
                    self._conn.execute("DROP TABLE IF EXISTS messages_fts")
                    for trig in ("messages_ai", "messages_ad", "messages_au"):
                        self._conn.execute(f"DROP TRIGGER IF EXISTS {trig}")

            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    content='messages',
                    content_rowid='rowid',
                    tokenize='trigram'
                )
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                END
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END
                """
            )
            # Rebuild index so existing messages are covered by trigram.
            self._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
            self._conn.commit()
            self._has_fts = True
        except sqlite3.OperationalError:
            self._has_fts = False

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def session_exists(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE key = ?", (key,)
            ).fetchone()
        return row is not None

    def upsert_session(
        self,
        key: str,
        *,
        created_at: str,
        updated_at: str,
        last_consolidated: int,
        metadata: dict[str, Any],
    ) -> None:
        payload = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (key, created_at, updated_at, last_consolidated, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    last_consolidated = excluded.last_consolidated,
                    metadata = excluded.metadata
                """,
                (key, created_at, updated_at, int(last_consolidated), payload),
            )
            self._conn.commit()

    def update_last_consolidated(self, key: str, last_consolidated: int) -> None:
        now = datetime.now().astimezone().isoformat()
        with self._lock:
            self._conn.execute(
                """
                UPDATE sessions
                SET last_consolidated = ?, updated_at = ?
                WHERE key = ?
                """,
                (int(last_consolidated), now, key),
            )
            self._conn.commit()

    def get_session_meta(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT key, created_at, updated_at, last_consolidated, metadata, last_user_at, last_proactive_at FROM sessions WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "key": row["key"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_consolidated": int(row["last_consolidated"] or 0),
            "metadata": json.loads(row["metadata"] or "{}"),
            "last_user_at": row["last_user_at"],
            "last_proactive_at": row["last_proactive_at"],
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT key, created_at, updated_at, last_user_at, last_proactive_at
                FROM sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                "key": str(row["key"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_user_at": row["last_user_at"],
                "last_proactive_at": row["last_proactive_at"],
            }
            for row in rows
        ]

    def update_presence(
        self,
        key: str,
        *,
        last_user_at: str | None = None,
        last_proactive_at: str | None = None,
    ) -> None:
        now = datetime.now().astimezone().isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (
                    key,
                    created_at,
                    updated_at,
                    last_consolidated,
                    metadata,
                    last_user_at,
                    last_proactive_at
                )
                VALUES (?, ?, ?, 0, '{}', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    last_user_at = COALESCE(excluded.last_user_at, sessions.last_user_at),
                    last_proactive_at = COALESCE(excluded.last_proactive_at, sessions.last_proactive_at)
                """,
                (key, now, now, last_user_at, last_proactive_at),
            )
            self._conn.commit()

    def get_presence(self, key: str) -> dict[str, str | None] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT last_user_at, last_proactive_at
                FROM sessions
                WHERE key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "last_user_at": row["last_user_at"],
            "last_proactive_at": row["last_proactive_at"],
        }

    def list_presence(self) -> dict[str, dict[str, str | None]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT key, last_user_at, last_proactive_at
                FROM sessions
                WHERE last_user_at IS NOT NULL OR last_proactive_at IS NOT NULL
                """
            ).fetchall()
        return {
            str(row["key"]): {
                "last_user_at": row["last_user_at"],
                "last_proactive_at": row["last_proactive_at"],
            }
            for row in rows
        }

    def most_recent_user_at(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT MAX(last_user_at) AS last_user_at
                FROM sessions
                WHERE last_user_at IS NOT NULL
                """
            ).fetchone()
        if row is None:
            return None
        return row["last_user_at"]

    def get_channel_metadata(self, channel: str) -> list[dict[str, Any]]:
        like_key = f"{channel}:%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, metadata FROM sessions WHERE key LIKE ?", (like_key,)
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            key = str(row["key"])
            chat_id = key.split(":", 1)[-1] if ":" in key else key
            results.append(
                {
                    "key": key,
                    "chat_id": chat_id,
                    "metadata": json.loads(row["metadata"] or "{}"),
                }
            )
        return results

    def count_messages(self, session_key: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(1) AS c FROM messages WHERE session_key = ?", (session_key,)
            ).fetchone()
        return int((row["c"] if row else 0) or 0)

    def next_seq(self, session_key: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) AS m FROM messages WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        max_seq = row["m"] if row is not None else -1
        if max_seq is None:
            max_seq = -1
        return int(max_seq) + 1

    def insert_message(
        self,
        session_key: str,
        *,
        role: str,
        content: str,
        ts: str,
        seq: int,
        tool_chain: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = f"{session_key}:{seq}"
        tool_chain_payload = (
            json.dumps(tool_chain, ensure_ascii=False) if tool_chain is not None else None
        )
        extra_payload = json.dumps(extra or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO messages (id, session_key, seq, role, content, tool_chain, extra, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, session_key, seq, role, content, tool_chain_payload, extra_payload, ts),
            )
            self._conn.commit()
        row = {
            "id": message_id,
            "session_key": session_key,
            "seq": seq,
            "role": role,
            "content": content,
            "timestamp": ts,
        }
        if tool_chain is not None:
            row["tool_chain"] = tool_chain
        if extra:
            row.update(extra)
        return row

    def fetch_session_messages(self, session_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE session_key = ?
                ORDER BY seq ASC
                """,
                (session_key,),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def fetch_by_ids_with_context(self, ids: list[str], context: int) -> list[dict[str, Any]]:
        """Fetch messages by ID, expanding each hit by ±context rows in its session.

        Returns messages ordered by (session_key, seq).
        Each dict includes ``in_source_ref: bool`` to distinguish hits from context.
        """
        if not ids:
            return []
        if context == 0:
            result = self.fetch_by_ids(ids)
            for m in result:
                m["in_source_ref"] = True
            return result

        id_set = set(ids)
        session_seqs: dict[str, set[int]] = {}
        for msg_id in ids:
            parts = msg_id.rsplit(":", 1)
            if len(parts) != 2:
                continue
            sk, seq_str = parts
            try:
                seq = int(seq_str)
            except ValueError:
                continue
            if sk not in session_seqs:
                session_seqs[sk] = set()
            session_seqs[sk].add(seq)

        if not session_seqs:
            return []

        results: list[dict[str, Any]] = []
        with self._lock:
            for sk, seqs in session_seqs.items():
                expanded: set[int] = set()
                for seq in seqs:
                    for s in range(max(0, seq - context), seq + context + 1):
                        expanded.add(s)
                placeholders = ",".join("?" * len(expanded))
                rows = self._conn.execute(
                    f"SELECT id, session_key, seq, role, content, tool_chain, extra, ts "
                    f"FROM messages WHERE session_key = ? AND seq IN ({placeholders}) ORDER BY seq",
                    [sk, *expanded],
                ).fetchall()
                for row in rows:
                    msg = self._row_to_message(row)
                    msg["in_source_ref"] = msg["id"] in id_set
                    results.append(msg)
        return results

    def fetch_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        order_expr = " ".join(f"WHEN ? THEN {i}" for i in range(len(ids)))
        sql = (
            "SELECT id, session_key, seq, role, content, tool_chain, extra, ts FROM messages "
            f"WHERE id IN ({placeholders}) ORDER BY CASE id {order_expr} END"
        )
        with self._lock:
            rows = self._conn.execute(sql, tuple(ids + ids)).fetchall()
        return [self._row_to_message(row) for row in rows]

    def search_messages(
        self,
        query: str,
        *,
        session_key: str | None = None,
        role: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))
        params: list[Any] = []
        where_parts: list[str] = []
        if session_key:
            where_parts.append("m.session_key = ?")
            params.append(session_key)
        if role:
            where_parts.append("m.role = ?")
            params.append(role)
        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        # Split into individual terms for both FTS and LIKE paths.
        terms = [t for t in query.split() if t]
        if not terms:
            terms = [query]

        term_conditions_or = " OR ".join("m.content LIKE ?" for _ in terms)
        score_expr = " + ".join(
            f"(CASE WHEN m.content LIKE ? THEN 1 ELSE 0 END)" for _ in terms
        )
        if self._has_fts:
            # 长词走 FTS，短词继续走 LIKE，再把两路结果合并去重。
            fts_terms = [t for t in terms if len(t) >= 3]
            if fts_terms:
                fts_query = " OR ".join(fts_terms)
                connector = "AND" if where_sql else "WHERE"
                count_params = [fts_query] + params[:]
                count_sql = (
                    "SELECT COUNT(1) AS c "
                    "FROM messages m "
                    "LEFT JOIN ("
                    "    SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?"
                    ") fts ON m.rowid = fts.rowid "
                    f"{where_sql} {connector} (fts.rowid IS NOT NULL OR ({term_conditions_or})) "
                )
                count_params.extend(f"%{t}%" for t in terms)
                fts_params: list[Any] = []
                fts_sql = (
                    "SELECT m.id, m.session_key, m.seq, m.role, m.content, m.tool_chain, m.extra, m.ts, "
                    f"({score_expr}) AS match_score, "
                    "fts.rank_score AS rank_score "
                    "FROM messages m "
                    "LEFT JOIN ("
                    "    SELECT rowid, bm25(messages_fts) AS rank_score "
                    "    FROM messages_fts WHERE messages_fts MATCH ?"
                    ") fts ON m.rowid = fts.rowid "
                    f"{where_sql} {connector} (fts.rowid IS NOT NULL OR ({term_conditions_or})) "
                    "ORDER BY match_score DESC, "
                    "CASE WHEN rank_score IS NULL THEN 1 ELSE 0 END ASC, "
                    "rank_score ASC, m.seq DESC LIMIT ? OFFSET ?"
                )
                fts_params.extend(f"%{t}%" for t in terms)
                fts_params.append(fts_query)
                fts_params.extend(params[:])
                fts_params.extend(f"%{t}%" for t in terms)
                fts_params.extend([limit, offset])
                try:
                    with self._lock:
                        count_row = self._conn.execute(count_sql, tuple(count_params)).fetchone()
                        rows = self._conn.execute(fts_sql, tuple(fts_params)).fetchall()
                    total = int((count_row["c"] if count_row else 0) or 0)
                    return [self._row_to_message(row) for row in rows], total
                except sqlite3.OperationalError:
                    pass

        # LIKE fallback: OR across all terms so any hit surfaces; rank by match count descending.
        like_params = params[:]
        count_params = params[:]
        connector = "AND" if where_sql else "WHERE"
        count_sql = f"SELECT COUNT(1) AS c FROM messages m {where_sql} {connector} ({term_conditions_or}) "
        count_params.extend(f"%{t}%" for t in terms)
        like_sql = (
            f"SELECT m.id, m.session_key, m.seq, m.role, m.content, m.tool_chain, m.extra, m.ts, "
            f"({score_expr}) AS match_score "
            f"FROM messages m {where_sql} {connector} ({term_conditions_or}) "
            f"ORDER BY match_score DESC, m.seq DESC LIMIT ? OFFSET ?"
        )
        # score_expr binds: one %t% per term; term_conditions_or binds: one %t% per term
        like_params.extend(f"%{t}%" for t in terms)  # for score_expr
        like_params.extend(f"%{t}%" for t in terms)  # for WHERE OR
        like_params.extend([limit, offset])
        with self._lock:
            count_row = self._conn.execute(count_sql, tuple(count_params)).fetchone()
            rows = self._conn.execute(like_sql, tuple(like_params)).fetchall()
        total = int((count_row["c"] if count_row else 0) or 0)
        return [self._row_to_message(row) for row in rows], total

    def _row_to_message(self, row: sqlite3.Row) -> dict[str, Any]:
        message: dict[str, Any] = {
            "id": row["id"],
            "session_key": row["session_key"],
            "seq": int(row["seq"]),
            "role": row["role"],
            "content": row["content"] or "",
            "timestamp": row["ts"],
        }
        tool_chain = row["tool_chain"]
        if tool_chain:
            message["tool_chain"] = json.loads(tool_chain)
        extra = json.loads(row["extra"] or "{}")
        if extra:
            message.update(extra)
        return message

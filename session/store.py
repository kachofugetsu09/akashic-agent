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

    def _ensure_fts(self) -> None:
        try:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    content='messages',
                    content_rowid='rowid'
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
                "SELECT key, created_at, updated_at, last_consolidated, metadata FROM sessions WHERE key = ?",
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
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT key, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                "key": str(row["key"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

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
        return int((row["m"] if row else -1) or -1) + 1

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
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        params: list[Any] = []
        where_parts: list[str] = []
        if session_key:
            where_parts.append("m.session_key = ?")
            params.append(session_key)
        if role:
            where_parts.append("m.role = ?")
            params.append(role)
        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        if self._has_fts:
            sql = (
                "SELECT m.id, m.session_key, m.seq, m.role, m.content, m.tool_chain, m.extra, m.ts "
                "FROM messages_fts f JOIN messages m ON m.rowid = f.rowid "
                "WHERE f.content MATCH ? "
            )
            params = [query] + params
            if where_sql:
                sql += "AND " + where_sql[6:] + " "
            sql += "ORDER BY f.rank, m.seq DESC LIMIT ?"
            params.append(limit)
        else:
            sql = (
                "SELECT m.id, m.session_key, m.seq, m.role, m.content, m.tool_chain, m.extra, m.ts "
                "FROM messages m "
                f"{where_sql} "
            )
            params_sql = params[:]
            if where_sql:
                sql += "AND m.content LIKE ? "
            else:
                sql += "WHERE m.content LIKE ? "
            params_sql.append(f"%{query}%")
            sql += "ORDER BY m.seq DESC LIMIT ?"
            params = params_sql + [limit]

        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [self._row_to_message(row) for row in rows]

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

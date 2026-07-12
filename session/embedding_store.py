from __future__ import annotations

import hashlib
import sqlite3
import struct
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path


class MessageEmbeddingStore:
    """sessions.db 中可复用的消息向量缓存。"""

    def __init__(self, db_path: str | Path) -> None:
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        with self._lock:
            _ = self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS message_embeddings (
                    message_id   TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    model        TEXT NOT NULL,
                    embedding    BLOB NOT NULL,
                    dim          INTEGER NOT NULL,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL,
                    PRIMARY KEY (message_id, model)
                );
                CREATE INDEX IF NOT EXISTS ix_message_embeddings_hash
                    ON message_embeddings (content_hash, model);
                CREATE TABLE IF NOT EXISTS message_embedding_migrations (
                    source_id      TEXT PRIMARY KEY,
                    completed_at   TEXT NOT NULL,
                    imported_count INTEGER NOT NULL
                );
                """
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._db.close()
            self._closed = True

    def _messages_table_exists_locked(self) -> bool:
        row = self._db.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'messages'
            """
        ).fetchone()
        return row is not None

    def get(
        self,
        *,
        message_id: str,
        content: str,
        model: str,
    ) -> list[float] | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT embedding, dim
                FROM message_embeddings
                WHERE message_id = ? AND model = ? AND content_hash = ?
                """,
                (message_id, model, _content_hash(content)),
            ).fetchone()
        if row is None:
            return None
        return _deserialize_f32(
            row["embedding"],
            expected_dim=row["dim"],
            message_id=message_id,
            model=model,
        )

    def upsert(
        self,
        *,
        message_id: str,
        content: str,
        model: str,
        embedding: list[float],
    ) -> None:
        if not embedding:
            raise ValueError("消息向量不能为空")
        blob = _serialize_f32(embedding)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            _ = self._db.execute(
                """
                INSERT INTO message_embeddings
                    (message_id, content_hash, model, embedding, dim, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id, model) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    embedding = excluded.embedding,
                    dim = excluded.dim,
                    updated_at = excluded.updated_at
                """,
                (
                    message_id,
                    _content_hash(content),
                    model,
                    blob,
                    len(embedding),
                    now,
                    now,
                ),
            )
            self._db.commit()

    def list(self, *, model: str) -> list[tuple[str, list[float]]]:
        with self._lock:
            if not self._messages_table_exists_locked():
                return []
            rows = self._db.execute(
                """
                SELECT
                    embedding.message_id,
                    embedding.content_hash,
                    embedding.embedding,
                    embedding.dim,
                    message.content
                FROM message_embeddings embedding
                INNER JOIN messages message ON message.id = embedding.message_id
                WHERE embedding.model = ?
                """,
                (model,),
            ).fetchall()
        return [
            (
                str(row["message_id"]),
                _deserialize_f32(
                    row["embedding"],
                    expected_dim=row["dim"],
                    message_id=str(row["message_id"]),
                    model=model,
                ),
            )
            for row in rows
            if str(row["content_hash"])
            == _content_hash(str(row["content"] or ""))
        ]

    def list_until(
        self,
        *,
        model: str,
        cutoff: str,
    ) -> list[tuple[str, list[float]]]:
        with self._lock:
            if not self._messages_table_exists_locked():
                return []
            rows = self._db.execute(
                """
                SELECT
                    embedding.message_id,
                    embedding.content_hash,
                    embedding.embedding,
                    embedding.dim,
                    message.content
                FROM message_embeddings embedding
                INNER JOIN messages message ON message.id = embedding.message_id
                WHERE embedding.model = ?
                  AND julianday(message.ts) <= julianday(?)
                ORDER BY julianday(message.ts), message.seq
                """,
                (model, cutoff),
            ).fetchall()
        return [
            (
                str(row["message_id"]),
                _deserialize_f32(
                    row["embedding"],
                    expected_dim=row["dim"],
                    message_id=str(row["message_id"]),
                    model=model,
                ),
            )
            for row in rows
            if str(row["content_hash"])
            == _content_hash(str(row["content"] or ""))
        ]

    def delete(self, message_ids: list[str]) -> int:
        clean_ids = [str(item).strip() for item in message_ids if str(item).strip()]
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        with self._lock:
            cursor = self._db.execute(
                f"DELETE FROM message_embeddings WHERE message_id IN ({placeholders})",
                clean_ids,
            )
            self._db.commit()
        return int(cursor.rowcount or 0)

    def import_legacy_rows_once(
        self,
        rows: Iterable[tuple[str, str, str, bytes, int, str, str]],
    ) -> int:
        source_id = "akasha_embedding_cache:v1"
        with self._lock:
            _ = self._db.execute("BEGIN IMMEDIATE")
            try:
                completed = self._db.execute(
                    "SELECT 1 FROM message_embedding_migrations WHERE source_id = ?",
                    (source_id,),
                ).fetchone()
                if completed is not None or not self._messages_table_exists_locked():
                    self._db.commit()
                    return 0
                messages = {
                    str(row["id"]): _content_hash(str(row["content"] or ""))
                    for row in self._db.execute(
                        "SELECT id, content FROM messages"
                    ).fetchall()
                }
                valid_rows = [
                    row
                    for row in rows
                    if messages.get(str(row[0])) == str(row[1])
                    and int(row[4]) > 0
                    and len(row[3]) == int(row[4]) * 4
                ]
                cursor = self._db.executemany(
                    """
                    INSERT OR IGNORE INTO message_embeddings
                        (message_id, content_hash, model, embedding, dim, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    valid_rows,
                )
                imported = int(cursor.rowcount or 0)
                _ = self._db.execute(
                    """
                    INSERT INTO message_embedding_migrations
                        (source_id, completed_at, imported_count)
                    VALUES (?, ?, ?)
                    """,
                    (source_id, datetime.now(timezone.utc).isoformat(), imported),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
            return imported


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _serialize_f32(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_f32(
    blob: object,
    *,
    expected_dim: object,
    message_id: str,
    model: str,
) -> list[float]:
    """校验缓存维度并反序列化一条 float32 向量。"""

    if (
        not isinstance(blob, bytes)
        or not isinstance(expected_dim, int)
        or expected_dim <= 0
        or len(blob) != expected_dim * 4
    ):
        raise ValueError(
            "消息向量缓存损坏: "
            f"message_id={message_id} model={model} "
            f"dim={expected_dim} bytes={len(blob) if isinstance(blob, bytes) else 'invalid'}"
        )
    return list(struct.unpack(f"{expected_dim}f", blob))

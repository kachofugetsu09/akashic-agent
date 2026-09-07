from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
import threading
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from session.log import MessageConflict, MessageLog
from session.message import Message


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


class MessageEmbeddings:
    """Core 的消息向量记录；文本投影由消费者选择，不授予 SQL、删除或重新嵌入能力。"""

    def __init__(self, log: MessageLog | None):
        self._log = log

    def bind(self, text: Callable[[Message], str]) -> EmbeddingRecords:
        if self._log is None:
            raise RuntimeError("candidate 验证期禁止访问正式消息向量")
        return EmbeddingRecords(self._log, text)


class EmbeddingRecords:
    """给定纯文本投影后，只读写实际消息对应的一份固定模型向量。"""

    def __init__(self, log: MessageLog, text: Callable[[Message], str]):
        self._log = log
        self._text = text

    def _content(self, message: Message) -> str:
        actual = self._log.reader(message.session_id).get(message.message_id)
        if actual is None or actual != message:
            raise MessageConflict("向量来源不等于实际已接纳消息")
        text = self._text(actual)
        if not isinstance(text, str):
            raise TypeError("消息向量文本投影必须返回字符串")
        return _content_hash(text)

    def read(self, message: Message, *, model: str, dimension: int) -> tuple[float, ...] | None:
        """已存在但与正文、空间或维度不匹配时失败，不能当作缺失后重新嵌入。"""
        self._check_space(model, dimension)
        # 同一个 Core 存储锁保护来源读取与记录操作；插件不会获得 connection。
        with self._log._lock:  # pyright: ignore[reportPrivateUsage]
            content_hash = self._content(message)
            return self._read(message.message_id, model, dimension, content_hash)

    def _read(self, message_id: str, model: str, dimension: int, content_hash: str) -> tuple[float, ...] | None:
        connection = self._log._connection  # pyright: ignore[reportPrivateUsage]
        row = connection.execute(
            "SELECT content_hash, embedding, dim FROM message_embeddings WHERE message_id=? AND model=?",
            (message_id, model),
        ).fetchone()
        if row is None:
            return None
        if row["content_hash"] != content_hash or row["dim"] != dimension:
            raise ValueError(f"消息向量与固定文本或维度不匹配: {message_id}")
        vector = tuple(_deserialize_f32(row["embedding"], expected_dim=dimension,
                                        message_id=message_id, model=model))
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"消息向量含非有限值: {message_id}")
        return vector

    def save(self, message: Message, *, model: str, embedding: Sequence[float]) -> None:
        """只增加首次取得的向量；同身份重放必须相同，不覆盖重建所需历史。"""
        vector = list(embedding)
        self._check_space(model, len(vector))
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("消息向量必须包含有限值")
        blob = _serialize_f32(vector)
        encoded = tuple(_deserialize_f32(blob, expected_dim=len(vector),
                                         message_id=message.message_id, model=model))
        if not all(math.isfinite(value) for value in encoded):
            raise ValueError("消息向量不能超出 float32 范围")

        def insert() -> None:
            content_hash = self._content(message)
            existing = self._read(message.message_id, model, len(vector), content_hash)
            if existing is not None:
                if existing != encoded:
                    raise MessageConflict("消息模型向量已经固定，不能覆盖历史")
                return
            stamp = datetime.now(timezone.utc).isoformat()
            _ = self._log._connection.execute(  # pyright: ignore[reportPrivateUsage]
                "INSERT INTO message_embeddings "
                "(message_id,content_hash,model,embedding,dim,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (message.message_id, content_hash, model, blob, len(vector), stamp, stamp),
            )
        _ = self._log._write(insert)  # pyright: ignore[reportPrivateUsage]

    @staticmethod
    def _check_space(model: str, dimension: int) -> None:
        if not isinstance(model, str) or not model or type(dimension) is not int or dimension < 1:
            raise ValueError("向量需要固定模型身份和正维度")

from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .log import MessageConflict, MessageLog
from plugins.ledger.contract import Message


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
    """独立派生库的消息向量；不授予源消息写入或删除能力。"""

    def __init__(self, log: MessageLog, path: Path):
        self._log = log
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        try:
            with self._connection:
                self._connection.execute("""CREATE TABLE IF NOT EXISTS message_embeddings (
                    message_id TEXT NOT NULL, content_hash TEXT NOT NULL,
                    model TEXT NOT NULL, embedding BLOB NOT NULL, dim INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (message_id, model)
                )""")
                self._connection.execute("""CREATE INDEX IF NOT EXISTS ix_message_embeddings_hash
                    ON message_embeddings (content_hash, model)""")
        except BaseException:
            self._connection.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def bind(self, text: Callable[[Message], str]) -> EmbeddingRecords:
        return EmbeddingRecords(self, text)


class EmbeddingRecords:
    """给定纯文本投影后，只读写实际消息对应的一份固定模型向量。"""

    def __init__(self, store: MessageEmbeddings, text: Callable[[Message], str]):
        self._store = store
        self._text = text

    def _content(self, message: Message) -> str:
        actual = self._store._log.reader(message.session_id).get(message.message_id)
        if actual is None or actual != message:
            raise MessageConflict("向量来源不等于实际已接纳消息")
        text = self._text(actual)
        if not isinstance(text, str):
            raise TypeError("消息向量文本投影必须返回字符串")
        return _content_hash(text)

    def read(self, message: Message, *, model: str, dimension: int) -> tuple[float, ...] | None:
        """已存在但与正文、空间或维度不匹配时失败，不能当作缺失后重新嵌入。"""
        self._check_space(model, dimension)
        # 消息只追加，跨库操作不声称与源库事务原子提交。
        with self._store._lock:
            content_hash = self._content(message)
            return self._read(message.message_id, model, dimension, content_hash)

    def _read(self, message_id: str, model: str, dimension: int, content_hash: str) -> tuple[float, ...] | None:
        connection = self._store._connection
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
            _ = self._store._connection.execute(
                "INSERT INTO message_embeddings "
                "(message_id,content_hash,model,embedding,dim,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (message.message_id, content_hash, model, blob, len(vector), stamp, stamp),
            )
        with self._store._lock, self._store._connection:
            insert()

    @staticmethod
    def _check_space(model: str, dimension: int) -> None:
        if not isinstance(model, str) or not model or type(dimension) is not int or dimension < 1:
            raise ValueError("向量需要固定模型身份和正维度")

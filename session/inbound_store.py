from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import cast

_SCHEMA = {
    "inbound_handoffs": """CREATE TABLE IF NOT EXISTS inbound_handoffs (
    handoff_id TEXT PRIMARY KEY, dedupe_key TEXT UNIQUE,
    channel TEXT NOT NULL, sender TEXT NOT NULL, chat_id TEXT NOT NULL,
    session_key TEXT NOT NULL, content TEXT NOT NULL, timestamp TEXT NOT NULL,
    media_json TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
)""",
    "idx_inbound_handoffs_session": """CREATE INDEX IF NOT EXISTS idx_inbound_handoffs_session ON inbound_handoffs(session_key, created_at)""",
}


def init_inbound_handoffs(connection: sqlite3.Connection) -> None:
    """只初始化缺失的空表；已有表及索引必须符合原 schema。"""
    def sql(value: str) -> str:
        return re.sub(r"\s+", "", value.lower().replace("if not exists", "")).rstrip(";")

    # 1. 先核对完整结构，损坏的部分 schema 不得自动修补。
    existing: set[str] = set()
    for name, statement in _SCHEMA.items():
        row = connection.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()
        if row is not None:
            if sql(row[0]) != sql(statement):
                raise RuntimeError(f"{name} schema 不匹配")
            existing.add(name)
    if existing and existing != set(_SCHEMA):
        raise RuntimeError("inbound_handoffs schema 不完整")
    # 2. DDL 与调用方的写事务一起提交，不修改既有行。
    for statement in _SCHEMA.values():
        _ = connection.execute(statement)


class InboundHandoffStore:
    """独占输入交接记录；不持有消息、历史或会话删除权限。"""

    def __init__(self, path: str | Path):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            with self._conn:
                _ = self._conn.execute("BEGIN IMMEDIATE")
                init_inbound_handoffs(self._conn)
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def reserve_inbound_handoff(
        self,
        *,
        handoff_id: str,
        dedupe_key: str | None,
        channel: str,
        sender: str,
        chat_id: str,
        session_key: str,
        content: str,
        timestamp: str,
        media_json: str,
        metadata_json: str,
        created_at: str,
    ) -> tuple[str, bool]:
        """在 MessageBus 暴露输入前持久接纳完整交接记录。"""

        # 1. 固定完整身份；客户端重投只允许传输时间不同。
        fields = (
            handoff_id,
            dedupe_key,
            channel,
            sender,
            chat_id,
            session_key,
            content,
            timestamp,
            media_json,
            metadata_json,
            created_at,
        )
        if not all(
            isinstance(value, str) and value for value in fields if value is not None
        ):
            raise ValueError("inbound handoff fields must be non-empty strings")
        identity: dict[str, str | None] = {
            "dedupe_key": dedupe_key,
            "channel": channel,
            "sender": sender,
            "chat_id": chat_id,
            "session_key": session_key,
            "content": content,
            "timestamp": timestamp,
            "media_json": media_json,
            "metadata_json": metadata_json,
        }
        stable_identity = {
            key: value for key, value in identity.items() if key != "timestamp"
        }

        def validate_existing(row: sqlite3.Row, *, include_timestamp: bool) -> None:
            expected_identity = identity if include_timestamp else stable_identity
            for column, expected in expected_identity.items():
                if row[column] != expected:
                    raise RuntimeError(
                        "inbound handoff identity conflict: "
                        f"handoff_id={handoff_id} field={column}"
                    )

        # 2. 复用相同记录；并发插入的胜出记录也必须核对身份。
        with self._lock:
            existing_by_id = self._conn.execute(
                "SELECT * FROM inbound_handoffs WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
            if dedupe_key is not None:
                existing_by_dedupe = self._conn.execute(
                    "SELECT * FROM inbound_handoffs WHERE dedupe_key = ?",
                    (dedupe_key,),
                ).fetchone()
            else:
                existing_by_dedupe = None
            if (
                existing_by_id is not None
                and existing_by_dedupe is not None
                and existing_by_id["handoff_id"] != existing_by_dedupe["handoff_id"]
            ):
                raise RuntimeError(
                    "inbound handoff identity conflict: handoff_id and dedupe_key differ"
                )
            existing = existing_by_id or existing_by_dedupe
            if existing is not None:
                validate_existing(
                    existing, include_timestamp=existing_by_id is not None
                )
                return str(existing["handoff_id"]), False
            cursor = self._conn.execute(
                """
                INSERT INTO inbound_handoffs(
                    handoff_id, dedupe_key, channel, sender, chat_id,
                    session_key, content, timestamp, media_json,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                fields,
            )
            row = self._conn.execute(
                "SELECT * FROM inbound_handoffs WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
            if row is None and dedupe_key is not None:
                row = self._conn.execute(
                    "SELECT * FROM inbound_handoffs WHERE dedupe_key = ?",
                    (dedupe_key,),
                ).fetchone()
            if row is None:
                self._conn.rollback()
                raise RuntimeError(f"inbound handoff disappeared: {handoff_id}")
            try:
                validate_existing(
                    row,
                    include_timestamp=row["handoff_id"] == handoff_id,
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return str(row["handoff_id"]), cursor.rowcount == 1

    def list_inbound_handoffs(
        self,
        *,
        limit: int | None = None,
        after: tuple[str, str] | None = None,
    ) -> list[dict[str, str | None]]:
        """按 durable 到达顺序读取有限页，跳过已扫描且仍在处理的记录。"""

        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise ValueError("inbound handoff limit 必须是正整数")
        limit_sql = "" if limit is None else " LIMIT ?"
        after_sql = "" if after is None else " WHERE (created_at, handoff_id) > (?, ?)"
        parameters = (() if after is None else after) + (() if limit is None else (limit,))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT handoff_id, dedupe_key, channel, sender, chat_id,
                       session_key, content, timestamp, media_json,
                       metadata_json, created_at
                FROM inbound_handoffs
                """ + after_sql + " ORDER BY created_at ASC, handoff_id ASC" + limit_sql,
                parameters,
            ).fetchall()
        return [{key: cast(str | None, row[key]) for key in row.keys()} for row in rows]

    def has_inbound_handoff(
        self,
        *,
        session_key: str,
        client_message_id: str,
    ) -> bool:
        """检查客户端消息是否仍有尚未完成的交接。"""

        return self.read_inbound_handoff(
            session_key=session_key,
            client_message_id=client_message_id,
        ) is not None

    def read_inbound_handoff(
        self,
        *,
        session_key: str,
        client_message_id: str,
    ) -> dict[str, str | None] | None:
        """读取一个仍由 durable queue 持有的 exact handoff。"""

        dedupe_key = f"{session_key}:{client_message_id}"
        with self._lock:
            row = self._conn.execute(
                """
                SELECT handoff_id, dedupe_key, channel, sender, chat_id,
                       session_key, content, timestamp, media_json,
                       metadata_json, created_at
                FROM inbound_handoffs
                WHERE dedupe_key = ?
                """,
                (dedupe_key,),
            ).fetchone()
        if row is None:
            return None
        return {key: cast(str | None, row[key]) for key in row.keys()}

    def complete_inbound_handoff(self, handoff_id: str) -> None:
        """处理 owner 确认完成后释放唯一交接记录。"""

        if not isinstance(handoff_id, str) or not handoff_id:
            raise ValueError("handoff_id must be a non-empty string")
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM inbound_handoffs WHERE handoff_id = ?",
                (handoff_id,),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise RuntimeError(f"inbound handoff not found: {handoff_id}")
            self._conn.commit()


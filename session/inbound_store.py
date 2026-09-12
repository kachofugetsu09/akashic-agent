from __future__ import annotations

import json
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


def _project_handoff_metadata(metadata_json: str) -> str:
    """Project historical handoff metadata onto the neutral durable contract."""

    try:
        value = json.loads(metadata_json)
    except (TypeError, ValueError):
        return metadata_json
    if not isinstance(value, dict):
        return metadata_json
    projected = dict(value)
    if projected.get("durable_inbound") is not True and projected.get(
        "mobile_v3_handoff"
    ) is True:
        projected["durable_inbound"] = True
    if "durable_handoff_id" not in projected and isinstance(
        projected.get("mobile_handoff_id"), str
    ):
        projected["durable_handoff_id"] = projected["mobile_handoff_id"]
    if "provider_message_id" not in projected and isinstance(
        projected.get("client_message_id"), str
    ):
        projected["provider_message_id"] = projected["client_message_id"]
    if "durable_attachment_refs" not in projected and isinstance(
        projected.get("mobile_v3_attachment_refs"), list
    ):
        projected["durable_attachment_refs"] = projected[
            "mobile_v3_attachment_refs"
        ]
    return json.dumps(projected, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _project_handoff_row(row: sqlite3.Row) -> dict[str, str | None]:
    """Return one row without modifying its authoritative SQLite bytes."""

    result = {key: cast(str | None, row[key]) for key in row.keys()}
    metadata_json = result.get("metadata_json")
    if isinstance(metadata_json, str):
        result["metadata_json"] = _project_handoff_metadata(metadata_json)
    return result


def _row_provider_message_id(row: sqlite3.Row) -> str | None:
    """Read the projected provider identity without changing the stored row."""

    metadata_json = row["metadata_json"]
    if not isinstance(metadata_json, str):
        return None
    try:
        metadata = json.loads(_project_handoff_metadata(metadata_json))
    except (TypeError, ValueError):
        return None
    value = metadata.get("provider_message_id") if isinstance(metadata, dict) else None
    return value if isinstance(value, str) and value else None


def _provider_message_id_from_json(metadata_json: str) -> str | None:
    """Read the projected provider identity from an incoming JSON value."""

    try:
        metadata = json.loads(_project_handoff_metadata(metadata_json))
    except (TypeError, ValueError):
        return None
    value = metadata.get("provider_message_id") if isinstance(metadata, dict) else None
    return value if isinstance(value, str) and value else None


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

        # 1. 固定完整身份；客户端重投只允许传输时间不同。历史别名只在
        # 读取旧 pending 行时投影，不能让新的写入继续产生旧 metadata。
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

        def validate_existing(
            row: sqlite3.Row,
            *,
            include_timestamp: bool,
            allow_legacy_dedupe: bool = False,
        ) -> None:
            expected_identity = identity if include_timestamp else stable_identity
            for column, expected in expected_identity.items():
                if column == "dedupe_key" and allow_legacy_dedupe:
                    continue
                actual = row[column]
                if column == "metadata_json" and isinstance(actual, str):
                    try:
                        actual_value = json.loads(_project_handoff_metadata(actual))
                        expected_value = json.loads(
                            _project_handoff_metadata(cast(str, expected))
                        )
                    except (TypeError, ValueError):
                        actual_value = _project_handoff_metadata(actual)
                        expected_value = expected
                    if actual_value != expected_value:
                        raise RuntimeError(
                            "inbound handoff identity conflict: "
                            f"handoff_id={handoff_id} field={column}"
                        )
                    continue
                if actual != expected:
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
            legacy_by_identity = self._find_identity_locked(
                channel=channel,
                session_key=session_key,
                provider_message_id=_provider_message_id_from_json(metadata_json),
            )
            if (
                existing_by_id is not None
                and existing_by_dedupe is not None
                and existing_by_id["handoff_id"] != existing_by_dedupe["handoff_id"]
            ):
                raise RuntimeError(
                    "inbound handoff identity conflict: handoff_id and dedupe_key differ"
                )
            existing = existing_by_id or existing_by_dedupe or legacy_by_identity
            if existing is not None:
                validate_existing(
                    existing,
                    include_timestamp=existing_by_id is not None,
                    allow_legacy_dedupe=(
                        (
                            existing is legacy_by_identity
                            and existing_by_dedupe is None
                            and existing_by_id is None
                        )
                        or (
                            existing_by_id is not None
                            and existing_by_id["dedupe_key"] != dedupe_key
                        )
                    ),
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
                row = self._find_identity_locked(
                    channel=channel,
                    session_key=session_key,
                    provider_message_id=_provider_message_id_from_json(metadata_json),
                )
            if row is None:
                self._conn.rollback()
                raise RuntimeError(f"inbound handoff disappeared: {handoff_id}")
            try:
                validate_existing(
                    row,
                    include_timestamp=row["handoff_id"] == handoff_id,
                    allow_legacy_dedupe=(row["dedupe_key"] != dedupe_key),
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
        return [_project_handoff_row(row) for row in rows]

    def has_inbound_handoff(
        self,
        *,
        channel: str,
        session_key: str,
        provider_message_id: str,
    ) -> bool:
        """检查 provider message 是否仍有尚未完成的交接。"""

        return self.read_inbound_handoff(
            channel=channel,
            session_key=session_key,
            provider_message_id=provider_message_id,
        ) is not None

    def read_inbound_handoff(
        self,
        *,
        channel: str,
        session_key: str,
        provider_message_id: str,
    ) -> dict[str, str | None] | None:
        """读取一个仍由 durable queue 持有的 exact handoff。"""

        with self._lock:
            row = self._find_identity_locked(
                channel=channel,
                session_key=session_key,
                provider_message_id=provider_message_id,
            )
        if row is None:
            return None
        return _project_handoff_row(row)

    def _find_identity_locked(
        self,
        *,
        channel: str,
        session_key: str,
        provider_message_id: str | None,
    ) -> sqlite3.Row | None:
        """Find one channel-scoped identity, including an old dedupe key."""

        if provider_message_id is None:
            return None
        dedupe_key = f"{channel}:{session_key}:{provider_message_id}"
        row = self._conn.execute(
            "SELECT * FROM inbound_handoffs WHERE dedupe_key = ?",
            (dedupe_key,),
        ).fetchone()
        if row is not None:
            return row
        rows = self._conn.execute(
            """
            SELECT * FROM inbound_handoffs
            WHERE channel = ? AND session_key = ?
            ORDER BY created_at ASC, handoff_id ASC
            """,
            (channel, session_key),
        ).fetchall()
        matches = [
            candidate
            for candidate in rows
            if _row_provider_message_id(candidate) == provider_message_id
        ]
        if len(matches) > 1:
            raise RuntimeError("inbound handoff channel identity 不唯一")
        if matches:
            return matches[0]
        return None

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

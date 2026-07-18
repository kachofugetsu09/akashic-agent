from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from agent.control.errors import TurnNotFoundError, TurnStateTransitionError
from agent.control.models import (
    TurnError,
    TurnItem,
    TurnRecord,
    TurnStatus,
    TurnUsage,
    parse_rfc3339,
)

logger = logging.getLogger(__name__)

_FTS_CAPABILITY_ERROR_MARKERS = (
    "no such module: fts5",
    "no such tokenizer: trigram",
    "unknown tokenizer: trigram",
)
_MESSAGE_COLUMN_FIELDS = frozenset(
    {"id", "session_key", "seq", "role", "content", "timestamp", "tool_chain"}
)
_TURN_TRANSITIONS = {
    TurnStatus.QUEUED: frozenset({TurnStatus.IN_PROGRESS, TurnStatus.CANCELLED}),
    TurnStatus.IN_PROGRESS: frozenset(
        {
            TurnStatus.COMPLETED,
            TurnStatus.INTERRUPTED,
            TurnStatus.FAILED,
            TurnStatus.CANCELLED,
        }
    ),
}


def _resolve_path_text(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.startswith(("http://", "https://")):
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except OSError:
        return ""


def _decode_session_metadata(
    raw: str | bytes | bytearray | None,
    session_key: str,
) -> dict[str, Any]:
    """解析并校验 sessions.metadata 的 JSON object 契约。"""

    metadata = _decode_json_payload(
        raw,
        fallback="{}",
        field="session metadata",
        identifier=session_key,
    )
    if not isinstance(metadata, dict):
        raise ValueError(f"session metadata 必须是 JSON object: {session_key}")
    return cast(dict[str, Any], metadata)


def _decode_message_extra(
    raw: str | bytes | bytearray | None,
    message_id: str,
) -> dict[str, Any]:
    """解析并校验一条消息的 extra JSON object。"""

    extra = _decode_json_payload(
        raw,
        fallback="{}",
        field="message extra",
        identifier=message_id,
    )
    if not isinstance(extra, dict):
        raise ValueError(f"message extra 必须是 JSON object: {message_id}")
    extra_dict = cast(dict[str, Any], extra)
    reserved_fields = _MESSAGE_COLUMN_FIELDS.intersection(extra_dict)
    if reserved_fields:
        fields = ", ".join(sorted(reserved_fields))
        raise ValueError(
            f"message extra 不得覆盖消息列字段 ({fields}): {message_id}"
        )
    media: object = extra_dict.get("media")
    if "media" in extra_dict and (
        not isinstance(media, list)
        or not all(isinstance(item, str) for item in cast(list[object], media))
    ):
        raise ValueError(f"message media 必须是字符串数组: {message_id}")
    source_refs: object = extra_dict.get("source_refs")
    if "source_refs" in extra_dict and (
        not isinstance(source_refs, list)
        or not all(
            isinstance(item, dict) for item in cast(list[object], source_refs)
        )
    ):
        raise ValueError(f"message source_refs 必须是对象数组: {message_id}")
    proactive = extra_dict.get("proactive")
    if "proactive" in extra_dict and not isinstance(proactive, bool):
        raise ValueError(f"message proactive 必须是布尔值: {message_id}")
    delivery_id = extra_dict.get("delivery_id")
    if "delivery_id" in extra_dict and (
        not isinstance(delivery_id, str) or not delivery_id or len(delivery_id) > 128
    ):
        raise ValueError(f"message delivery_id 必须是 1..128 字符串: {message_id}")
    for field in ("state_summary_tag", "reasoning_content"):
        value = extra_dict.get(field)
        if field in extra_dict and not isinstance(value, str):
            raise ValueError(f"message {field} 必须是字符串: {message_id}")
    if "model_state" in extra_dict:
        _validate_model_state(extra_dict["model_state"], message_id)
    return extra_dict


def _validate_model_state(value: object, message_id: str) -> None:
    """在数据库边界校验 Responses continuation state。"""
    if not isinstance(value, dict):
        raise ValueError(f"message model_state 必须是 JSON object: {message_id}")
    state = cast(dict[str, object], value)
    if state.get("schema_version") != 1:
        raise ValueError(f"message model_state schema_version 无效: {message_id}")
    for field in ("runtime_id", "transport", "model"):
        if not isinstance(state.get(field), str) or not state[field]:
            raise ValueError(f"message model_state.{field} 必须是非空字符串: {message_id}")
    items = state.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"message model_state.items 必须是对象数组: {message_id}")
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValueError(f"message model_state 超过 2 MiB: {message_id}")


def _decode_json_payload(
    raw: str | bytes | bytearray | None,
    *,
    fallback: str,
    field: str,
    identifier: str,
) -> object:
    """在 SQLite 反序列化边界统一转换 JSON 损坏错误。"""

    try:
        return json.loads(fallback if raw is None else raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError(f"{field} JSON 损坏: {identifier}") from exc


def _decode_turn_input(raw: object, turn_id: str) -> tuple[str, dict[str, Any]]:
    """解析并校验 turn 输入及其 metadata。"""
    payload = _decode_json_payload(
        cast(str | bytes | bytearray | None, raw),
        fallback="{}",
        field="turn input",
        identifier=turn_id,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"turn input 必须是 JSON object: {turn_id}")
    data = cast(dict[str, object], payload)
    input_text = data.get("input")
    metadata = data.get("metadata")
    if not isinstance(input_text, str):
        raise ValueError(f"turn input.input 必须是字符串: {turn_id}")
    if not isinstance(metadata, dict):
        raise ValueError(f"turn input.metadata 必须是 JSON object: {turn_id}")
    return input_text, cast(dict[str, Any], metadata)


def _decode_turn_items(raw: object, turn_id: str) -> list[TurnItem]:
    """解析并校验 turn item 数组。"""
    payload = _decode_json_payload(
        cast(str | bytes | bytearray | None, raw),
        fallback="[]",
        field="turn items",
        identifier=turn_id,
    )
    if not isinstance(payload, list):
        raise ValueError(f"turn items 必须是 JSON array: {turn_id}")
    return [TurnItem.from_dict(item) for item in cast(list[object], payload)]


def _decode_turn_usage(raw: object, turn_id: str) -> TurnUsage | None:
    if raw is None:
        return None
    payload = _decode_json_payload(
        cast(str | bytes | bytearray, raw),
        fallback="null",
        field="turn usage",
        identifier=turn_id,
    )
    return TurnUsage.from_dict(payload)


def _decode_turn_error(raw: object, turn_id: str) -> TurnError | None:
    if raw is None:
        return None
    payload = _decode_json_payload(
        cast(str | bytes | bytearray, raw),
        fallback="null",
        field="turn error",
        identifier=turn_id,
    )
    return TurnError.from_dict(payload)


def _decode_required_turn_time(raw: object, field_name: str, turn_id: str) -> datetime:
    value = parse_rfc3339(raw, field_name)
    if value is None:
        raise ValueError(f"turn {field_name} 不能为空: {turn_id}")
    return value


def _decode_message_tool_chain(
    raw: str | bytes | bytearray,
    message_id: str,
) -> list[dict[str, object]]:
    """解析并校验消息工具链的容器结构。"""

    tool_chain = _decode_json_payload(
        raw,
        fallback="[]",
        field="message tool_chain",
        identifier=message_id,
    )
    if not isinstance(tool_chain, list):
        raise ValueError(f"message tool_chain 必须是 JSON array: {message_id}")

    # 1. 校验每个工具轮次和调用容器。
    raw_groups = cast(list[object], tool_chain)
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise ValueError(
                f"message tool_chain[{group_index}] 必须是 JSON object: {message_id}"
            )
        group = cast(dict[str, object], raw_group)
        raw_calls = group.get("calls")
        if not isinstance(raw_calls, list):
            raise ValueError(
                f"message tool_chain[{group_index}].calls 必须是 JSON array: {message_id}"
            )
        # 2. 校验调用字段；稀疏工具链仍可用于消息查询展示。
        calls = cast(list[object], raw_calls)
        for call_index, raw_call in enumerate(calls):
            if not isinstance(raw_call, dict):
                raise ValueError(
                    "message tool_chain[{}].calls[{}] 必须是 JSON object: {}".format(
                        group_index, call_index, message_id
                    )
                )
            call = cast(dict[str, object], raw_call)
            for field in ("call_id", "name"):
                if field in call and not isinstance(call[field], str):
                    raise ValueError(
                        "message tool_chain[{}].calls[{}].{} 必须是字符串: {}".format(
                            group_index, call_index, field, message_id
                        )
                    )
            arguments = call.get("arguments")
            if arguments is not None and not isinstance(arguments, dict):
                raise ValueError(
                    "message tool_chain[{}].calls[{}].arguments 必须是 JSON object: {}".format(
                        group_index, call_index, message_id
                    )
                )
            result = call.get("result")
            if result is not None and not isinstance(result, str):
                raise ValueError(
                    "message tool_chain[{}].calls[{}].result 必须是字符串: {}".format(
                        group_index, call_index, message_id
                    )
                )

        # 3. 可选的展示字段若存在，也必须保持字符串契约。
        for field in ("text", "reasoning_content"):
            value = group.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"message tool_chain[{group_index}].{field} 必须是字符串: {message_id}"
                )
        if "model_state" in group:
            _validate_model_state(group["model_state"], message_id)
    return cast(list[dict[str, object]], tool_chain)


class SessionStore:
    """SQLite-backed store for session metadata and messages."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._closed = False
        self._has_fts = False
        self._init_schema()

    def __del__(self) -> None:
        if not self._closed:
            try:
                self.close()
            except sqlite3.Error as cleanup_error:
                logger.warning(
                    "SessionStore 析构关闭失败 db=%s err=%s",
                    self.db_path,
                    cleanup_error,
                )

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    key               TEXT PRIMARY KEY,
                    created_at        TEXT NOT NULL,
                    updated_at        TEXT NOT NULL,
                    last_consolidated INTEGER NOT NULL DEFAULT 0,
                    metadata          TEXT
                )
                """)
            self._ensure_session_columns()
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS session_admissions (
                    admission_id TEXT PRIMARY KEY,
                    session_key  TEXT NOT NULL,
                    created_at   TEXT NOT NULL
                )
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_admissions_session
                ON session_admissions(session_key)
                """)
            self._conn.execute("""
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
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id             TEXT PRIMARY KEY,
                    session_key    TEXT NOT NULL,
                    status         TEXT NOT NULL,
                    input_json     TEXT NOT NULL,
                    items_json     TEXT NOT NULL,
                    usage_json     TEXT,
                    error_json     TEXT,
                    final_response TEXT,
                    created_at     TEXT NOT NULL,
                    started_at     TEXT,
                    completed_at   TEXT
                )
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_session_created
                ON turns(session_key, created_at, id)
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_status
                ON turns(status)
                """)
            self._ensure_next_seq_values()
            self._ensure_fts()
            self._conn.commit()

    def _ensure_session_columns(self) -> None:
        rows = self._conn.execute("PRAGMA table_info(sessions)").fetchall()
        existing = {str(row["name"]) for row in rows}
        if "last_user_at" not in existing:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN last_user_at TEXT")
        if "last_proactive_at" not in existing:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN last_proactive_at TEXT")
        if "next_seq" not in existing:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN next_seq INTEGER NOT NULL DEFAULT 0"
            )

    def _ensure_next_seq_values(self) -> None:
        rows = self._conn.execute("SELECT key, next_seq FROM sessions").fetchall()
        for row in rows:
            session_key = str(row["key"])
            current = int(row["next_seq"] or 0)
            seq_row = self._conn.execute(
                "SELECT COALESCE(MAX(seq) + 1, 0) AS next_seq FROM messages WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            required = int((seq_row["next_seq"] if seq_row else 0) or 0)
            if current < required:
                self._conn.execute(
                    "UPDATE sessions SET next_seq = ? WHERE key = ?",
                    (required, session_key),
                )

    def _ensure_fts(self) -> None:
        """确保全文索引可用，并仅在创建或修复时重建已有消息。"""

        needs_rebuild = False
        try:
            # 1. 发现旧索引或缺失索引时，准备一次性重建。
            existing = self._conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            needs_rebuild = existing is None
            if existing:
                table_sql = "".join(str(existing["sql"] or "").split()).lower()
                is_trigram = "tokenize='trigram'" in table_sql
                if not is_trigram:
                    self._conn.execute("DROP TABLE IF EXISTS messages_fts")
                    for trig in ("messages_ai", "messages_ad", "messages_au"):
                        self._conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
                    needs_rebuild = True

                trigger_rows = self._conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'trigger' AND name IN (?, ?, ?)",
                    ("messages_ai", "messages_ad", "messages_au"),
                ).fetchall()
                needs_rebuild = needs_rebuild or len(trigger_rows) < 3

            # 2. 确保索引和消息写入触发器存在。
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    content='messages',
                    content_rowid='rowid',
                    tokenize='trigram'
                )
                """)
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END
                """)
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                END
                """)
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END
                """)
            # 3. 正常重启依赖触发器增量维护，避免重复扫描整张 messages 表。
            if needs_rebuild:
                self._conn.execute(
                    "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')"
                )
            self._conn.commit()
            self._has_fts = True
        except sqlite3.OperationalError as exc:
            if not any(
                marker in str(exc).lower() for marker in _FTS_CAPABILITY_ERROR_MARKERS
            ):
                raise
            logger.warning(
                "SQLite FTS5/trigram 不可用，已禁用 session 全文检索: %s", exc
            )
            self._has_fts = False

    def _has_message_embeddings_locked(self) -> bool:
        row = self._conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'message_embeddings'
            """
        ).fetchone()
        return row is not None

    def _delete_message_embeddings_locked(self, message_ids: list[str]) -> None:
        if not message_ids or not self._has_message_embeddings_locked():
            return
        placeholders = ",".join("?" for _ in message_ids)
        self._conn.execute(
            f"DELETE FROM message_embeddings WHERE message_id IN ({placeholders})",
            tuple(message_ids),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
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
            "metadata": _decode_session_metadata(row["metadata"], str(row["key"])),
            "last_user_at": row["last_user_at"],
            "last_proactive_at": row["last_proactive_at"],
        }

    def create_turn(self, record: TurnRecord) -> TurnRecord:
        """持久化一个 queued turn 并返回数据库中的正式记录。"""
        if record.status is not TurnStatus.QUEUED:
            raise TurnStateTransitionError("turn 创建时必须处于 queued 状态")
        if record.started_at is not None or record.completed_at is not None:
            raise TurnStateTransitionError("queued turn 不得包含 started_at/completed_at")
        if (
            record.usage is not None
            or record.error is not None
            or record.final_response is not None
        ):
            raise TurnStateTransitionError("queued turn 不得包含 usage/error/final_response")

        # 1. 在写入前完成所有 JSON 编码，序列化失败时数据库保持不变。
        input_json = json.dumps(
            {"input": record.input, "metadata": record.metadata},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        items_json = json.dumps(
            [item.to_dict() for item in record.items],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        usage_json = (
            json.dumps(record.usage.to_dict(), ensure_ascii=False, separators=(",", ":"))
            if record.usage is not None
            else None
        )
        error_json = (
            json.dumps(record.error.to_dict(), ensure_ascii=False, separators=(",", ":"))
            if record.error is not None
            else None
        )

        # 2. 单条 INSERT 建立不可变 turn identity。
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO turns (
                    id, session_key, status, input_json, items_json,
                    usage_json, error_json, final_response,
                    created_at, started_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.thread_id,
                    record.status.value,
                    input_json,
                    items_json,
                    usage_json,
                    error_json,
                    record.final_response,
                    record.created_at.isoformat(),
                    None,
                    None,
                ),
            )
            self._conn.commit()
        stored = self.read_turn(record.id)
        if stored is None:
            raise RuntimeError(f"turn 创建后无法读取: {record.id}")
        return stored

    def read_turn(self, turn_id: str) -> TurnRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, session_key, status, input_json, items_json,
                       usage_json, error_json, final_response,
                       created_at, started_at, completed_at
                FROM turns
                WHERE id = ?
                """,
                (turn_id,),
            ).fetchone()
        return self._row_to_turn(row) if row is not None else None

    def transition_turn(
        self,
        turn_id: str,
        *,
        expected_status: TurnStatus,
        status: TurnStatus,
        thread_id: str | None = None,
        items: list[TurnItem] | None = None,
        usage: TurnUsage | None = None,
        error: TurnError | None = None,
        final_response: str | None = None,
        now: datetime | None = None,
    ) -> TurnRecord:
        """用单条 CAS 更新 turn 状态，状态漂移时明确失败。"""
        expected_status = TurnStatus(expected_status)
        status = TurnStatus(status)
        allowed = _TURN_TRANSITIONS.get(expected_status, frozenset())
        if status not in allowed:
            raise TurnStateTransitionError(
                f"非法 turn 状态转换: {expected_status.value} -> {status.value}"
            )
        if status is TurnStatus.FAILED and error is None:
            raise TurnStateTransitionError("failed turn 必须包含 error")
        timestamp = now or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("turn transition 时间必须包含时区")
        timestamp = timestamp.astimezone(UTC)

        # 1. 只更新本次调用明确拥有的终态字段。
        set_parts = ["status = ?"]
        params: list[object] = [status.value]
        if status is TurnStatus.IN_PROGRESS:
            set_parts.append("started_at = ?")
            params.append(timestamp.isoformat())
        if status.is_terminal:
            set_parts.append("completed_at = ?")
            params.append(timestamp.isoformat())
        if items is not None:
            set_parts.append("items_json = ?")
            params.append(
                json.dumps(
                    [item.to_dict() for item in items],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        if usage is not None:
            set_parts.append("usage_json = ?")
            params.append(
                json.dumps(usage.to_dict(), ensure_ascii=False, separators=(",", ":"))
            )
        if error is not None:
            set_parts.append("error_json = ?")
            params.append(
                json.dumps(error.to_dict(), ensure_ascii=False, separators=(",", ":"))
            )
        if final_response is not None:
            set_parts.append("final_response = ?")
            params.append(final_response)

        # 2. status 和可选 thread identity 共同构成 compare-and-set 条件。
        where_parts = ["id = ?", "status = ?"]
        params.extend([turn_id, expected_status.value])
        if thread_id is not None:
            where_parts.append("session_key = ?")
            params.append(thread_id)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE turns SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}",
                tuple(params),
            )
            if cursor.rowcount != 1:
                current = self._conn.execute(
                    "SELECT session_key, status FROM turns WHERE id = ?", (turn_id,)
                ).fetchone()
                self._conn.rollback()
                if current is None:
                    raise TurnNotFoundError(f"turn 不存在: {turn_id}")
                if thread_id is not None and str(current["session_key"]) != thread_id:
                    raise TurnNotFoundError(
                        f"turn 不属于 thread: {thread_id}/{turn_id}"
                    )
                raise TurnStateTransitionError(
                    f"turn CAS 失败，期望 {expected_status.value}，实际 {current['status']}: {turn_id}"
                )
            self._conn.commit()

        # 3. 返回同一连接提交后可重读的正式记录。
        stored = self.read_turn(turn_id)
        if stored is None:
            raise RuntimeError(f"turn 转换后无法读取: {turn_id}")
        return stored

    def list_turns(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        before: tuple[str, str] | None = None,
    ) -> list[TurnRecord]:
        """按创建时间倒序读取一个 thread 的稳定 turn 页面。"""
        if limit <= 0 or limit > 200:
            raise ValueError("turn list limit 必须在 1..200")
        where = "session_key = ?"
        params: list[object] = [thread_id]
        if before is not None:
            where += " AND (created_at < ? OR (created_at = ? AND id < ?))"
            params.extend([before[0], before[0], before[1]])
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, session_key, status, input_json, items_json,
                       usage_json, error_json, final_response,
                       created_at, started_at, completed_at
                FROM turns
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._row_to_turn(row) for row in rows]

    def delete_thread_turns(self, thread_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM turns WHERE session_key = ?", (thread_id,)
            )
            self._conn.commit()
        return int(cursor.rowcount or 0)

    def _row_to_turn(self, row: sqlite3.Row) -> TurnRecord:
        """在 SQLite 边界把 turn 行恢复成严格领域对象。"""
        turn_id = str(row["id"])
        input_text, metadata = _decode_turn_input(row["input_json"], turn_id)
        return TurnRecord(
            id=turn_id,
            thread_id=str(row["session_key"]),
            status=TurnStatus(str(row["status"])),
            input=input_text,
            metadata=metadata,
            items=_decode_turn_items(row["items_json"], turn_id),
            usage=_decode_turn_usage(row["usage_json"], turn_id),
            error=_decode_turn_error(row["error_json"], turn_id),
            final_response=cast(str | None, row["final_response"]),
            created_at=_decode_required_turn_time(
                row["created_at"], "created_at", turn_id
            ),
            started_at=parse_rfc3339(row["started_at"], "started_at"),
            completed_at=parse_rfc3339(row["completed_at"], "completed_at"),
        )

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("""
                SELECT key, created_at, updated_at, last_user_at, last_proactive_at
                FROM sessions
                ORDER BY updated_at DESC
                """).fetchall()
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

    def list_sessions_for_dashboard(
        self,
        *,
        q: str = "",
        channel: str = "",
        updated_from: str = "",
        updated_to: str = "",
        has_proactive: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 200))
        offset = (safe_page - 1) * safe_page_size
        safe_sort_by = (
            sort_by
            if sort_by
            in {
                "updated_at",
                "created_at",
                "last_user_at",
                "last_proactive_at",
            }
            else "updated_at"
        )
        safe_sort_order = "ASC" if str(sort_order).lower() == "asc" else "DESC"

        params: list[Any] = []
        where_parts: list[str] = []
        query = (q or "").strip()
        if query:
            where_parts.append("(s.key LIKE ? OR COALESCE(s.metadata, '') LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like])
        if channel:
            where_parts.append("s.key LIKE ?")
            params.append(f"{channel}:%")
        if updated_from:
            where_parts.append("s.updated_at >= ?")
            params.append(updated_from)
        if updated_to:
            where_parts.append("s.updated_at <= ?")
            params.append(updated_to)
        if has_proactive is True:
            where_parts.append("s.last_proactive_at IS NOT NULL")
        if has_proactive is False:
            where_parts.append("s.last_proactive_at IS NULL")

        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        count_sql = f"""
            SELECT COUNT(1) AS c
            FROM sessions s
            {where_sql}
        """
        data_sql = f"""
            SELECT
                s.key,
                s.created_at,
                s.updated_at,
                s.last_consolidated,
                s.metadata,
                s.last_user_at,
                s.last_proactive_at,
                (
                    SELECT m.content
                    FROM messages m
                    WHERE m.session_key = s.key
                      AND m.role = 'user'
                      AND TRIM(COALESCE(m.content, '')) != ''
                    ORDER BY m.seq ASC
                    LIMIT 1
                ) AS first_message_content,
                (
                    SELECT COUNT(1)
                    FROM messages m
                    WHERE m.session_key = s.key
                ) AS message_count
            FROM sessions s
            {where_sql}
            ORDER BY s.{safe_sort_by} {safe_sort_order}, s.key ASC
            LIMIT ? OFFSET ?
        """
        with self._lock:
            count_row = self._conn.execute(count_sql, tuple(params)).fetchone()
            rows = self._conn.execute(
                data_sql,
                tuple([*params, safe_page_size, offset]),
            ).fetchall()
        total = int((count_row["c"] if count_row else 0) or 0)
        return [
            {
                "key": str(row["key"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_consolidated": int(row["last_consolidated"] or 0),
                "metadata": _decode_session_metadata(
                    row["metadata"], str(row["key"])
                ),
                "last_user_at": row["last_user_at"],
                "last_proactive_at": row["last_proactive_at"],
                "first_message_content": row["first_message_content"],
                "message_count": int(row["message_count"] or 0),
            }
            for row in rows
        ], total

    def create_session(
        self,
        *,
        key: str,
        metadata: dict[str, Any] | None = None,
        last_consolidated: int = 0,
        last_user_at: str | None = None,
        last_proactive_at: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now().astimezone().isoformat()
        payload = json.dumps(metadata or {}, ensure_ascii=False)
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
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    now,
                    now,
                    int(last_consolidated),
                    payload,
                    last_user_at,
                    last_proactive_at,
                ),
            )
            self._conn.commit()
        meta = self.get_session_meta(key)
        if meta is None:
            raise ValueError(f"session 创建失败: {key}")
        return meta

    def update_session(
        self,
        key: str,
        *,
        metadata: dict[str, Any] | None = None,
        last_consolidated: int | None = None,
        last_user_at: str | None = None,
        last_proactive_at: str | None = None,
    ) -> dict[str, Any] | None:
        set_parts = ["updated_at = ?"]
        params: list[Any] = [datetime.now().astimezone().isoformat()]
        if metadata is not None:
            set_parts.append("metadata = ?")
            params.append(json.dumps(metadata, ensure_ascii=False))
        if last_consolidated is not None:
            set_parts.append("last_consolidated = ?")
            params.append(int(last_consolidated))
        if last_user_at is not None:
            set_parts.append("last_user_at = ?")
            params.append(last_user_at)
        if last_proactive_at is not None:
            set_parts.append("last_proactive_at = ?")
            params.append(last_proactive_at)
        params.append(key)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE sessions SET {', '.join(set_parts)} WHERE key = ?",
                tuple(params),
            )
            self._conn.commit()
        if cur.rowcount <= 0:
            return None
        return self.get_session_meta(key)

    def delete_session(self, key: str, *, cascade: bool = False) -> bool:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._require_sessions_not_admitted_locked([key])
                if not cascade:
                    row = self._conn.execute(
                        """
                        SELECT
                            (SELECT COUNT(1) FROM messages WHERE session_key = ?) +
                            (SELECT COUNT(1) FROM turns WHERE session_key = ?) AS c
                        """,
                        (key, key),
                    ).fetchone()
                    count = int((row["c"] if row else 0) or 0)
                    if count > 0:
                        raise ValueError(
                            "session 下仍有 messages 或 turns，需使用 cascade 删除"
                        )
                else:
                    if self._has_message_embeddings_locked():
                        self._conn.execute(
                            """
                            DELETE FROM message_embeddings
                            WHERE message_id IN (
                                SELECT id FROM messages WHERE session_key = ?
                            )
                            """,
                            (key,),
                        )
                    self._conn.execute(
                        "DELETE FROM messages WHERE session_key = ?",
                        (key,),
                    )
                    self._conn.execute(
                        "DELETE FROM turns WHERE session_key = ?",
                        (key,),
                    )
                cur = self._conn.execute(
                    "DELETE FROM sessions WHERE key = ?",
                    (key,),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return cur.rowcount > 0

    def delete_sessions_batch(self, keys: list[str], *, cascade: bool = False) -> int:
        clean_keys = [str(key).strip() for key in keys if str(key).strip()]
        if not clean_keys:
            return 0
        placeholders = ",".join("?" for _ in clean_keys)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._require_sessions_not_admitted_locked(clean_keys)
                if not cascade:
                    row = self._conn.execute(
                        f"""
                        SELECT
                            (SELECT COUNT(1) FROM messages
                             WHERE session_key IN ({placeholders})) +
                            (SELECT COUNT(1) FROM turns
                             WHERE session_key IN ({placeholders})) AS c
                        """,
                        tuple([*clean_keys, *clean_keys]),
                    ).fetchone()
                    count = int((row["c"] if row else 0) or 0)
                    if count > 0:
                        raise ValueError(
                            "选中的 session 中仍有 messages 或 turns，需使用 cascade 删除"
                        )
                else:
                    if self._has_message_embeddings_locked():
                        self._conn.execute(
                            f"""
                            DELETE FROM message_embeddings
                            WHERE message_id IN (
                                SELECT id
                                FROM messages
                                WHERE session_key IN ({placeholders})
                            )
                            """,
                            tuple(clean_keys),
                        )
                    self._conn.execute(
                        f"DELETE FROM messages WHERE session_key IN ({placeholders})",
                        tuple(clean_keys),
                    )
                    self._conn.execute(
                        f"DELETE FROM turns WHERE session_key IN ({placeholders})",
                        tuple(clean_keys),
                    )
                cur = self._conn.execute(
                    f"DELETE FROM sessions WHERE key IN ({placeholders})",
                    tuple(clean_keys),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return int(cur.rowcount or 0)

    def acquire_session_admission(self, key: str, admission_id: str) -> bool:
        """仅在会话仍存在时创建处理租约，并阻止并发删除。"""

        # 1. 用写事务串行化“存在校验”和租约创建
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT 1 FROM sessions WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False

                # 2. 租约落库后，其他连接的删除操作才能继续竞争写锁
                self._conn.execute(
                    """
                    INSERT INTO session_admissions(admission_id, session_key, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (admission_id, key, datetime.now(UTC).isoformat()),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return True

    def release_session_admission(self, admission_id: str) -> None:
        """释放已完成入站消息持有的会话处理租约。"""

        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM session_admissions WHERE admission_id = ?",
                (admission_id,),
            )
            self._conn.commit()
        if cur.rowcount != 1:
            raise RuntimeError(f"session admission 不存在: {admission_id}")

    def clear_session_admissions(self) -> None:
        """在唯一 runtime 启动时清理上次异常退出遗留的处理租约。"""

        with self._lock:
            self._conn.execute("DELETE FROM session_admissions")
            self._conn.commit()

    def _require_sessions_not_admitted_locked(self, keys: list[str]) -> None:
        placeholders = ",".join("?" for _ in keys)
        row = self._conn.execute(
            f"""
            SELECT session_key
            FROM session_admissions
            WHERE session_key IN ({placeholders})
            LIMIT 1
            """,
            tuple(keys),
        ).fetchone()
        if row is not None:
            raise ValueError(
                f"session 正在处理消息，暂时不能删除: {row['session_key']}"
            )

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
            rows = self._conn.execute("""
                SELECT key, last_user_at, last_proactive_at
                FROM sessions
                WHERE last_user_at IS NOT NULL OR last_proactive_at IS NOT NULL
                """).fetchall()
        return {
            str(row["key"]): {
                "last_user_at": row["last_user_at"],
                "last_proactive_at": row["last_proactive_at"],
            }
            for row in rows
        }

    def most_recent_user_at(self) -> str | None:
        with self._lock:
            row = self._conn.execute("""
                SELECT MAX(last_user_at) AS last_user_at
                FROM sessions
                WHERE last_user_at IS NOT NULL
                """).fetchone()
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
                    "metadata": _decode_session_metadata(row["metadata"], key),
                }
            )
        return results

    def count_messages(self, session_key: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(1) AS c FROM messages WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        return int((row["c"] if row else 0) or 0)

    def _next_seq_locked(self, session_key: str) -> int:
        meta = self._conn.execute(
            "SELECT next_seq FROM sessions WHERE key = ?",
            (session_key,),
        ).fetchone()
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq) + 1, 0) AS next_seq FROM messages WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        from_messages = int((row["next_seq"] if row else 0) or 0)
        if meta is None:
            return from_messages
        return max(int(meta["next_seq"] or 0), from_messages)

    def next_seq(self, session_key: str) -> int:
        with self._lock:
            return self._next_seq_locked(session_key)

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
            json.dumps(tool_chain, ensure_ascii=False)
            if tool_chain is not None
            else None
        )
        extra_payload = json.dumps(extra or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO messages (id, session_key, seq, role, content, tool_chain, extra, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_key,
                    seq,
                    role,
                    content,
                    tool_chain_payload,
                    extra_payload,
                    ts,
                ),
            )
            self._conn.execute(
                """
                UPDATE sessions
                SET next_seq = CASE WHEN next_seq < ? THEN ? ELSE next_seq END
                WHERE key = ?
                """,
                (int(seq) + 1, int(seq) + 1, session_key),
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

    def _prepare_message_batch(
        self,
        key: str,
        *,
        start_seq: int,
        messages: list[dict[str, Any]],
    ) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
        """序列化待写消息并构造提交成功后的内存行。"""

        insert_rows: list[tuple[Any, ...]] = []
        result_rows: list[dict[str, Any]] = []

        # 1. 先完成序列化，失败时不改变数据库和内存消息。
        for offset, message in enumerate(messages):
            seq = start_seq + offset
            message_id = f"{key}:{seq}"
            tool_chain = message.get("tool_chain")
            extra = message["extra"]
            tool_chain_payload = (
                json.dumps(tool_chain, ensure_ascii=False)
                if tool_chain is not None
                else None
            )
            insert_rows.append(
                (
                    message_id,
                    key,
                    seq,
                    str(message["role"]),
                    str(message["content"]),
                    tool_chain_payload,
                    json.dumps(extra, ensure_ascii=False),
                    str(message["timestamp"]),
                )
            )
            row = {
                "id": message_id,
                "session_key": key,
                "seq": seq,
                "role": str(message["role"]),
                "content": str(message["content"]),
                "timestamp": str(message["timestamp"]),
            }
            if tool_chain is not None:
                row["tool_chain"] = tool_chain
            row.update(extra)
            result_rows.append(row)
        return insert_rows, result_rows

    def persist_session(
        self,
        key: str,
        *,
        created_at: str,
        updated_at: str,
        last_consolidated: int,
        metadata: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """原子更新 session 元数据并追加新消息。"""

        metadata_payload = json.dumps(metadata, ensure_ascii=False)
        result_rows: list[dict[str, Any]] = []

        # 1. 元数据和消息必须同成同败，避免磁盘留下半个 turn。
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    """
                    INSERT INTO sessions (key, created_at, updated_at, last_consolidated, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        last_consolidated = excluded.last_consolidated,
                        metadata = excluded.metadata
                    """,
                    (key, created_at, updated_at, int(last_consolidated), metadata_payload),
                )
                if messages:
                    start_seq = self._next_seq_locked(key)
                    insert_rows, result_rows = self._prepare_message_batch(
                        key,
                        start_seq=start_seq,
                        messages=messages,
                    )
                    self._conn.executemany(
                        """
                        INSERT INTO messages (id, session_key, seq, role, content, tool_chain, extra, ts)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        insert_rows,
                    )
                    next_seq = start_seq + len(insert_rows)
                    self._conn.execute(
                        """
                        UPDATE sessions
                        SET next_seq = CASE WHEN next_seq < ? THEN ? ELSE next_seq END
                        WHERE key = ?
                        """,
                        (next_seq, next_seq, key),
                    )
        return result_rows

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

    def list_messages_for_dashboard(
        self,
        *,
        session_key: str | None = None,
        q: str = "",
        role: str = "",
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "ts",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 200))
        offset = (safe_page - 1) * safe_page_size
        safe_sort = "ASC" if str(sort_order).lower() == "asc" else "DESC"
        safe_sort_by = (
            sort_by if sort_by in {"ts", "seq", "role", "session_key"} else "ts"
        )

        params: list[Any] = []
        where_parts: list[str] = []
        if session_key:
            where_parts.append("session_key = ?")
            params.append(session_key)
        term = (q or "").strip()
        if term:
            where_parts.append("content LIKE ?")
            params.append(f"%{term}%")
        if role:
            where_parts.append("role = ?")
            params.append(role)
        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        count_sql = f"SELECT COUNT(1) AS c FROM messages {where_sql}"
        data_sql = f"""
            SELECT id, session_key, seq, role, content, tool_chain, extra, ts
            FROM messages
            {where_sql}
            ORDER BY {safe_sort_by} {safe_sort}, seq {safe_sort}, id ASC
            LIMIT ? OFFSET ?
        """
        with self._lock:
            count_row = self._conn.execute(count_sql, tuple(params)).fetchone()
            rows = self._conn.execute(
                data_sql,
                tuple([*params, safe_page_size, offset]),
            ).fetchall()
        total = int((count_row["c"] if count_row else 0) or 0)
        return [self._row_to_message(row) for row in rows], total

    def media_path_exists(self, path: str | Path) -> bool:
        target = _resolve_path_text(path)
        if not target:
            return False
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, extra FROM messages WHERE extra LIKE ?",
                ('%"media"%',),
            ).fetchall()
        for row in rows:
            message_id = str(row["id"])
            extra = _decode_message_extra(row["extra"], message_id)
            if "media" not in extra:
                continue
            media = extra["media"]
            if not isinstance(media, list) or not all(
                isinstance(item, str) for item in media
            ):
                raise ValueError(f"message media 必须是字符串数组: {message_id}")
            for item in cast(list[str], media):
                if _resolve_path_text(item) == target:
                    return True
        return False

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_message(row)

    def get_message_by_client_id(
        self,
        session_key: str,
        client_message_id: str,
    ) -> dict[str, Any] | None:
        """按会话和移动端消息 ID 解析唯一的 canonical 消息。"""

        # 1. client_message_id 只存在于受校验的 extra JSON 中
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE session_key = ?
                  AND json_extract(extra, '$.client_message_id') = ?
                ORDER BY seq ASC
                LIMIT 2
                """,
                (session_key, client_message_id),
            ).fetchall()

        # 2. 重复标识违反移动消息幂等契约，不能静默选中其中一条
        if len(rows) > 1:
            raise RuntimeError(
                f"同一会话存在重复 client_message_id: {session_key} {client_message_id}"
            )
        return None if not rows else self._row_to_message(rows[0])

    def update_message(
        self,
        message_id: str,
        *,
        role: str | None = None,
        content: str | None = None,
        tool_chain: Any | None = None,
        extra: dict[str, Any] | None = None,
        ts: str | None = None,
    ) -> dict[str, Any] | None:
        set_parts: list[str] = []
        params: list[Any] = []
        if role is not None:
            set_parts.append("role = ?")
            params.append(role)
        if content is not None:
            set_parts.append("content = ?")
            params.append(content)
        if tool_chain is not None:
            set_parts.append("tool_chain = ?")
            params.append(json.dumps(tool_chain, ensure_ascii=False))
        if extra is not None:
            set_parts.append("extra = ?")
            params.append(json.dumps(extra, ensure_ascii=False))
        if ts is not None:
            set_parts.append("ts = ?")
            params.append(ts)
        if not set_parts:
            return self.get_message(message_id)

        with self._lock:
            row = self._conn.execute(
                "SELECT session_key, content FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return None
            session_key = str(row["session_key"])
            params.append(message_id)
            cur = self._conn.execute(
                f"UPDATE messages SET {', '.join(set_parts)} WHERE id = ?",
                tuple(params),
            )
            if content is not None and content != str(row["content"] or ""):
                self._delete_message_embeddings_locked([message_id])
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE key = ?",
                (datetime.now().astimezone().isoformat(), session_key),
            )
            self._conn.commit()
        if cur.rowcount <= 0:
            return None
        return self.get_message(message_id)

    def delete_message(self, message_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT session_key FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return False
            session_key = str(row["session_key"])
            cur = self._conn.execute(
                "DELETE FROM messages WHERE id = ?",
                (message_id,),
            )
            self._delete_message_embeddings_locked([message_id])
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE key = ?",
                (datetime.now().astimezone().isoformat(), session_key),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def delete_messages_batch(self, ids: list[str]) -> int:
        clean_ids = [
            str(message_id).strip() for message_id in ids if str(message_id).strip()
        ]
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        now = datetime.now().astimezone().isoformat()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT DISTINCT session_key FROM messages WHERE id IN ({placeholders})",
                tuple(clean_ids),
            ).fetchall()
            cur = self._conn.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                tuple(clean_ids),
            )
            self._delete_message_embeddings_locked(clean_ids)
            for row in rows:
                self._conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE key = ?",
                    (now, str(row["session_key"])),
                )
            self._conn.commit()
        return int(cur.rowcount or 0)

    def delete_session_messages_and_update_cursor(
        self,
        session_key: str,
        *,
        ids: list[str],
        last_consolidated: int,
    ) -> int:
        clean_ids = [
            str(message_id).strip() for message_id in ids if str(message_id).strip()
        ]
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        now = datetime.now().astimezone().isoformat()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                seq_rows = self._conn.execute(
                    f"""
                    SELECT id, seq
                    FROM messages
                    WHERE session_key = ? AND id IN ({placeholders})
                    """,
                    tuple([session_key, *clean_ids]),
                ).fetchall()
                next_seq = (
                    max(int(row["seq"]) for row in seq_rows) + 1 if seq_rows else 0
                )
                deleted_ids = [str(row["id"]) for row in seq_rows]
                cur = self._conn.execute(
                    f"""
                    DELETE FROM messages
                    WHERE session_key = ? AND id IN ({placeholders})
                    """,
                    tuple([session_key, *clean_ids]),
                )
                self._delete_message_embeddings_locked(deleted_ids)
                self._conn.execute(
                    """
                    UPDATE sessions
                    SET last_consolidated = ?,
                        updated_at = ?,
                        next_seq = CASE WHEN next_seq < ? THEN ? ELSE next_seq END
                    WHERE key = ?
                    """,
                    (int(last_consolidated), now, next_seq, next_seq, session_key),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return int(cur.rowcount or 0)

    def fetch_by_ids_with_context(
        self, ids: list[str], context: int
    ) -> list[dict[str, Any]]:
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
                        count_row = self._conn.execute(
                            count_sql, tuple(count_params)
                        ).fetchone()
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
        """校验并反序列化一行消息，拒绝损坏的 JSON 载荷。"""

        message_id = str(row["id"])
        role = row["role"]
        if role not in {"user", "assistant"}:
            raise ValueError(f"message role 无效: {message_id}")
        content = row["content"]
        if not isinstance(content, str):
            raise ValueError(f"message content 必须是字符串: {message_id}")
        timestamp = row["ts"]
        if not isinstance(timestamp, str):
            raise ValueError(f"message timestamp 必须是字符串: {message_id}")
        message: dict[str, Any] = {
            "id": message_id,
            "session_key": row["session_key"],
            "seq": int(row["seq"]),
            "role": role,
            "content": content,
            "timestamp": timestamp,
        }
        raw_tool_chain = row["tool_chain"]
        if raw_tool_chain is not None:
            tool_chain = _decode_message_tool_chain(raw_tool_chain, message_id)
            if tool_chain:
                message["tool_chain"] = tool_chain
        extra_dict = _decode_message_extra(row["extra"], message_id)
        if extra_dict:
            message.update(extra_dict)
        return message

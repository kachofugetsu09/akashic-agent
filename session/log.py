from __future__ import annotations

# 三个窄接口在同一存储模块内共享实现，不向消费者公开 connection。
# pyright: reportPrivateUsage=false

import asyncio
import json
import re
import sqlite3
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from session.message import (
    Body,
    CallRef,
    ContentPart,
    Control,
    Input,
    Message,
    Output,
    ToolCall,
    ToolResult,
    freeze_json,
)
from session.message_codec import decode_body, encode_body, json_value

_SCHEMA = {
    "sessions": """CREATE TABLE IF NOT EXISTS sessions (
                        key TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT,
                        next_seq INTEGER NOT NULL DEFAULT 0
                    );""",
    "messages": """CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_key TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        ts TEXT NOT NULL,
                        author TEXT NOT NULL,
                        source TEXT NOT NULL,
                        body TEXT NOT NULL,
                        UNIQUE(session_key, seq)
                    );""",
    "bindings": """CREATE TABLE IF NOT EXISTS bindings (
                        binding_id TEXT PRIMARY KEY,
                        descriptor TEXT NOT NULL
                    );""",
    "message_bindings": """CREATE TABLE IF NOT EXISTS message_bindings (
                        message_id TEXT NOT NULL REFERENCES messages(id),
                        binding_id TEXT NOT NULL REFERENCES bindings(binding_id),
                        PRIMARY KEY(message_id, binding_id)
                    );""",
    "message_call_result": """CREATE UNIQUE INDEX IF NOT EXISTS message_call_result
                    ON messages (
                        json_extract(body, '$.call_ref.message_id'),
                        json_extract(body, '$.call_ref.part_index')
                    ) WHERE json_extract(body, '$.kind')='tool_result';""",
}

_LEGACY_SESSION_SCHEMA = """CREATE TABLE sessions (
    key TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    last_consolidated INTEGER NOT NULL DEFAULT 0, metadata TEXT,
    last_user_at TEXT, last_proactive_at TEXT, next_seq INTEGER NOT NULL DEFAULT 0
)"""


def _sql(value: str) -> str:
    """只归一化 SQL 排版与标识符，保留字符串内的大小写和空白。"""
    tokens = re.findall(
        r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|[A-Za-z_][A-Za-z_0-9]*|[^\s]", value
    )
    words = [
        token if token.startswith("'") else token.strip('"').lower() for token in tokens
    ]
    for index in range(len(words) - 2):
        if words[index : index + 3] == ["if", "not", "exists"]:
            del words[index : index + 3]
            break
    return "".join(words).rstrip(";")


def _check_schema(connection: sqlite3.Connection) -> None:
    """启动前核对表与约束，不能把同列名的异构库当作已经迁移。"""
    for name, statement in _SCHEMA.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name=?",
            (name,),
        ).fetchone()
        if row is None:
            continue
        allowed = {_sql(statement)}
        if name == "sessions":
            allowed.add(_sql(_LEGACY_SESSION_SCHEMA))
        if _sql(row["sql"]) not in allowed:
            raise RuntimeError(f"{name} schema 不匹配，请先完成对应 yoyo 迁移")


class MessageConflict(ValueError):
    """消息身份重用或来源前缀已经变化。"""


class WriterExpired(RuntimeError):
    """任务已释放写入权，不能再提交新的输出。"""


class MessageLog:
    """SQLite 消息权威存储；只向消费者分配窄 reader/writer。"""

    def __init__(self, path: str | Path):
        self._lock = threading.RLock()
        self._closed = False
        self._listeners: dict[asyncio.Event, asyncio.AbstractEventLoop] = {}
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        _ = self._connection.execute("PRAGMA foreign_keys=ON")
        try:
            _check_schema(self._connection)
            with self._connection:
                for statement in _SCHEMA.values():
                    _ = self._connection.execute(statement)
        except BaseException:
            self._connection.close()
            raise

    def reader(self, session_id: str) -> MessageReader:
        return MessageReader(self, session_id)

    def writer(
        self,
        session_id: str,
        *,
        author: str,
        source: str,
        body_types: tuple[
            type[Input] | type[Output] | type[ToolResult] | type[Control], ...
        ],
        content: Mapping[str, Callable[[ContentPart], tuple[str, ...]]],
        call_ref: CallRef | None = None,
        check_call: Callable[[ToolCall], None] | None = None,
    ) -> MessageWriter:
        """组合签发纯校验函数；内容校验返回耐久 binding 引用，不得产生写入或外部效果。"""
        if ToolResult in body_types and call_ref is None:
            raise ValueError("工具结果 writer 必须绑定具体 call_ref")
        return MessageWriter(
            self,
            session_id,
            author,
            source,
            body_types,
            dict(content),
            call_ref,
            check_call,
        )

    def save_binding(self, binding_id: str, descriptor: Mapping[str, object]) -> None:
        """保存不可变资源绑定；重复身份必须仍描述同一实际实现。"""
        if not isinstance(binding_id, str) or not binding_id:
            raise ValueError("资源身份不能为空")
        payload = json.dumps(
            json_value(freeze_json(descriptor)),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        with self._lock, self._connection:
            _ = self._connection.execute("BEGIN IMMEDIATE")
            row = self._connection.execute(
                "SELECT descriptor FROM bindings WHERE binding_id=?",
                (binding_id,),
            ).fetchone()
            if row is not None:
                if row[0] != payload:
                    raise MessageConflict("资源身份已用于另一份 descriptor")
                return
            _ = self._connection.execute(
                "INSERT INTO bindings VALUES (?, ?)", (binding_id, payload)
            )

    def close(self) -> None:
        """释放数据库并唤醒所有追赶者，让它们正常退出。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()
            for event, loop in self._listeners.items():
                _ = loop.call_soon_threadsafe(event.set)


class MessageReader:
    def __init__(self, log: MessageLog, session_id: str):
        self._log = log
        self._session_id = session_id

    def read(
        self,
        *,
        after_seq: int = -1,
        through_seq: int | None = None,
        source: str | None = None,
        limit: int = 1000,
    ) -> tuple[Message, ...]:
        """读取固定序号范围，不创建 Session，也不持有修改或发送能力。"""
        if limit < 1 or after_seq < -1:
            raise ValueError("读取需要正 limit 和不小于 -1 的 after_seq")
        sql = "SELECT * FROM messages WHERE session_key=? AND seq>?"
        values: list[object] = [self._session_id, after_seq]
        if through_seq is not None:
            sql += " AND seq<=?"
            values.append(through_seq)
        if source is not None:
            sql += " AND source=?"
            values.append(source)
        sql += " ORDER BY seq LIMIT ?"
        values.append(limit)
        with self._log._lock:
            rows = self._log._connection.execute(sql, values).fetchall()
        return tuple(_message(row) for row in rows)

    def head(self, *, source: str | None = None) -> int:
        sql = "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE session_key=?"
        values = [self._session_id]
        if source is not None:
            sql += " AND source=?"
            values.append(source)
        with self._log._lock:
            return self._log._connection.execute(sql, values).fetchone()[0]

    async def follow(self, *, after_seq: int = -1) -> AsyncIterator[Message]:
        """先订阅再从日志追赶；通知只唤醒，正文和进度始终来自 seq。"""
        event = asyncio.Event()
        with self._log._lock:
            if self._log._closed:
                return
            self._log._listeners[event] = asyncio.get_running_loop()
        try:
            while True:
                event.clear()
                with self._log._lock:
                    if self._log._closed:
                        return
                    messages = self.read(after_seq=after_seq)
                if not messages:
                    _ = await event.wait()
                    continue
                for message in messages:
                    after_seq = message.seq
                    yield message
        finally:
            with self._log._lock:
                del self._log._listeners[event]


class MessageWriter:
    def __init__(
        self,
        log: MessageLog,
        session_id: str,
        author: str,
        source: str,
        body_types: tuple[type[Body], ...],
        content: Mapping[str, Callable[[ContentPart], tuple[str, ...]]],
        call_ref: CallRef | None,
        check_call: Callable[[ToolCall], None] | None,
    ):
        self._log: MessageLog = log
        self._session_id: str = session_id
        self._author: str = author
        self._source: str = source
        self._body_types = body_types
        self._content = content
        self._call_ref = call_ref
        self._check_call = check_call
        self._active = True

    def expire(self) -> None:
        with self._log._lock:
            self._active = False

    def append(
        self, message_id: str, body: Body, *, expected_source_head: int | None = None
    ) -> Message:
        """先处理同 ID 重放，再条件提交一条消息与其耐久资源引用。"""
        # 1. 固定 writer 的能力范围；内容 schema 由其注册 owner 验证。
        if type(body) not in self._body_types:
            raise PermissionError("writer 未获授该消息类型")
        if isinstance(body, ToolResult) and body.call_ref != self._call_ref:
            raise PermissionError("工具结果不属于 writer 获授的调用")
        payload = encode_body(body)
        connection = self._log._connection
        with self._log._lock, connection:
            _ = connection.execute("BEGIN IMMEDIATE")
            old = connection.execute(
                "SELECT * FROM messages WHERE id=?", (message_id,)
            ).fetchone()
            if old is not None:
                if (old["session_key"], old["author"], old["source"], old["body"]) != (
                    self._session_id,
                    self._author,
                    self._source,
                    payload,
                ):
                    raise MessageConflict("message_id 已用于不同的不可变内容")
                return _message(old)
            if not self._active:
                raise WriterExpired("writer 已失效")
            head = connection.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE session_key=? AND source=?",
                (self._session_id, self._source),
            ).fetchone()[0]
            if expected_source_head is not None and head != expected_source_head:
                raise MessageConflict(
                    f"来源 head 已变化: {head} != {expected_source_head}"
                )
            binding_ids = self._check_parts(body)
            if isinstance(body, ToolResult):
                self._check_call_result(body)

            # 2. Session 自己分配不复用的序号；Control 不得指向尚未接纳的前缀。
            now = datetime.now(UTC)
            stamp = now.isoformat()
            _ = connection.execute(
                "INSERT OR IGNORE INTO sessions (key, created_at, updated_at, next_seq) VALUES (?, ?, ?, 0)",
                (self._session_id, stamp, stamp),
            )
            seq = connection.execute(
                "SELECT next_seq FROM sessions WHERE key=?", (self._session_id,)
            ).fetchone()[0]
            message = Message(
                message_id, self._session_id, seq, now, self._author, self._source, body
            )
            _ = connection.execute(
                "INSERT INTO messages (id,session_key,seq,ts,author,source,body) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    self._session_id,
                    seq,
                    stamp,
                    self._author,
                    self._source,
                    payload,
                ),
            )
            for binding_id in binding_ids:
                _ = connection.execute(
                    "INSERT INTO message_bindings VALUES (?, ?)",
                    (message_id, binding_id),
                )
            _ = connection.execute(
                "UPDATE sessions SET next_seq=?, updated_at=? WHERE key=?",
                (seq + 1, stamp, self._session_id),
            )

        # 3. commit 后才唤醒读者；丢失此通知仍能从日志重新追赶。
        with self._log._lock:
            for event, loop in self._log._listeners.items():
                _ = loop.call_soon_threadsafe(event.set)
        return message

    def _check_parts(self, body: Body) -> set[str]:
        """只为新提交验证内容和调用 grant；已提交身份的重放直接返回收据。"""
        bindings: set[str] = set()
        if not isinstance(body, Control):
            for part in body.parts:
                if isinstance(part, ToolCall):
                    if self._check_call is None:
                        raise PermissionError("writer 未获授工具调用提出权")
                    self._check_call(part)
                    bindings.add(part.binding_id)
                else:
                    try:
                        validator = self._content[part.kind]
                    except KeyError as exc:
                        raise PermissionError(
                            f"writer 未获授内容类型 {part.kind}"
                        ) from exc
                    references = validator(part)
                    if not isinstance(references, tuple) or any(
                        not isinstance(ref, str) or not ref for ref in references
                    ):
                        raise TypeError(
                            "内容 owner 必须返回非空 binding ID 的 tuple，无引用返回 ()"
                        )
                    bindings.update(references)
        return bindings

    def _check_call_result(self, body: ToolResult) -> None:
        """在提交事务内校验调用地址与唯一结果，结果 writer 不得跨来源写入。"""
        connection = self._log._connection
        call = connection.execute(
            "SELECT * FROM messages WHERE id=?", (body.call_ref.message_id,)
        ).fetchone()
        if call is None or (call["session_key"], call["source"]) != (
            self._session_id,
            self._source,
        ):
            raise ValueError("调用不在 writer 获授的 Session/source 内")
        request = decode_body(call["body"])
        index = body.call_ref.part_index
        if (
            not isinstance(request, Output)
            or index >= len(request.parts)
            or not isinstance(request.parts[index], ToolCall)
        ):
            raise ValueError("call_ref 未指向真实工具调用")
        previous = connection.execute(
            "SELECT id FROM messages WHERE json_extract(body, '$.kind')='tool_result' "
            "AND json_extract(body, '$.call_ref.message_id')=? "
            "AND json_extract(body, '$.call_ref.part_index')=?",
            (body.call_ref.message_id, index),
        ).fetchone()
        if previous is not None:
            raise MessageConflict("该工具调用已经有结果消息")


def _message(row: sqlite3.Row) -> Message:
    return Message(
        row["id"],
        row["session_key"],
        row["seq"],
        datetime.fromisoformat(row["ts"]),
        row["author"],
        row["source"],
        decode_body(row["body"]),
    )

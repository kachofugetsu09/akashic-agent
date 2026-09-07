from __future__ import annotations

# 三个窄接口在同一存储模块内共享实现，不向消费者公开 connection。
# pyright: reportPrivateUsage=false

import asyncio
import json
import inspect
import re
import sqlite3
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from dataclasses import dataclass
from typing import TypeVar, cast
from pathlib import Path
from types import MappingProxyType

from session.artifacts import AttachmentKind, AttachmentRef
from session.message import (
    Body,
    CallRef,
    ContentPart,
    ContentReferences,
    Control,
    Input,
    Message,
    Output,
    ToolCall,
    ToolResult,
    freeze_json,
)
from session.message_codec import decode_body, encode_body, json_value

_T = TypeVar("_T")

_SCHEMA = {
    "attachments": """CREATE TABLE IF NOT EXISTS attachments (
        artifact_id TEXT PRIMARY KEY, storage_key TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL CHECK (kind IN ('image', 'file')),
        filename TEXT, media_type TEXT,
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        sha256 TEXT NOT NULL CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
        state TEXT NOT NULL CHECK (state = 'ready'), created_at TEXT NOT NULL
    );""",
    "message_attachments": """CREATE TABLE IF NOT EXISTS message_attachments (
        message_id TEXT NOT NULL, ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        artifact_id TEXT NOT NULL, PRIMARY KEY (message_id, ordinal),
        FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
        FOREIGN KEY (artifact_id) REFERENCES attachments(artifact_id)
    );""",
    "idx_message_attachments_artifact": """CREATE INDEX IF NOT EXISTS idx_message_attachments_artifact
        ON message_attachments(artifact_id, message_id, ordinal);""",
    "message_embeddings": """CREATE TABLE IF NOT EXISTS message_embeddings (
        message_id TEXT NOT NULL, content_hash TEXT NOT NULL,
        model TEXT NOT NULL, embedding BLOB NOT NULL, dim INTEGER NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (message_id, model)
    );""",
    "ix_message_embeddings_hash": """CREATE INDEX IF NOT EXISTS ix_message_embeddings_hash
        ON message_embeddings (content_hash, model);""",
    "owner_records": """CREATE TABLE IF NOT EXISTS owner_records (
        owner TEXT NOT NULL, key TEXT NOT NULL, version INTEGER NOT NULL,
        value TEXT NOT NULL, PRIMARY KEY(owner, key)
    );""",
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

_LEGACY_ATTACHMENT_SCHEMA = """CREATE TABLE message_attachments (
    message_id TEXT NOT NULL, ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    artifact_id TEXT NOT NULL, direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    PRIMARY KEY (message_id, ordinal),
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY (artifact_id) REFERENCES attachments(artifact_id)
)"""

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
        if name == "message_attachments":
            allowed.add(_sql(_LEGACY_ATTACHMENT_SCHEMA))
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
            fresh = (
                self._connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='messages'"
                ).fetchone()
                is None
            )
            with self._connection:
                for name, statement in _SCHEMA.items():
                    # 新库由 owner 初始化；已有库的新持久能力只能由 yoyo 接纳。
                    if name in {"owner_records", "message_embeddings", "ix_message_embeddings_hash",
                                "attachments", "message_attachments", "idx_message_attachments_artifact"} and not fresh:
                        continue
                    _ = self._connection.execute(statement)
        except BaseException:
            self._connection.close()
            raise

    def owner(self, name: str) -> OwnerStore:
        """组合只向 owner 授予自身的记录空间，不授予 SQL 或其他空间。"""
        if not isinstance(name, str) or not name:
            raise ValueError("状态 owner 不能为空")
        with self._lock:
            if (
                self._connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='owner_records'"
                ).fetchone()
                is None
            ):
                raise RuntimeError("owner_records 缺失，请先运行对应 yoyo 迁移")
        return OwnerStore(self, name)

    def _write(self, callback: Callable[[], _T]) -> _T:
        """同步事务不跨 await；全部权威写成功后才通知日志读者。"""
        with self._lock:
            if self._connection.in_transaction:
                raise RuntimeError("事务内写入必须使用当前 transaction 接口")
            with self._connection:
                _ = self._connection.execute("BEGIN IMMEDIATE")
                result = callback()
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("存储事务回调必须同步，不能跨 await")
            for event, loop in self._listeners.items():
                _ = loop.call_soon_threadsafe(event.set)
            return result

    def catalog(self) -> MessageCatalog:
        return MessageCatalog(self)

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
        content: Mapping[str, Callable[[ContentPart], ContentReferences]],
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

    def read_binding(self, binding_id: str) -> Mapping[str, object]:
        """读取不可变绑定；缺失引用不能用当前实现补齐。"""
        with self._lock:
            row = self._connection.execute(
                "SELECT descriptor FROM bindings WHERE binding_id=?", (binding_id,)
            ).fetchone()
        if row is None:
            raise KeyError(binding_id)
        value = json.loads(row[0])
        if not isinstance(value, dict):
            raise ValueError("binding descriptor 必须是对象")
        return cast(Mapping[str, object], freeze_json(cast(dict[str, object], value)))

    def close(self) -> None:
        """释放数据库并唤醒所有追赶者，让它们正常退出。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()
            for event, loop in self._listeners.items():
                _ = loop.call_soon_threadsafe(event.set)


class MessageCatalog:
    """只读会话目录；一次 heads 快照固定跨会话消费的消息上界。"""

    def __init__(self, log: MessageLog | None):
        self._storage = log

    @property
    def _log(self) -> MessageLog:
        if self._storage is None:
            raise RuntimeError("candidate 验证期禁止读取正式会话目录")
        return self._storage

    def snapshot_heads(self) -> Mapping[str, int]:
        """单条查询取得同一数据库快照，不逐会话读取可能变化的 head。"""
        with self._log._lock:
            rows = self._log._connection.execute(
                "SELECT s.key, COALESCE(MAX(m.seq), -1) AS head "
                "FROM sessions s LEFT JOIN messages m ON m.session_key=s.key "
                "GROUP BY s.key ORDER BY s.key"
            ).fetchall()
        return MappingProxyType({row["key"]: row["head"] for row in rows})

    def reader(self, session_id: str) -> MessageReader:
        return self._log.reader(session_id)

    async def follow(self) -> AsyncIterator[Mapping[str, int]]:
        """先订阅再取 heads；通知可合并，消费者始终按快照重读事实。"""
        event = asyncio.Event()
        with self._log._lock:
            if self._log._closed:
                return
            self._log._listeners[event] = asyncio.get_running_loop()
        previous: Mapping[str, int] | None = None
        try:
            while True:
                event.clear()
                with self._log._lock:
                    if self._log._closed:
                        return
                    heads = self.snapshot_heads()
                if heads != previous:
                    previous = heads
                    yield heads
                else:
                    _ = await event.wait()
        finally:
            with self._log._lock:
                del self._log._listeners[event]


class MessageReader:
    def __init__(self, log: MessageLog, session_id: str):
        self._log = log
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        return self._session_id

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

    def snapshot(self, *, through_seq: int | None = None) -> tuple[Message, ...]:
        """固定上界后分页读取完整前缀，供需要完整历史的只读投影消费。"""
        head = self.head() if through_seq is None else through_seq
        messages: list[Message] = []
        cursor = -1
        while cursor < head:
            page = self.read(after_seq=cursor, through_seq=head)
            if not page:
                break
            messages.extend(page)
            cursor = page[-1].seq
        return tuple(messages)

    def get(self, message_id: str) -> Message | None:
        """按不可变身份读取消息，不能跨 reader 获授的 Session。"""
        with self._log._lock:
            row = self._log._connection.execute(
                "SELECT * FROM messages WHERE id=? AND session_key=?",
                (message_id, self._session_id),
            ).fetchone()
        return None if row is None else _message(row)

    def attachments(self, message_id: str) -> tuple[AttachmentRef, ...]:
        """只读取已获授 Session 中该消息的有序附件引用，不暴露路径或任意 ID 查询。"""
        with self._log._lock:
            if self.get(message_id) is None:
                raise LookupError("消息不在 reader 获授的 Session 中")
            rows = self._log._connection.execute(
                "SELECT a.* FROM message_attachments ma JOIN attachments a ON a.artifact_id=ma.artifact_id "
                "WHERE ma.message_id=? ORDER BY ma.ordinal", (message_id,),
            ).fetchall()
        return tuple(_artifact_ref(row) for row in rows)

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
        content: Mapping[str, Callable[[ContentPart], ContentReferences]],
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

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def source(self) -> str:
        return self._source

    def check(self, body: Body) -> None:
        """预先核对提交权限与引用；不占序号，实际提交仍在事务内核对实时状态。"""
        with self._log._lock:
            self._check_grant(body)
            if not self._active:
                raise WriterExpired("writer 已失效")
            _ = self._check_parts(body)
            if isinstance(body, ToolResult):
                self._check_call_result(body)

    def _check_grant(self, body: Body) -> None:
        if type(body) not in self._body_types:
            raise PermissionError("writer 未获授该消息类型")
        if isinstance(body, ToolResult) and body.call_ref != self._call_ref:
            raise PermissionError("工具结果不属于 writer 获授的调用")

    def expire(self) -> None:
        with self._log._lock:
            self._active = False

    def append(
        self, message_id: str, body: Body, *, expected_source_head: int | None = None
    ) -> Message:
        """先处理同 ID 重放，再条件提交一条消息与其耐久资源引用。"""
        return self._log._write(
            lambda: self._append(
                message_id, body, expected_source_head=expected_source_head
            )
        )

    def _append(
        self, message_id: str, body: Body, *, expected_source_head: int | None = None
    ) -> Message:
        # 1. 固定 writer 的能力范围；内容 schema 由其注册 owner 验证。
        self._check_grant(body)
        payload = encode_body(body)
        connection = self._log._connection
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
            raise MessageConflict(f"来源 head 已变化: {head} != {expected_source_head}")
        binding_ids, artifacts = self._check_parts(body)
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
        for ordinal, artifact in enumerate(artifacts):
            _ = connection.execute(
                "INSERT INTO message_attachments (message_id,ordinal,artifact_id) VALUES (?,?,?)",
                (message_id, ordinal, artifact),
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

        return message

    def _check_parts(self, body: Body) -> tuple[set[str], tuple[str, ...]]:
        """只为新提交验证内容和调用 grant；已提交身份的重放直接返回收据。"""
        bindings: set[str] = set()
        artifacts: list[str] = []
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
                    if not isinstance(references, ContentReferences):
                        raise TypeError("内容 owner 必须返回 ContentReferences")
                    bindings.update(references.binding_ids)
                    artifacts.extend(references.artifact_ids)
        if artifacts:
            connection = self._log._connection
            row = connection.execute("SELECT sql FROM sqlite_master WHERE name='message_attachments'").fetchone()
            if row is None or _sql(row[0]) != _sql(_SCHEMA["message_attachments"]):
                raise RuntimeError("附件引用写入需要先完成对应 yoyo 迁移")
            for artifact_id in set(artifacts):
                row = connection.execute(
                    "SELECT state FROM attachments WHERE artifact_id=?", (artifact_id,)
                ).fetchone()
                if row is None or row["state"] != "ready":
                    raise ValueError("附件引用必须指向已发布的不可变资源")
        return bindings, tuple(artifacts)

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


@dataclass(frozen=True, slots=True)
class OwnerRecord:
    version: int
    value: Mapping[str, object]


class OwnerStore:
    """一个 owner 的窄持久接口；结果正文仍由 Message 独占。"""

    def __init__(self, log: MessageLog, owner: str):
        self._log = log
        self._owner = owner

    def check_access(self, *capabilities: MessageReader | MessageWriter) -> None:
        """在产生外部效果前确认获授能力可参与同一 authority 的事务。"""
        if any(capability._log is not self._log for capability in capabilities):
            raise ValueError("原子提交不能跨存储 authority")

    def read(self, key: str) -> OwnerRecord | None:
        with self._log._lock:
            row = self._log._connection.execute(
                "SELECT version,value FROM owner_records WHERE owner=? AND key=?",
                (self._owner, key),
            ).fetchone()
        return None if row is None else _owner_record(row)

    def list(self) -> tuple[tuple[str, OwnerRecord], ...]:
        with self._log._lock:
            rows = self._log._connection.execute(
                "SELECT key,version,value FROM owner_records WHERE owner=? ORDER BY key",
                (self._owner,),
            ).fetchall()
        return tuple((row["key"], _owner_record(row)) for row in rows)

    def transact(self, callback: Callable[[OwnerTransaction], _T]) -> _T:
        """把自身状态与已获授的 Message 写入放在一个同步事务内。"""
        transaction = OwnerTransaction(self)

        def invoke() -> _T:
            value = callback(transaction)
            transaction._check_active()
            return value

        try:
            return self._log._write(invoke)
        finally:
            transaction._active = False


class OwnerTransaction:
    def __init__(self, store: OwnerStore):
        self._store = store
        self._active = True
        self._failed = False

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("存储 transaction 已结束")
        if self._failed:
            raise RuntimeError("存储 transaction 已失败，必须回滚")

    def _perform(self, operation: Callable[[], _T]) -> _T:
        self._check_active()
        try:
            return operation()
        except BaseException:
            # 即使调用方捕获了部分 INSERT 后的错误，外层事务也必须整体回滚。
            self._failed = True
            raise

    def read(self, key: str) -> OwnerRecord | None:
        self._check_active()
        return self._store.read(key)

    def save(
        self, key: str, value: Mapping[str, object], *, expected_version: int | None
    ) -> OwnerRecord:
        """由 owner 校验领域状态；存储只拥有版本 CAS 和 JSON 边界。"""
        return self._perform(
            lambda: self._save(key, value, expected_version=expected_version)
        )

    def _save(
        self, key: str, value: Mapping[str, object], *, expected_version: int | None
    ) -> OwnerRecord:
        self._check_active()
        if not isinstance(key, str) or not key:
            raise ValueError("状态 key 不能为空")
        if expected_version is not None and (
            type(expected_version) is not int or expected_version < 0
        ):
            raise ValueError("状态版本必须是非负整数或 None")
        frozen = freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise TypeError("owner 状态必须是 JSON 对象")
        current = self.read(key)
        if (None if current is None else current.version) != expected_version:
            raise MessageConflict("owner 记录版本已变化")
        version = 0 if current is None else current.version + 1
        payload = json.dumps(
            json_value(cast(Mapping[str, object], frozen)),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        _ = self._store._log._connection.execute(
            "INSERT INTO owner_records VALUES (?,?,?,?) ON CONFLICT(owner,key) DO UPDATE SET version=excluded.version,value=excluded.value",
            (self._store._owner, key, version, payload),
        )
        return OwnerRecord(version, cast(Mapping[str, object], frozen))

    def append(
        self,
        writer: MessageWriter,
        message_id: str,
        body: Body,
        *,
        expected_source_head: int | None = None,
    ) -> Message:
        self._check_active()
        if writer._log is not self._store._log:
            raise ValueError("原子提交不能跨存储 authority")
        return self._perform(
            lambda: writer._append(
                message_id, body, expected_source_head=expected_source_head
            )
        )


def _owner_record(row: sqlite3.Row) -> OwnerRecord:
    if type(row["version"]) is not int or row["version"] < 0:
        raise ValueError("owner 记录版本无效")
    value = freeze_json(json.loads(row["value"]))
    if not isinstance(value, Mapping):
        raise ValueError("owner 记录不是 JSON 对象")
    return OwnerRecord(row["version"], cast(Mapping[str, object], value))


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


def _artifact_ref(row: sqlite3.Row) -> AttachmentRef:
    """数据库边界只接受已发布的完整附件元数据。"""
    if row["state"] != "ready":
        raise ValueError("附件尚未发布")
    return AttachmentRef(
        row["artifact_id"], AttachmentKind(row["kind"]), row["filename"],
        row["media_type"], row["size_bytes"], row["sha256"],
    )

from __future__ import annotations

# 三个窄接口在同一存储模块内共享实现，不向消费者公开 connection。
# pyright: reportPrivateUsage=false

import asyncio
import json
import inspect
import logging
import re
import sqlite3
import threading
from collections.abc import AsyncGenerator, Callable, Generator, Iterable, Iterator, Mapping, Sequence
from bisect import bisect_right
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from dataclasses import dataclass
from typing import Literal, TypeVar, cast, overload
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from weakref import WeakValueDictionary

from core.common.file_io import run_file_io
from session.artifacts import AttachmentKind, AttachmentRef
from session.artifact_store import ARTIFACT_SCHEMA
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
    freeze_metadata,
)
from session.message_codec import decode_body, encode_body, json_value

_T = TypeVar("_T")
_logger = logging.getLogger(__name__)

MESSAGE_SOURCE_INDEX_SCHEMA = """CREATE INDEX IF NOT EXISTS message_source_seq
    ON messages (session_key, source, seq);"""
# 小行强缓存：每轮重复读取的近期消息不再依赖调用者持有引用；上界约 512×8KB。
_DECODE_STRONG_LIMIT = 512
_DECODE_STRONG_BODY = 8192
_ATTACHMENT_MEMO_SIZE = 8192
MESSAGE_BODY_KIND_INDEX_SCHEMA = """CREATE INDEX IF NOT EXISTS message_source_kind_seq
    ON messages (session_key, source, json_extract(body, '$.kind'), seq,
                 json_extract(body, '$.finish'));"""


_SCOPE_DIMENSION = re.compile(r"[a-z][a-z0-9_]{0,31}")
_SCOPE_VALUE_LIMIT = 128


@dataclass(frozen=True, slots=True)
class SessionAttributes:
    """会话接纳时固定的独立事实；存储不替展示或学习消费者作决定。

    scope 是宽键中已声明的维度；缺失维度即 default，Core 不解释维度含义。
    """

    visibility: Literal["listed", "internal"] = "listed"
    learning: Literal["eligible", "excluded"] = "eligible"
    scope: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.visibility not in ("listed", "internal") or self.learning not in ("eligible", "excluded"):
            raise ValueError("Session 属性无效")
        names = [name for name, _ in self.scope]
        if names != sorted(set(names)):
            raise ValueError("Session scope 维度必须唯一且有序")
        for name, value in self.scope:
            if not isinstance(name, str) or _SCOPE_DIMENSION.fullmatch(name) is None:
                raise ValueError(f"Session scope 维度名无效: {name!r}")
            if (
                not isinstance(value, str) or not value or value == "default"
                or len(value) > _SCOPE_VALUE_LIMIT or value != value.strip()
            ):
                raise ValueError(f"Session scope 维度值无效: {name}")

    @classmethod
    def scoped(
        cls, dimensions: Mapping[str, str], *,
        visibility: Literal["listed", "internal"] = "listed",
        learning: Literal["eligible", "excluded"] = "eligible",
    ) -> SessionAttributes:
        return cls(visibility, learning, tuple(sorted(dimensions.items())))

    def dimension(self, name: str) -> str:
        """缺失维度按 default 解析；新增维度不需要迁移旧 Session。"""
        return dict(self.scope).get(name, "default")


@dataclass(frozen=True, slots=True)
class SessionEntry:
    """目录中的只读事实；不持有执行状态，也不替 UI 生成标题。"""

    session_id: str
    created_at: datetime
    updated_at: datetime
    attributes: SessionAttributes
    metadata: Mapping[str, object] | None
    head_seq: int
    message_count: int
    first_message: Message | None
    """显式标题覆盖；None 表示由表示边界按首条消息推导。"""
    title: str | None = None


@dataclass(frozen=True, slots=True)
class SessionPage:
    items: tuple[SessionEntry, ...]
    total: int
    next_cursor: tuple[str, str] | None


@dataclass(frozen=True, slots=True)
class MessagePage:
    """同一读取快照中的有序消息、引用和固定上界，不保存副本或消费进度。"""

    messages: tuple[Message, ...]
    attachments: Mapping[str, tuple[AttachmentRef, ...]]
    bindings: Mapping[str, Mapping[str, object]]
    through_seq: int
    has_more: bool


class InvalidPage(ValueError):
    """调用者的分页范围或游标无效；与持久记录损坏区分。"""


def encode_attributes(attributes: SessionAttributes) -> str:
    # 空 scope 不写键，已有 Session 行与默认列值保持逐字节一致。
    payload: dict[str, object] = {"visibility": attributes.visibility, "learning": attributes.learning}
    if attributes.scope:
        payload["scope"] = dict(attributes.scope)
    return json.dumps(payload, sort_keys=True)


def decode_attributes(raw: str) -> SessionAttributes:
    """属性只有一份固定 schema，不从会话名称或任意 metadata 猜测。"""
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        if len(pairs) != len({key for key, _ in pairs}):
            raise ValueError("Session 属性字段重复")
        return dict(pairs)
    value: object = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Session 属性必须是对象")
    data = cast(dict[str, object], value)
    if set(data) - {"scope"} != {"visibility", "learning"}:
        raise ValueError("Session 属性字段无效")
    raw_scope = data.get("scope", {})
    if not isinstance(raw_scope, dict) or ("scope" in data and not raw_scope):
        raise ValueError("Session scope 必须是非空对象")
    scope = cast(dict[str, object], raw_scope)
    if any(not isinstance(item, str) for item in scope.values()):
        raise ValueError("Session scope 维度值必须是字符串")
    return SessionAttributes(cast(Literal["listed", "internal"], data["visibility"]),
                             cast(Literal["eligible", "excluded"], data["learning"]),
                             tuple(sorted(cast(dict[str, str], scope).items())))


_OLD_SESSION_SCHEMA = """CREATE TABLE sessions (
    key TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    metadata TEXT, next_seq INTEGER NOT NULL DEFAULT 0
);"""
_SESSION_ATTRIBUTES_COLUMN = (
    "attributes TEXT NOT NULL DEFAULT '{\"learning\": \"eligible\", \"visibility\": \"listed\"}'"
)
# 软删除标记：NULL 表示正常，时间戳表示已逻辑失效；由 yoyo 只增加列接纳。
_SESSION_DELETED_COLUMN = "deleted_at TEXT"
# 标题覆盖：NULL 表示沿用首条消息推导，非空为显式管理状态；由 yoyo 只增加列接纳。
_SESSION_TITLE_COLUMN = "title TEXT"
_SESSION_TITLE_MAX = 200

_MESSAGE_METADATA_COLUMN = "metadata TEXT NOT NULL DEFAULT '{}'"

_MESSAGE_PREFIX_SCHEMA = {
    "message_prefix_revision": """CREATE TABLE IF NOT EXISTS message_prefix_revision (
        singleton INTEGER PRIMARY KEY CHECK (singleton=1),
        revision INTEGER NOT NULL CHECK (typeof(revision)='integer' AND revision>=0)
    );""",
    "message_prefix_update": """CREATE TRIGGER IF NOT EXISTS message_prefix_update
        AFTER UPDATE ON messages BEGIN
            UPDATE message_prefix_revision SET revision=revision+1 WHERE singleton=1;
        END;""",
    "message_prefix_delete": """CREATE TRIGGER IF NOT EXISTS message_prefix_delete
        AFTER DELETE ON messages BEGIN
            UPDATE message_prefix_revision SET revision=revision+1 WHERE singleton=1;
        END;""",
    # REPLACE 的隐式删除不保证触发 DELETE；插入前也检查被替换的身份。
    "message_prefix_insert": """CREATE TRIGGER IF NOT EXISTS message_prefix_insert
        BEFORE INSERT ON messages
        WHEN NEW.seq <= (SELECT MAX(seq) FROM messages WHERE session_key=NEW.session_key)
            OR EXISTS (SELECT 1 FROM messages WHERE id=NEW.id)
        BEGIN
            UPDATE message_prefix_revision SET revision=revision+1 WHERE singleton=1;
        END;""",
}


_SCHEMA = {
    "attachments": ARTIFACT_SCHEMA["attachments"],
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
    "sessions": f"""CREATE TABLE IF NOT EXISTS sessions (
                        key TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT,
                        next_seq INTEGER NOT NULL DEFAULT 0,
                        {_SESSION_ATTRIBUTES_COLUMN},
                        {_SESSION_DELETED_COLUMN},
                        {_SESSION_TITLE_COLUMN}
                    );""",
    "messages": f"""CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_key TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        ts TEXT NOT NULL,
                        author TEXT NOT NULL,
                        source TEXT NOT NULL,
                        body TEXT NOT NULL,
                        {_MESSAGE_METADATA_COLUMN},
                        UNIQUE(session_key, seq)
                    );""",
    "message_source_seq": MESSAGE_SOURCE_INDEX_SCHEMA,
    "message_source_kind_seq": MESSAGE_BODY_KIND_INDEX_SCHEMA,
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
    **_MESSAGE_PREFIX_SCHEMA,
}

_OLD_MESSAGE_SCHEMA = _SCHEMA["messages"].replace(
    "                        " + _MESSAGE_METADATA_COLUMN + ",\n", ""
)

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


def _session_schemas() -> Mapping[str, bool]:
    """保留两条已知旧表 lineage，已加管理列的同形库同样是已知身份。"""
    values = {_sql(_SCHEMA["sessions"]): True}
    for old in (_LEGACY_SESSION_SCHEMA, _OLD_SESSION_SCHEMA):
        values[_sql(old)] = False
        base = old.rstrip().rstrip(";").rstrip()
        for suffix in (
            ", " + _SESSION_ATTRIBUTES_COLUMN,
            ", " + _SESSION_ATTRIBUTES_COLUMN + ", " + _SESSION_DELETED_COLUMN,
            ", " + _SESSION_ATTRIBUTES_COLUMN + ", " + _SESSION_TITLE_COLUMN,
            ", " + _SESSION_ATTRIBUTES_COLUMN + ", " + _SESSION_DELETED_COLUMN
            + ", " + _SESSION_TITLE_COLUMN,
        ):
            values[_sql(base[:-1] + suffix + ")")] = True
    return values


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
        if name == "messages":
            # 已发布 yoyo 的中间步骤仍通过同一日志读取/追加无扩展消息。
            allowed.add(_sql(_OLD_MESSAGE_SCHEMA))
        if name == "sessions":
            allowed.update(_session_schemas())
        if name == "message_attachments":
            allowed.add(_sql(_LEGACY_ATTACHMENT_SCHEMA))
        if _sql(row["sql"]) not in allowed:
            raise RuntimeError(f"{name} schema 不匹配，请先完成对应 yoyo 迁移")


def create_message_source_index(connection: sqlite3.Connection) -> None:
    """Build a source prefix index without changing message rows or their order."""
    _check_schema(connection)
    _ = connection.execute(MESSAGE_SOURCE_INDEX_SCHEMA)
    _check_schema(connection)


def create_message_body_kind_index(connection: sqlite3.Connection) -> None:
    """Index body kinds so Input and Control lookups skip unrelated bodies."""
    _check_schema(connection)
    _ = connection.execute(MESSAGE_BODY_KIND_INDEX_SCHEMA)
    _check_schema(connection)


def create_message_prefix_revision(connection: sqlite3.Connection) -> None:
    """增加前缀失效标记；只由消息变更的同一事务推进，不改写消息。"""
    _check_schema(connection)
    for statement in _MESSAGE_PREFIX_SCHEMA.values():
        connection.execute(statement)
    connection.execute(
        "INSERT INTO message_prefix_revision VALUES (1,0) ON CONFLICT DO NOTHING"
    )
    _check_schema(connection)


class MessageConflict(ValueError):
    """消息身份、引用或来源前缀发生冲突。"""


class SourceHeadConflict(MessageConflict):
    """来源 head 的 CAS 失败，事务未提交；调用者可重新选择前缀。"""


class WriterExpired(RuntimeError):
    """任务已释放写入权，不能再提交新的输出。"""


@dataclass
class _ReadConnection:
    connection: sqlite3.Connection
    lock: threading.RLock
    heads_version: int | None = None
    heads: Mapping[str, int] | None = None


class _ReadLocal(threading.local):
    def __init__(self) -> None:
        self.current: _ReadConnection | None = None


class _BorrowedRead:
    """_read 的类实现：每轮百余次借用不再支付生成器 contextmanager 的开关成本。"""

    __slots__ = ("_log", "_snapshot", "_read_conn", "_connection", "_reentrant")

    def __init__(self, log: MessageLog, snapshot: bool) -> None:
        self._log = log
        self._snapshot = snapshot
        self._read_conn: _ReadConnection | None = None
        self._connection: sqlite3.Connection | None = None
        self._reentrant = False

    def __enter__(self) -> sqlite3.Connection:
        log = self._log
        current = log._reads.current
        if current is not None:
            self._reentrant = True
            return current.connection
        # 1. 重入当前线程的写事务；另一个线程持有 writer 时直接读已提交快照。
        if log._writer_lock.acquire(blocking=False):
            try:
                if log._closed:
                    raise RuntimeError("MessageLog is closed")
                if log._writer_connection.in_transaction:
                    self._reentrant = True
                    return log._writer_connection
            finally:
                log._writer_lock.release()
        # 2. 只读准入与 close 共享短锁，不等待写事务的磁盘或内容校验。
        with log._read_admission:
            if log._closed:
                raise RuntimeError("MessageLog is closed")
            if log._idle_reads:
                read = log._idle_reads.pop()
            else:
                connection = sqlite3.connect(
                    log._path.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
                try:
                    connection.row_factory = sqlite3.Row
                    _ = connection.execute("PRAGMA query_only=ON")
                except BaseException:
                    connection.close()
                    raise
                read = _ReadConnection(connection, threading.RLock())
            connection = read.connection
            try:
                if self._snapshot:
                    _ = connection.execute("BEGIN")
            except BaseException:
                connection.close()
                raise
        # Callbacks may use other readers of this same log; all see this snapshot.
        self._read_conn = read
        self._connection = connection
        log._reads.current = read
        return connection

    def __exit__(self, exc_type: object, exc: object, tb: object) -> Literal[False]:
        if self._reentrant:
            return False
        log = self._log
        connection = self._connection
        read = self._read_conn
        assert connection is not None and read is not None
        log._reads.current = None
        try:
            # 归还前结束快照；下次借用必须观察新的已提交状态。
            if self._snapshot:
                connection.rollback()
        except BaseException:
            connection.close()
            raise
        with log._read_admission:
            if not log._closed and len(log._idle_reads) < 4:
                log._idle_reads.append(read)
            else:
                connection.close()
        return False


class MessageLog:
    """SQLite 消息权威存储；只向消费者分配窄 reader/writer。"""

    def __init__(self, path: str | Path):
        self._writer_lock = threading.RLock()
        self._read_admission = threading.Lock()
        self._idle_reads: list[_ReadConnection] = []
        self._listener_lock = threading.Lock()
        self._decode_lock = threading.RLock()
        self._view_lock = threading.Lock()
        self._message_views: WeakValueDictionary[str, _MessagePrefix] = WeakValueDictionary()
        self._decoded: WeakValueDictionary[tuple[object, ...], Message] = WeakValueDictionary()
        self._decoded_strong: dict[tuple[object, ...], Message] = {}
        self._decoded_owners: dict[tuple[object, ...], OwnerRecord] = {}
        self._decoded_attributes: dict[str, SessionAttributes] = {}
        # 已提交消息的附件绑定只随消息同事务写入、本层不再变更；按 message_id
        # 备忘查询结果，每轮材料准备只读取新增消息，由 _decode_lock 保护。
        # 命名数据管理操作（撤销/删除 Session）经另一连接物理删除后，必须经
        # invalidate_attachment_memo 显式失效对应项。
        self._attachment_memo: dict[str, tuple[AttachmentRef, ...]] = {}
        self._reads = _ReadLocal()
        self._path = Path(path).resolve()
        self._closed = False
        self._notify_pending: set[type[Body]] | None = set()
        self._notify_owed: set[type[Body]] | None = set()
        self._defer_notify = threading.local()
        self._notify_in_flight: set[asyncio.Event] = set()
        self._listeners: dict[asyncio.Event, tuple[asyncio.AbstractEventLoop, type[Body] | None]] = {}
        self._writer_connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        _ = self._connection.execute("PRAGMA foreign_keys=ON")
        try:
            _check_schema(self._connection)
            # WAL lets a pinned history read coexist with short committed writes.
            mode = self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode != "wal":
                raise RuntimeError("MessageLog requires a file-backed WAL database")
            # 普通进程崩溃保留提交；宿主故障可能丢失尾部事务，保证边界见 ADR-0099。
            self._connection.execute("PRAGMA synchronous=NORMAL")
            fresh = (
                self._connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='messages'"
                ).fetchone()
                is None
            )
            with self._connection:
                for name, statement in _SCHEMA.items():
                    # 新库由 owner 初始化；已有库的新持久能力只能由 yoyo 接纳。
                    if not fresh and (name in _MESSAGE_PREFIX_SCHEMA or name in {
                        "owner_records", "message_embeddings", "ix_message_embeddings_hash", "message_source_seq", "message_source_kind_seq",
                        "attachments", "message_attachments", "idx_message_attachments_artifact",
                    }):
                        continue
                    _ = self._connection.execute(statement)
                if fresh:
                    self._connection.execute("INSERT INTO message_prefix_revision VALUES (1,0)")
            prefix_schema = {
                row[0] for row in self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE name IN (?,?,?,?)", tuple(_MESSAGE_PREFIX_SCHEMA),
                )
            }
            if prefix_schema and prefix_schema != _MESSAGE_PREFIX_SCHEMA.keys():
                raise RuntimeError("消息前缀标记迁移不完整")
            self._has_prefix_revision = bool(prefix_schema)
            if self._has_prefix_revision and self._connection.execute(
                "SELECT revision FROM message_prefix_revision WHERE singleton=1"
            ).fetchone() is None:
                raise RuntimeError("消息前缀标记缺少初始行")
            self._has_metadata = "metadata" in {
                row["name"] for row in self._connection.execute("PRAGMA table_info(messages)")
            }
            self._has_deleted = "deleted_at" in {
                row["name"] for row in self._connection.execute("PRAGMA table_info(sessions)")
            }
            self._has_title = "title" in {
                row["name"] for row in self._connection.execute("PRAGMA table_info(sessions)")
            }
        except BaseException:
            self._connection.close()
            raise

    @property
    def _connection(self) -> sqlite3.Connection:
        read = self._reads.current
        return self._writer_connection if read is None else read.connection

    @property
    def _lock(self):
        read = self._reads.current
        return self._writer_lock if read is None else read.lock

    def _decode(self, row: sqlite3.Row) -> Message:
        """查询仍读真实行；小行另有有界强缓存，大行只复用仍被持有的不可变消息。"""
        key = tuple(row)
        # 独立只读连接共享解码结果；此短锁不等待 writer 的事务或磁盘操作。
        with self._decode_lock:
            message = self._decoded.get(key)
            if message is None:
                message = self._decoded_strong.get(key)
        if message is None:
            message = _message(row)
            with self._decode_lock:
                if len(row["body"]) <= _DECODE_STRONG_BODY:
                    if len(self._decoded_strong) >= _DECODE_STRONG_LIMIT:
                        self._decoded_strong.pop(next(iter(self._decoded_strong)))
                    self._decoded_strong[key] = message
                else:
                    self._decoded[key] = message
        return message

    def backup(self, destination: Path) -> None:
        """向新文件保存已提交的完整数据库，供隔离宿主独立打开。"""
        with self._lock:
            # 备份未提交连接会等待自身事务；调用者必须先完成原消息事务。
            if self._connection.in_transaction:
                raise RuntimeError("消息事务未结束，不能建立副本")
            with destination.open("xb"):
                pass
            with closing(sqlite3.connect(destination)) as snapshot:
                self._connection.backup(snapshot)

    def read_bindings(self) -> tuple[Mapping[str, object], ...]:
        """列出不可变绑定，供宿主保存数据库副本所需的归档闭包。"""
        with self._read():
            identities = self._connection.execute("SELECT binding_id FROM bindings ORDER BY binding_id").fetchall()
            return tuple(self.read_binding(row[0]) for row in identities)

    def owner(self, name: str) -> OwnerStore:
        """组合只向 owner 授予自身的记录空间，不授予 SQL 或其他空间。"""
        if not isinstance(name, str) or not name:
            raise ValueError("状态 owner 不能为空")
        if not getattr(self, "_has_owner_records", False):
            with self._read(snapshot=False):
                if (
                    self._connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE name='owner_records'"
                    ).fetchone()
                    is None
                ):
                    raise RuntimeError("owner_records 缺失，请先运行对应 yoyo 迁移")
            # 表只在迁移中创建，从不删除；首次确认后不再重复查询。
            self._has_owner_records = True
        return OwnerStore(self, name)

    def ensure_session(
        self, session_id: str, attributes: SessionAttributes, *,
        initializers: tuple[tuple[OwnerStore, Callable[[str, SessionAttributes, OwnerTransaction], None]], ...] = (),
    ) -> SessionAttributes:
        """原子接纳固定属性；同 ID 重试不能修改已有会话的事实。"""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Session ID 不能为空")
        payload = encode_attributes(attributes)
        def create() -> SessionAttributes:
            stamp = datetime.now(UTC).isoformat()
            inserted = self._connection.execute(
                "INSERT OR IGNORE INTO sessions (key,created_at,updated_at,attributes) VALUES (?,?,?,?)",
                (session_id, stamp, stamp, payload),
            )
            current = self.catalog().attributes(session_id)
            if current != attributes:
                raise MessageConflict("同一 Session 的固定属性不能改变")
            # Only the new row admits initial owner state, in this same transaction.
            if inserted.rowcount == 1:
                self._changed()
                for store, initialize in initializers:
                    if store._log is not self:
                        raise ValueError("Session 初始化不能跨存储 authority")
                    transaction = OwnerTransaction(store)
                    try:
                        result = initialize(session_id, current, transaction)
                        transaction._check_active()
                        if inspect.isawaitable(result):
                            if inspect.iscoroutine(result):
                                result.close()
                            raise TypeError("Session 初始化必须同步，不能跨 await")
                    finally:
                        transaction._active = False
            return current
        return self._write(create)

    async def ensure_session_async(
        self, session_id: str, attributes: SessionAttributes, *,
        initializers: tuple[tuple[OwnerStore, Callable[[str, SessionAttributes, OwnerTransaction], None]], ...] = (),
    ) -> SessionAttributes:
        """完整 create-once 事务离开 loop；取消仍排空已开始的写入。"""
        self._check_async_operation()
        return await _run_commit(self, lambda: self.ensure_session(
            session_id, attributes, initializers=initializers,
        ), None)

    def set_session_deleted(self, session_id: str, *, deleted: bool) -> str | None:
        """用户显式的数据管理操作：软删或恢复 Session，消息与其他 owner 状态不动。

        幂等：重复设置返回当前值且不产生写操作；返回软删时间戳或 None。
        """
        def change() -> str | None:
            column = "deleted_at" if self._has_deleted else "NULL AS deleted_at"
            row = self._connection.execute(
                f"SELECT {column} FROM sessions WHERE key=?", (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            if not self._has_deleted:
                if deleted:
                    raise RuntimeError("sessions 缺少 deleted_at，请先完成 yoyo 迁移")
                return None
            current = row["deleted_at"]
            if not deleted:
                if current is None:
                    return None
                self._connection.execute(
                    "UPDATE sessions SET deleted_at=NULL WHERE key=?", (session_id,),
                )
                self._changed()
                return None
            if current is not None:
                return current
            stamp = datetime.now(UTC).isoformat()
            self._connection.execute(
                "UPDATE sessions SET deleted_at=? WHERE key=?", (stamp, session_id),
            )
            self._changed()
            return stamp

        return self._write(change)

    def set_session_title(self, session_id: str, title: str | None) -> str | None:
        """用户显式的数据管理操作：覆盖或清除会话显示标题，消息与属性不动。

        规范化：去首尾空白，空值清除覆盖、回到首条消息推导标题；长度上限
        `_SESSION_TITLE_MAX`；`updated_at` 不变——重命名不改变目录排序事实。
        幂等：与当前值相同不产生写操作。返回落库后的覆盖值或 None。
        """
        normalized = None if title is None else title.strip() or None
        if normalized is not None and len(normalized) > _SESSION_TITLE_MAX:
            raise ValueError(f"会话标题不能超过 {_SESSION_TITLE_MAX} 个字符")

        def change() -> str | None:
            column = "title" if self._has_title else "NULL AS title"
            row = self._connection.execute(
                f"SELECT {column} FROM sessions WHERE key=?", (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            if not self._has_title:
                if normalized is not None:
                    raise RuntimeError("sessions 缺少 title，请先完成 yoyo 迁移")
                return None
            if row["title"] == normalized:
                return normalized
            self._connection.execute(
                "UPDATE sessions SET title=? WHERE key=?", (normalized, session_id),
            )
            self._changed()
            return normalized

        return self._write(change)

    def _check_async_operation(self) -> None:
        """线程工作不能离开调用者自己的未提交或只读事务。"""
        lock = self._lock
        if lock.acquire(blocking=False):
            try:
                if self._connection.in_transaction:
                    raise RuntimeError("Async storage cannot leave an active storage transaction")
            finally:
                lock.release()

    def _write(self, callback: Callable[[], _T]) -> _T:
        """同步事务不跨 await；全部权威写成功后才通知日志读者。"""
        with self._lock:
            if self._connection.in_transaction:
                raise RuntimeError("事务内写入必须使用当前 transaction 接口")
            with self._connection:
                _ = self._connection.execute("BEGIN IMMEDIATE")
                self._notify_pending = set()
                result = callback()
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("存储事务回调必须同步，不能跨 await")
            # owner 账本或 embedding 的单独提交不会改变消息订阅结果。
            if self._notify_pending is None or self._notify_pending:
                if getattr(self._defer_notify, "active", False):
                    # 异步提交路径：唤醒投递让给提交者恢复后再做，
                    # listener 的追赶读不再排在提交者延续之前。
                    with self._listener_lock:
                        if self._notify_pending is None:
                            self._notify_owed = None
                        elif self._notify_owed is not None:
                            self._notify_owed.update(self._notify_pending)
                else:
                    self._notify(self._notify_pending)
            return result

    def _changed(self, body_type: type[Body] | None = None) -> None:
        """事务中只收集消息类型，提交后再选订阅者，避免漏掉并发新订阅。"""
        if body_type is None:
            self._notify_pending = None
        elif self._notify_pending is not None:
            self._notify_pending.add(body_type)

    def flush_notify(self) -> None:
        """交付异步提交欠下的唤醒；不同提交的消息类型在同一锁内合并。"""
        with self._listener_lock:
            pending, self._notify_owed = self._notify_owed, set()
        self._notify(pending)

    def _notify(self, changed: set[type[Body]] | None) -> None:
        """逐个通知已注册读者；提交已经成立，observer 失败不污染返回结果。

        唤醒按 Event 合并：唤醒在途或 Event 已置位时，读者 clear 后的重读
        必然观察到本提交（通知先于提交之后的任何重读到达），重复投递
        call_soon 只会挤占循环。在途标记在投递回调内清除，清除与置位之间
        的提交重新排队，不会丢唤醒。
        只有 loop 确认已关闭的 listener 才移除；无法确认死亡的订阅保留，
        告警如实记录，由 follow 周期核对兜底恢复持久事实。
        """
        if changed is not None and not changed:
            return
        with self._listener_lock:
            pending = [
                (event, loop) for event, (loop, wake_on) in self._listeners.items()
                if (changed is None or wake_on is None or wake_on in changed)
                and not event.is_set() and event not in self._notify_in_flight
            ]
            self._notify_in_flight.update(event for event, _ in pending)
        for event, loop in pending:
            try:
                _ = loop.call_soon_threadsafe(self._deliver, event)
            except BaseException as error:
                with self._listener_lock:
                    self._notify_in_flight.discard(event)
                is_closed = getattr(loop, "is_closed", None)
                try:
                    dead = bool(is_closed()) if callable(is_closed) else False
                except Exception:
                    dead = False
                if dead:
                    # 拒绝投递且 loop 确认关闭：永远无法再唤醒，确认死亡才移除。
                    with self._listener_lock:
                        _ = self._listeners.pop(event, None)
                    _logger.warning("日志 listener 已死亡并移除: %r", error)
                else:
                    # 无法确认死亡的订阅保留；持久事实由 level 触发轮询兜底。
                    _logger.warning("日志 listener 通知失败，保留订阅待周期核对: %r", error)

    def _deliver(self, event: asyncio.Event) -> None:
        """在读者 loop 上完成一次合并唤醒：先清在途标记再置位，间隙提交重新排队。"""
        with self._listener_lock:
            self._notify_in_flight.discard(event)
        event.set()

    def _read(self, *, snapshot: bool = True) -> _BorrowedRead:
        """借用只读连接；组合读取固定快照，单条查询使用 SQLite 自身的快照。"""
        return _BorrowedRead(self, snapshot)

    def catalog(self) -> MessageCatalog:
        return MessageCatalog(self)

    def reader(self, session_id: str) -> MessageReader:
        return MessageReader(self, session_id)

    def invalidate_attachment_memo(self, message_ids: Iterable[str]) -> None:
        """权威删除路径提交后失效对应附件备忘；未备忘的 id 忽略。"""
        with self._decode_lock:
            for message_id in message_ids:
                self._attachment_memo.pop(message_id, None)

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
        metadata_keys: frozenset[str] = frozenset(),
        update_metadata: Callable[[Body], Mapping[str, object | None]] | None = None,
        message_metadata_keys: frozenset[str] = frozenset(),
        check_metadata: Callable[[Mapping[str, object]], None] | None = None,
    ) -> MessageWriter:
        """绑定纯检查与元数据投影；投影只改获授键，None 移除键，不执行外部效果。"""
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
            metadata_keys,
            update_metadata,
            message_metadata_keys,
            check_metadata,
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
        """读取不可变绑定；缺失引用不能用当前实现补齐。

        binding_id 是内容的 sha256，descriptor 不可变，解码结果可永久缓存。
        """
        cached = getattr(self, "_binding_cache", None)
        if cached is None:
            cached = self._binding_cache = {}
        value = cached.get(binding_id)
        if value is None:
            with self._read(snapshot=False):
                row = self._connection.execute(
                    "SELECT descriptor FROM bindings WHERE binding_id=?", (binding_id,)
                ).fetchone()
            if row is None:
                raise KeyError(binding_id)
            value = cached[binding_id] = _json_object(row[0], f"binding {binding_id}")
        return value

    def close(self) -> None:
        """释放数据库并唤醒所有追赶者，让它们正常退出。"""
        with self._writer_lock:
            with self._read_admission:
                if self._closed:
                    return
                self._closed = True
                idle_reads, self._idle_reads = self._idle_reads, []
            for read in idle_reads:
                read.connection.close()
            self._writer_connection.close()
            with self._listener_lock:
                listeners = tuple(self._listeners.items())
            for event, (loop, _) in listeners:
                try:
                    _ = loop.call_soon_threadsafe(event.set)
                except BaseException as error:
                    with self._listener_lock:
                        _ = self._listeners.pop(event, None)
                    _logger.warning("日志 listener 已死亡并移除: %r", error)


class MessageCatalog:
    """只读会话目录；一次 heads 快照固定跨会话消费的消息上界。"""

    def __init__(self, log: MessageLog | None):
        self._storage = log

    @property
    def _log(self) -> MessageLog:
        if self._storage is None:
            raise RuntimeError("candidate 验证期禁止读取正式会话目录")
        return self._storage

    def snapshot_heads(
        self, *, prefix: str = "", visibility: Literal["listed", "internal"] | None = None,
    ) -> Mapping[str, int]:
        """单条查询取得同一数据库快照，不逐会话读取可能变化的 head。

        只在同一只读连接内按该连接自己的 data_version 复用目录：版本与数据来自
        同一快照。跨连接共享（旧版 _heads_shared）曾把旧快照的查询结果挂到
        writer 的新版本键下（评审 #1127），已撤除；follower 减少查询的正确手段
        是复用连接与同版本命中，不是跨连接共享结果。
        """
        log = self._log
        if visibility not in (None, "listed", "internal"):
            raise InvalidPage("目录 visibility 无效")
        filtered = bool(prefix) or visibility is not None
        with log._read(snapshot=False) as connection:
            # 1. data_version 只可在同一连接上比较；writer 未提交视图不能复用。
            read = log._reads.current
            current = None if read is None else connection.execute("PRAGMA data_version").fetchone()[0]
            if not filtered and read is not None and read.heads is not None and read.heads_version == current:
                return read.heads
            # 2. 每个只读连接只保留最近一份目录，其他连接提交后重新查询。
            where = ["substr(s.key,1,?)=?"]
            values: list[object] = [len(prefix), prefix]
            if visibility is not None:
                where.append("json_extract(s.attributes,'$.visibility')=?")
                values.append(visibility)
                if log._has_deleted:
                    where.append("s.deleted_at IS NULL")
            rows = connection.execute(
                "SELECT s.key, COALESCE((SELECT m.seq FROM messages m "
                "WHERE m.session_key=s.key ORDER BY m.seq DESC LIMIT 1), -1) AS head "
                "FROM sessions s WHERE " + " AND ".join(where) + " ORDER BY s.key", values,
            ).fetchall()
            heads = MappingProxyType({row["key"]: row["head"] for row in rows})
            if not filtered and read is not None:
                read.heads_version, read.heads = current, heads
            return heads

    def reader(self, session_id: str) -> MessageReader:
        return self._log.reader(session_id)

    def attributes(self, session_id: str) -> SessionAttributes:
        return self.reader(session_id).attributes

    def exists(self, session_id: str) -> bool:
        """Check admission without treating empty or nullable metadata as absence."""
        with self._log._read():
            return self._log._connection.execute(
                "SELECT 1 FROM sessions WHERE key=?", (session_id,),
            ).fetchone() is not None

    def sessions(
        self, *, prefix: str = "", visibility: Literal["listed", "internal"] | None = None,
        after: tuple[str, str] | None = None, limit: int = 50,
    ) -> SessionPage:
        """按最近活跃时间读取 live 目录；软删会话默认排除，续页期间更新的会话须刷新首页重新取得。"""
        if not 1 <= limit <= 200:
            raise InvalidPage("目录 limit 必须在 1 到 200 之间")
        if visibility not in (None, "listed", "internal"):
            raise InvalidPage("目录 visibility 无效")
        where = ["substr(s.key,1,?)=?"]
        values: list[object] = [len(prefix), prefix]
        if self._log._has_deleted:
            where.append("s.deleted_at IS NULL")
        if visibility is not None:
            where.append("json_extract(s.attributes,'$.visibility')=?")
            values.append(visibility)
        base_where = " AND ".join(where)
        page_values = list(values)
        if after is not None:
            try:
                _ = _timestamp(after[0], "Session 目录 cursor")
            except ValueError as error:
                raise InvalidPage(str(error)) from error
            where.append("(julianday(s.updated_at)<julianday(?) OR "
                         "(julianday(s.updated_at)=julianday(?) AND s.key>?))")
            page_values.extend((after[0], after[0], after[1]))
        # 1. CROSS JOIN 固定 page 为外层，避免 SQLite 为 GROUP BY 扫描全部消息。
        sql = """
            WITH page AS MATERIALIZED (
                SELECT s.*, %s AS title FROM sessions s WHERE %s
                ORDER BY julianday(s.updated_at) DESC, s.key ASC LIMIT ?
            ), stats AS (
                SELECT m.session_key, COUNT(*) AS message_count,
                       MAX(m.seq) AS head_seq, MIN(m.seq) AS first_seq
                FROM page p CROSS JOIN messages m
                WHERE m.session_key=p.key
                GROUP BY p.key
            )
            SELECT p.*, COALESCE(t.message_count,0) AS message_count,
                   COALESCE(t.head_seq,-1) AS head_seq,
                   m.id AS first_id, m.seq AS first_seq, m.ts AS first_ts,
                   m.author AS first_author, m.source AS first_source, m.body AS first_body,
                   %s AS first_metadata
            FROM page p LEFT JOIN stats t ON t.session_key=p.key
            LEFT JOIN messages m ON m.session_key=p.key AND m.seq=t.first_seq
            ORDER BY julianday(p.updated_at) DESC, p.key ASC
        """ % (
            "s.title" if self._log._has_title else "NULL",
            " AND ".join(where),
            "m.metadata" if self._log._has_metadata else "'{}'",
        )
        with self._log._read() as connection:
            total = connection.execute("SELECT COUNT(*) FROM sessions s WHERE " + base_where, values).fetchone()[0]
            rows = connection.execute(sql, [*page_values, limit + 1]).fetchall()
        # 2. 只返回事实；标题、空会话呈现和历史标签由 adapter 决定。
        entries = tuple(_session_entry(row) for row in rows[:limit])
        cursor = (entries[-1].updated_at.isoformat(), entries[-1].session_id) if len(rows) > limit else None
        return SessionPage(entries, total, cursor)

    def snapshot_attributes(self) -> Mapping[str, SessionAttributes]:
        with self._log._read():
            rows = self._log._connection.execute("SELECT key, attributes FROM sessions ORDER BY key").fetchall()
        return MappingProxyType({row["key"]: decode_attributes(row["attributes"]) for row in rows})

    async def follow(
        self, *, poll_interval: float | None = None, wake_on: type[Body] | None = None,
        prefix: str = "", visibility: Literal["listed", "internal"] | None = None,
    ) -> AsyncGenerator[Mapping[str, int]]:
        """先订阅再取 heads；通知只降低延迟，消费者始终按快照重读事实。

        wake_on 只限制进程内消息唤醒，初始追赶与返回的 heads 仍包含全部消息；
        会话管理变化和关闭仍通知所有订阅。
        poll_interval 给出有界重扫节奏：进程内唤醒丢失时，已提交的持久变化
        最多在一个周期后被重新发现。
        prefix 与 visibility 限定只读目录；指定 visibility 时排除软删会话。
        """
        event = asyncio.Event()
        with self._log._listener_lock:
            if self._log._closed:
                return
            self._log._listeners[event] = (asyncio.get_running_loop(), wake_on)
        previous: Mapping[str, int] | None = None
        try:
            while True:
                event.clear()
                if self._log._closed:
                    return
                heads = self.snapshot_heads(prefix=prefix, visibility=visibility)
                if heads != previous:
                    previous = heads
                    yield heads
                # 消费期间的提交仍会置位；无需先重查一次空变化再等待。
                if poll_interval is None:
                    _ = await event.wait()
                else:
                    try:
                        _ = await asyncio.wait_for(event.wait(), poll_interval)
                    except TimeoutError:
                        pass
        finally:
            with self._log._listener_lock:
                self._log._listeners.pop(event, None)
                self._log._notify_in_flight.discard(event)


class _MessagePrefix:
    """一个日志实例内可追加的已提交前缀；旧快照用固定长度隔离后续追加。"""

    def __init__(self, session_id: str, revision: int | None):
        self.session_id = session_id
        self.revision = revision
        self.messages: list[Message] = []
        self.seqs: list[int] = []


@dataclass(frozen=True, slots=True)
class MessageSnapshot(Sequence[Message]):
    """固定消息读面；只提供消息和前缀关系，不持有数据库或写入能力。"""

    _prefix: _MessagePrefix
    _count: int
    through_seq: int

    @property
    def session_id(self) -> str:
        return self._prefix.session_id

    @property
    def prefix_revision(self) -> int | None:
        return self._prefix.revision

    def extends(self, previous: MessageSnapshot) -> bool:
        """相同存储读面只追加；截短或前缀变化不能复用旧投影。"""
        return self._prefix is previous._prefix and self._count >= previous._count

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[Message]:
        return islice(self._prefix.messages, self._count)

    @overload
    def __getitem__(self, index: int) -> Message: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Message, ...]: ...

    def __getitem__(self, index: int | slice) -> Message | tuple[Message, ...]:
        if isinstance(index, slice):
            return tuple(self._prefix.messages[i] for i in range(*index.indices(self._count)))
        position = index + self._count if index < 0 else index
        if not 0 <= position < self._count:
            raise IndexError(index)
        return self._prefix.messages[position]


class MessageReader:
    def __init__(self, log: MessageLog, session_id: str):
        self._log = log
        self._session_id = session_id

    def incremental(self) -> MessageReader:
        """创建本次程序的只读视图，旧前缀复用解码结果，后续读取追赶新增消息。"""
        return _IncrementalMessageReader(self._log, self._session_id)

    def committed_snapshot(self, *, through_seq: int | None = None) -> MessageSnapshot:
        """从同一 SQL 快照核对上界与前缀版本，只加载新尾部。

        已有读写事务使用自己的真实 SQL 视图，不发布或复用共享前缀。
        旧快照保持固定内容；最后一个消费者释放后，日志不保留该会话正文。
        """
        log = self._log
        # 1. 显式事务保留原读面，包括调用者尚未提交的写入。
        if log._reads.current is not None:
            return self._transaction_snapshot(through_seq)
        with log._view_lock, log._read() as connection:
            if connection is log._writer_connection:
                return self._transaction_snapshot(through_seq)
            # 2. 两项事实来自同一 RO 事务，外部追加和前缀改写都可见。
            revision = None if not log._has_prefix_revision else connection.execute(
                "SELECT revision FROM message_prefix_revision WHERE singleton=1"
            ).fetchone()[0]
            head = self.head()
            upper = head if through_seq is None else min(head, through_seq)
            prefix = log._message_views.get(self._session_id)
            if prefix is None or revision is None or prefix.revision != revision:
                prefix = _MessagePrefix(self._session_id, revision)
            previous = prefix.seqs[-1] if prefix.seqs else -1
            if upper > previous:
                tail = MessageReader.scan(self, tuple, after_seq=previous, through_seq=upper)
                prefix.messages.extend(tail)
                prefix.seqs.extend(message.seq for message in tail)
            # 3. 只有真实提交后的读取进入共享读面，不依赖 observer 唤醒。
            log._message_views[self._session_id] = prefix
            return MessageSnapshot(prefix, bisect_right(prefix.seqs, upper), upper)

    def _transaction_snapshot(self, through_seq: int | None) -> MessageSnapshot:
        """事务内快照独占前缀，不能为之后的已提交读取签发复用证明。"""
        messages = MessageReader.scan(self, tuple, through_seq=through_seq)
        prefix = _MessagePrefix(self._session_id, None)
        prefix.messages.extend(messages)
        prefix.seqs.extend(message.seq for message in messages)
        return MessageSnapshot(prefix, len(messages), messages[-1].seq if messages else -1)

    async def committed_snapshot_async(self, *, through_seq: int | None = None) -> MessageSnapshot:
        """在线程内签发一次固定读面；取消先排空实际读取。"""
        self._check_async_snapshot()
        return await run_file_io(lambda: self.committed_snapshot(through_seq=through_seq))

    def _check_async_snapshot(self) -> None:
        """异步读取不得离开调用线程自己的未提交事务。"""
        self._log._check_async_operation()

    async def read_async(self, consume: Callable[[MessageReader], _T]) -> _T:
        """在固定只读快照内执行纯读取，取消先排空连接再返回。"""
        self._check_async_snapshot()

        def read() -> _T:
            reader = MessageReader(self._log, self._session_id)
            with reader.read_snapshot():
                result = consume(reader)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("异步读取的 worker 回调必须同步，不能跨 await")
                return result

        return await run_file_io(read)

    async def snapshot_async(self, *, through_seq: int) -> tuple[Message, ...]:
        """Read a fixed prefix off-loop and drain its connection before cancellation."""
        self._check_async_snapshot()
        return await run_file_io(
            lambda: MessageReader(self._log, self._session_id).snapshot(through_seq=through_seq)
        )

    @contextmanager
    def read_snapshot(self) -> Generator[None]:
        """在调用者线程固定已提交只读视图，不等待其他 writer 或提供 SQL。"""
        with self._log._read():
            yield

    def source_changed(self, source: str, through_seq: int) -> bool:
        """只判断后续 Input/Control，不解码无关历史。"""
        with self._log._read(snapshot=False) as connection:
            return connection.execute(
                "SELECT 1 FROM messages WHERE session_key=? AND source=? AND seq>? "
                "AND json_extract(body, '$.kind') IN ('input','control') LIMIT 1",
                (self._session_id, source, through_seq),
            ).fetchone() is not None

    @property
    def session_id(self) -> str:
        return self._session_id

    def metadata(self) -> Mapping[str, object] | None:
        """读取不可变元数据副本；未知 Session 返回 None，不创建会话。"""
        with self._log._read(snapshot=False):
            row = self._log._connection.execute(
                "SELECT metadata FROM sessions WHERE key=?", (self._session_id,),
            ).fetchone()
        if row is None or row["metadata"] is None:
            return None
        return _json_object(row["metadata"], f"Session {self._session_id} metadata")

    @property
    def attributes(self) -> SessionAttributes:
        with self._log._read(snapshot=False):
            row = self._log._connection.execute("SELECT attributes FROM sessions WHERE key=?", (self._session_id,)).fetchone()
        if row is None:
            raise ValueError("Session 尚未接纳")
        raw = row["attributes"]
        cached = self._log._decoded_attributes.get(raw)
        if cached is None:
            cached = decode_attributes(raw)
            self._log._decoded_attributes[raw] = cached
        return cached

    @property
    def deleted(self) -> bool:
        """软删只是目录与导航的呈现状态；直接读取仍返回原消息。"""
        column = "deleted_at" if self._log._has_deleted else "NULL AS deleted_at"
        with self._log._read(snapshot=False):
            row = self._log._connection.execute(
                f"SELECT {column} FROM sessions WHERE key=?", (self._session_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Session 尚未接纳")
        return row["deleted_at"] is not None

    @property
    def title(self) -> str | None:
        """显式标题覆盖；None 表示沿用首条消息推导。"""
        column = "title" if self._log._has_title else "NULL AS title"
        with self._log._read(snapshot=False):
            row = self._log._connection.execute(
                f"SELECT {column} FROM sessions WHERE key=?", (self._session_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Session 尚未接纳")
        return row["title"]

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
        with self._log._read(snapshot=False):
            rows = self._log._connection.execute(sql, values).fetchall()
            return tuple(self._log._decode(row) for row in rows)

    def source_names(self) -> frozenset[str]:
        """只读取本 Session 中出现过的来源，不解码消息正文。"""
        with self._log._read(snapshot=False):
            rows = self._log._connection.execute(
                "SELECT DISTINCT source FROM messages WHERE session_key=?", (self._session_id,),
            ).fetchall()
        return frozenset(row[0] for row in rows)

    def latest_input(self, source: str, *, through_seq: int) -> Message | None:
        """读取指定前缀中最后一条同来源 Input，后来输入不改变旧回复的目的地。"""
        with self._log._read(snapshot=False):
            row = self._log._connection.execute(
                "SELECT * FROM messages WHERE session_key=? AND source=? AND seq<=? "
                "AND json_extract(body,'$.kind')='input' ORDER BY seq DESC LIMIT 1",
                (self._session_id, source, through_seq),
            ).fetchone()
            return None if row is None else self._log._decode(row)

    def latest_input_seq(self, source: str, *, through_seq: int) -> int | None:
        """Read the last Input position without loading its content or metadata."""
        with self._log._read(snapshot=False):
            row = self._log._connection.execute(
                "SELECT seq FROM messages WHERE session_key=? AND source=? AND seq<=? "
                "AND json_extract(body,'$.kind')='input' ORDER BY seq DESC LIMIT 1",
                (self._session_id, source, through_seq),
            ).fetchone()
        return None if row is None else row[0]

    def latest_finished_output_seq(
        self, source: str, *, after_seq: int, through_seq: int,
    ) -> int | None:
        """Read the last terminal Output position in a fixed source range."""
        with self._log._read(snapshot=False):
            row = self._log._connection.execute(
                "SELECT seq FROM messages WHERE session_key=? AND source=? AND seq>? AND seq<=? "
                "AND json_extract(body,'$.kind')='output' "
                "AND json_extract(body,'$.finish') IN ('complete','quiet') ORDER BY seq DESC LIMIT 1",
                (self._session_id, source, after_seq, through_seq),
            ).fetchone()
        return None if row is None else row[0]

    def scan_controls(
        self, consume: Callable[[Iterable[tuple[int, Control]]], _T], *,
        source: str, after_seq: int, through_seq: int,
    ) -> _T:
        """Read ordered Control bodies in one prefix without unrelated message data."""
        with self._log._read():
            def controls() -> Generator[tuple[int, Control], None, None]:
                cursor = after_seq
                while cursor < through_seq:
                    rows = self._log._connection.execute(
                        "SELECT seq,body FROM messages WHERE session_key=? AND source=? AND seq>? AND seq<=? "
                        "AND json_extract(body,'$.kind')='control' ORDER BY seq LIMIT 64",
                        (self._session_id, source, cursor, through_seq),
                    ).fetchall()
                    if not rows:
                        return
                    for row in rows:
                        body = decode_body(row['body'])
                        assert isinstance(body, Control)
                        if body.through_seq >= row['seq']:
                            raise ValueError("Control cannot refer to an unaccepted prefix")
                        yield row['seq'], body
                    cursor = rows[-1]['seq']
            result = consume(controls())
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("Control scan callback must be synchronous")
            return result

    def latest_control(self, source: str, *, through_seq: int) -> Message | None:
        """读取固定前缀内最后一条同来源 Control，不解码较早的正文。"""
        with self._log._read():
            row = self._log._connection.execute(
                "SELECT * FROM messages WHERE session_key=? AND source=? AND seq<=? "
                "AND json_extract(body,'$.kind')='control' ORDER BY seq DESC LIMIT 1",
                (self._session_id, source, through_seq),
            ).fetchone()
            return None if row is None else self._log._decode(row)

    def scan(
        self, consume: Callable[[Iterable[Message]], _T], *, after_seq: int = -1,
        through_seq: int | None = None, source: str | None = None,
    ) -> _T:
        """在同一读快照内分页消费；回调必须同步完成，不能保留迭代器。"""
        with self._log._read():
            head = self.head() if through_seq is None else through_seq

            def messages() -> Generator[Message, None, None]:
                cursor = after_seq
                while cursor < head:
                    page = self.read(after_seq=cursor, through_seq=head, source=source, limit=64)
                    if not page:
                        break
                    yield from page
                    cursor = page[-1].seq

            with closing(messages()) as rows:
                result = consume(rows)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("消息扫描回调必须同步，不能跨 await")
                return result

    def snapshot(self, *, after_seq: int = -1, through_seq: int | None = None) -> tuple[Message, ...]:
        """固定上界后读取完整区间；无需完整正文的消费者应使用 scan。"""
        return self.scan(tuple, after_seq=after_seq, through_seq=through_seq)

    def read_page(
        self, *, after_seq: int = -1, through_seq: int | None = None, limit: int = 50,
    ) -> MessagePage:
        """从固定上界内向前追赶；多读一条判断是否续页，不遍历历史计数。"""
        return self._page(after_seq=after_seq, before_seq=None, through_seq=through_seq, limit=limit, tail=False)

    def read_tail(
        self, *, before_seq: int | None = None, through_seq: int | None = None, limit: int = 50,
    ) -> MessagePage:
        """取结尾或更早的一页，始终按原始 seq 升序返回。"""
        return self._page(after_seq=-1, before_seq=before_seq, through_seq=through_seq, limit=limit, tail=True)

    def _page(
        self, *, after_seq: int, before_seq: int | None, through_seq: int | None, limit: int, tail: bool,
    ) -> MessagePage:
        """消息与其附件、binding 在一个短读取事务内批量取得。"""
        if not 1 <= limit <= 200 or after_seq < -1 or before_seq is not None and before_seq < 0:
            raise InvalidPage("消息页范围或 limit 无效")
        with self._log._read() as connection:
            # 1. 区分未知与空 Session；首次读取把当前 head 固定为页面上界。
            row = connection.execute(
                "SELECT (SELECT COALESCE(MAX(seq),-1) FROM messages WHERE session_key=s.key) AS head "
                "FROM sessions s WHERE s.key=?", (self._session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(self._session_id)
            head = row["head"]
            through = head if through_seq is None else through_seq
            if not -1 <= through <= head or after_seq > through:
                raise InvalidPage("消息页上界或 cursor 超过已接纳前缀")
            before = through + 1 if before_seq is None else before_seq
            if before > through + 1:
                raise InvalidPage("消息页 before_seq 超过快照上界")
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_key=? AND seq>? AND seq<=? AND seq<? "
                + ("ORDER BY seq DESC LIMIT ?" if tail else "ORDER BY seq ASC LIMIT ?"),
                (self._session_id, after_seq, through, before, limit + 1),
            ).fetchall()
            selected = rows[:limit]
            if tail:
                selected.reverse()
            messages = tuple(self._log._decode(row) for row in selected)
            # 2. 只取该页引用，不逐消息查询，也不启动归档目标或执行任何能力。
            attachments, bindings = _page_references(connection, messages)
        return MessagePage(messages, attachments, bindings, through, len(rows) > limit)

    def get(self, message_id: str) -> Message | None:
        """按不可变身份读取消息，不能跨 reader 获授的 Session。"""
        with self._log._read(snapshot=False):
            row = self._log._connection.execute(
                "SELECT * FROM messages WHERE id=? AND session_key=?",
                (message_id, self._session_id),
            ).fetchone()
            return None if row is None else self._log._decode(row)

    def attachments(self, message_id: str) -> tuple[AttachmentRef, ...]:
        """只读取已获授 Session 中该消息的有序附件引用。"""
        return self.attachments_for((message_id,))

    def attachments_for(self, message_ids: tuple[str, ...]) -> tuple[AttachmentRef, ...]:
        """批量读取已获授消息的附件，按输入顺序保留重复引用。"""
        if not message_ids:
            return ()
        log = self._log
        memo = log._attachment_memo
        with log._decode_lock:
            missing = tuple(dict.fromkeys(mid for mid in message_ids if mid not in memo))
        if missing:
            with log._read():
                rows = log._connection.execute(
                    "SELECT m.id,ma.ordinal,a.* FROM messages m "
                    "LEFT JOIN message_attachments ma ON ma.message_id=m.id "
                    "LEFT JOIN attachments a ON a.artifact_id=ma.artifact_id "
                    "WHERE m.session_key=? AND m.id IN (SELECT value FROM json_each(?)) "
                    "ORDER BY m.seq,ma.ordinal", (self._session_id, json.dumps(missing)),
                ).fetchall()
            refs: dict[str, list[AttachmentRef]] = {}
            for row in rows:
                items = refs.setdefault(row["id"], [])
                if row["ordinal"] is not None:
                    if row["ordinal"] != len(items) or row["artifact_id"] is None:
                        raise ValueError(f"Message {row['id']} 附件引用损坏")
                    items.append(_artifact_ref(row))
            if refs.keys() != set(missing):
                raise LookupError("消息不在 reader 获授的 Session 中")
            with log._decode_lock:
                for identity in missing:
                    memo[identity] = tuple(refs[identity])
                while len(memo) > _ATTACHMENT_MEMO_SIZE:
                    _ = memo.pop(next(iter(memo)))
        with log._decode_lock:
            return tuple(ref for identity in message_ids for ref in memo[identity])

    def head(self, *, source: str | None = None) -> int:
        sql = "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE session_key=?"
        values = [self._session_id]
        if source is not None:
            sql += " AND source=?"
            values.append(source)
        with self._log._read(snapshot=False):
            return self._log._connection.execute(sql, values).fetchone()[0]

    async def follow_heads(self) -> AsyncGenerator[int, None]:
        """只订阅本 Session 的提交序号；通知不要求解码或保留正文。"""
        event = asyncio.Event()
        with self._log._listener_lock:
            if self._log._closed:
                return
            self._log._listeners[event] = (asyncio.get_running_loop(), None)
        previous = -1
        try:
            while True:
                event.clear()
                if self._log._closed:
                    return
                head = await self.read_async(lambda reader: reader.head())
                if head > previous:
                    previous = head
                    yield head
                await event.wait()
        finally:
            with self._log._listener_lock:
                self._log._listeners.pop(event, None)
                self._log._notify_in_flight.discard(event)

    async def follow(
        self, *, after_seq: int = -1, poll_interval: float | None = None
    ) -> AsyncGenerator[Message, None]:
        """先订阅再从日志追赶；通知只唤醒，正文和进度始终来自 seq。"""
        event = asyncio.Event()
        with self._log._listener_lock:
            if self._log._closed:
                return
            self._log._listeners[event] = (asyncio.get_running_loop(), None)
        try:
            while True:
                event.clear()
                if self._log._closed:
                    return
                messages = self.read(after_seq=after_seq)
                for message in messages:
                    after_seq = message.seq
                    yield message
                if poll_interval is None:
                    _ = await event.wait()
                else:
                    try:
                        _ = await asyncio.wait_for(event.wait(), poll_interval)
                    except TimeoutError:
                        pass
        finally:
            with self._log._listener_lock:
                self._log._listeners.pop(event, None)
                self._log._notify_in_flight.discard(event)


class _IncrementalMessageReader(MessageReader):
    """回复范围内复用已提交前缀；外部修改使整份派生视图失效。"""

    def __init__(self, log: MessageLog, session_id: str):
        super().__init__(log, session_id)
        self._prefix: tuple[int, tuple[Message, ...]] | None = None

    def incremental(self) -> MessageReader:
        return self

    def _external_version(self) -> int | None:
        """只检查原 writer 连接的外部版本；writer 忙时不等待或复用前缀。"""
        log = self._log
        if not log._writer_lock.acquire(blocking=False):
            return None
        try:
            if log._closed or log._writer_connection.in_transaction:
                return None
            return log._writer_connection.execute("PRAGMA data_version").fetchone()[0]
        finally:
            log._writer_lock.release()

    async def snapshot_async(self, *, through_seq: int) -> tuple[Message, ...]:
        """同步和异步共用增量读取；取消等待实际 worker 结束。"""
        self._check_async_snapshot()
        return await run_file_io(lambda: self.snapshot(through_seq=through_seq))

    def scan(
        self, consume: Callable[[Iterable[Message]], _T], *,
        after_seq: int = -1, through_seq: int | None = None, source: str | None = None,
    ) -> _T:
        """在同一只读快照内复用前缀并补读尾部，回调不能跨 await。"""
        if after_seq < -1:
            raise ValueError("读取需要正 limit 和不小于 -1 的 after_seq")
        # 1. 已有事务可能更早或尚未提交，必须读取其实际视图。
        if self._log._reads.current is not None:
            return super().scan(consume, after_seq=after_seq, through_seq=through_seq, source=source)
        prefix = self._prefix
        has_revision = self._log._has_prefix_revision
        before = None if has_revision else self._external_version()
        with self._log._read() as connection:
            if connection is self._log._writer_connection:
                return super().scan(consume, after_seq=after_seq, through_seq=through_seq, source=source)
            if has_revision:
                # 标记和消息共享当前 RO 快照，不受另一个线程的正常写入影响。
                before = connection.execute(
                    "SELECT revision FROM message_prefix_revision WHERE singleton=1"
                ).fetchone()[0]
            cached = () if prefix is None or before != prefix[0] else prefix[1]
            # 固定 RO 快照之后，原 writer 的正常追加不会改写已缓存的前缀。
            head = self.head()
            if through_seq is not None:
                head = min(head, through_seq)
            previous = cached[-1].seq if cached else -1
            messages = tuple(message for message in cached
                             if after_seq < message.seq <= head
                             and (source is None or message.source == source))
            if head > max(previous, after_seq):
                messages += super().scan(tuple, after_seq=max(previous, after_seq),
                                         through_seq=head, source=source)
            # 2. 外部修改可能夹在版本检查与 RO 快照之间；只在原快照内重读，
            # 不重启快照，也不向调用者交付旧前缀与新尾部的混合结果。
            stable = has_revision or before is not None and before == self._external_version()
            if not stable and cached:
                messages = super().scan(tuple, after_seq=after_seq, through_seq=head, source=source)
            with closing(message for message in messages) as rows:
                result = consume(rows)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError("消息扫描回调必须同步，不能跨 await")
        # 3. 发布完整已提交前缀；来源筛选或增量查询不覆盖完整视图。
        if stable and source is None and after_seq == -1:
            assert before is not None
            self._prefix = (before, messages)
        return result


async def _run_commit(
    log: "MessageLog", operation: Callable[[], _T], on_commit: Callable[[_T], None] | None,
) -> _T:
    """排空纯存储操作；取消也先在原 loop 交付已提交收据。

    listener 唤醒不在 worker 线程内联投递：worker 内联的 call_soon_threadsafe
    排在 executor 完成回调之前，follower 的追赶读会插队到提交者延续前面。
    改为提交者恢复后在 loop 上交付（flush_notify），唤醒语义不变。
    """
    committed: list[_T] = []

    def write() -> _T:
        log._defer_notify.active = True
        try:
            result = operation()
        finally:
            log._defer_notify.active = False
        committed.append(result)
        return result

    try:
        result = await run_file_io(write)
    except asyncio.CancelledError as cancellation:
        if committed:
            log.flush_notify()
            if on_commit is not None:
                try:
                    on_commit(committed[0])
                except BaseException as failure:
                    raise BaseExceptionGroup(
                        "提交已完成，但通知失败且调用者取消", [cancellation, failure],
                    ) from None
        raise
    log.flush_notify()
    if on_commit is not None:
        on_commit(result)
    return result


@dataclass(frozen=True, slots=True)
class _PreparedMessage:
    """固定 loop 上验证的内容引用与投影，供 SQL 提交重新核对实时事实。"""

    body: Body
    metadata: Mapping[str, object]
    bindings: frozenset[str]
    artifacts: tuple[str, ...]
    session_metadata: Mapping[str, object | None]


@dataclass(frozen=True, slots=True)
class PreparedAppend:
    """固定一次追加的 writer、身份和已校验输入；不授予额外写入权。"""

    _writer: MessageWriter
    _message_id: str
    _body: Body
    _metadata: Mapping[str, object]
    _prepared: _PreparedMessage | None
    _existing: Message | None

    def _append(self, expected_source_head: int | None) -> tuple[Message, bool]:
        previous = self._writer._replay(self._message_id, self._body, self._metadata)
        if previous is not None:
            return previous, False
        if self._prepared is None:
            raise MessageConflict("预备追加的既有消息已被删除")
        return self._writer._insert(
            self._message_id, self._prepared, expected_source_head,
        ), True


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
        metadata_keys: frozenset[str],
        update_metadata: Callable[[Body], Mapping[str, object | None]] | None,
        message_metadata_keys: frozenset[str],
        check_metadata: Callable[[Mapping[str, object]], None] | None,
    ):
        self._log: MessageLog = log
        self._session_id: str = session_id
        self._author: str = author
        self._source: str = source
        self._body_types = body_types
        self._content = content
        self._call_ref = call_ref
        self._check_call = check_call
        self._message_metadata_keys = frozenset(message_metadata_keys)
        self._check_metadata = check_metadata
        self._metadata_keys = frozenset(metadata_keys)
        self._update_metadata = update_metadata
        self._active = True
        self._grant_lock = threading.RLock()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def source(self) -> str:
        return self._source

    def check(self, body: Body) -> None:
        """预先核对提交权限与引用；不占序号，实际提交仍在事务内核对实时状态。"""
        with self._log._read():
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
        with self._grant_lock:
            self._active = False

    def append(
        self, message_id: str, body: Body, *, expected_source_head: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Message:
        """原子追加消息及其绑定 owner 计算的元数据变化，重放不重复更新。"""
        # SQL owner 排序追加；grant 只在 _insert 中持有短锁。
        with self._log._lock:
            return self._log._write(
                lambda: self._append(
                    message_id, body, expected_source_head=expected_source_head, metadata=metadata,
                )
            )

    def _metadata(self, body: Body, metadata: Mapping[str, object] | None) -> Mapping[str, object]:
        """固定 grant 范围和 metadata 表示，不调用当前内容 owner。"""
        self._check_grant(body)
        value = freeze_metadata({} if metadata is None else metadata)
        if self._check_metadata is None and not value.keys() <= self._message_metadata_keys:
            raise PermissionError("writer 未获授这些 Message metadata 命名空间")
        if value and not self._log._has_metadata:
            raise RuntimeError("Message metadata 尚未完成 yoyo 迁移")
        return value

    def _replay(
        self, message_id: str, body: Body, message_metadata: Mapping[str, object],
    ) -> Message | None:
        """旧身份先核对不可变内容；不重新执行已卸载 owner 的校验。"""
        connection = self._log._connection
        old = connection.execute(
            "SELECT * FROM messages WHERE id=?", (message_id,)
        ).fetchone()
        if old is None:
            if isinstance(body, ToolResult) and body._legacy_unknown:
                # 旧 unknown 仍只能重放；普通新正文留到实际 INSERT 时编码。
                encode_body(body, allow_legacy=False)
            return None
        payload = encode_body(body)
        if (old["session_key"], old["author"], old["source"], old["body"]) != (
            self._session_id,
            self._author,
            self._source,
            payload,
        ):
            raise MessageConflict("message_id 已用于不同的不可变内容")
        previous = _message(old)
        if (json.dumps(json_value(previous.metadata), sort_keys=True)
                != json.dumps(json_value(message_metadata), sort_keys=True)):
            raise MessageConflict("message_id 已用于不同的不可变 metadata")
        return previous

    def _prepare(self, body: Body, metadata: Mapping[str, object]) -> _PreparedMessage:
        """在调用者 scope 内计算纯投影和内容引用，线程不得调用 Context。"""
        if self._check_metadata is not None and metadata:
            self._check_metadata(metadata)
        bindings, artifacts = self._references(body)
        changes = {} if self._update_metadata is None else self._update_metadata(body)
        if not set(changes) <= self._metadata_keys:
            raise PermissionError("writer 未获授这些 Session metadata 键")
        return _PreparedMessage(
            body, metadata, frozenset(bindings), artifacts,
            cast(Mapping[str, object | None], freeze_json(dict(changes))),
        )

    async def prepare_async(
        self, message_id: str, body: Body, *, metadata: Mapping[str, object] | None = None,
    ) -> PreparedAppend:
        """在原 scope 验证新内容，返回可用于纯 SQL owner 事务的固定追加。"""
        self._log._check_async_operation()
        message_metadata = self._metadata(body, metadata)
        # 重放先于当前 owner 校验，已提交内容不依赖仍安装原插件。
        def replay() -> Message | None:
            with self._log._read():
                return self._replay(message_id, body, message_metadata)

        existing = await run_file_io(replay)
        with self._log._read():
            if existing is None and not self._active:
                raise WriterExpired("writer 已失效")
            prepared = None if existing is not None else self._prepare(body, message_metadata)
        return PreparedAppend(self, message_id, body, message_metadata, prepared, existing)

    async def append_async(
        self, message_id: str, body: Body, *,
        expected_source_head: int | None = None, metadata: Mapping[str, object] | None = None,
        on_commit: Callable[[Message, bool], None] | None = None,
    ) -> Message:
        """原 scope 验证后执行纯 SQL；取消仍在原 loop 交付真实收据。"""
        prepared = await self.prepare_async(message_id, body, metadata=metadata)
        if prepared._existing is not None:
            if on_commit is not None:
                on_commit(prepared._existing, False)
            return prepared._existing

        def write() -> tuple[Message, bool]:
            with self._log._lock:
                return self._log._write(lambda: prepared._append(expected_source_head))

        message, _ = await _run_commit(
            self._log, write, None if on_commit is None else lambda result: on_commit(*result),
        )
        return message

    def _append(
        self, message_id: str, body: Body, *, expected_source_head: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Message:
        message_metadata = self._metadata(body, metadata)
        previous = self._replay(message_id, body, message_metadata)
        if previous is not None:
            return previous
        if not self._active:
            raise WriterExpired("writer 已失效")
        prepared = self._prepare(body, message_metadata)
        return self._insert(message_id, prepared, expected_source_head)

    def _insert(
        self, message_id: str, prepared: _PreparedMessage, expected_source_head: int | None,
    ) -> Message:
        """只读取固定数据与 SQL 权威事实；不调用内容、metadata 或 Context owner。"""
        # grant 的短临界区决定本次写入是否已开始；撤权不等待磁盘提交。
        with self._grant_lock:
            if not self._active:
                raise WriterExpired("writer 已失效")
        connection = self._log._connection
        body, message_metadata = prepared.body, prepared.metadata
        payload = encode_body(body)
        head = connection.execute(
            "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE session_key=? AND source=?",
            (self._session_id, self._source),
        ).fetchone()[0]
        if expected_source_head is not None and head != expected_source_head:
            raise SourceHeadConflict(f"来源 head 已变化: {head} != {expected_source_head}")
        binding_ids, artifacts = prepared.bindings, prepared.artifacts
        self._check_artifacts(artifacts)
        if isinstance(body, ToolResult):
            self._check_call_result(body)
        session_metadata = prepared.session_metadata

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
            message_id, self._session_id, seq, now, self._author, self._source, body, message_metadata
        )
        columns = "id,session_key,seq,ts,author,source,body"
        values = (message_id, self._session_id, seq, stamp, self._author, self._source, payload)
        if self._log._has_metadata:
            columns += ",metadata"
            values += (json.dumps(json_value(message_metadata), ensure_ascii=False,
                                  sort_keys=True, separators=(",", ":"), allow_nan=False),)
        _ = connection.execute(
            f"INSERT INTO messages ({columns}) VALUES ({','.join('?' for _ in values)})", values,
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

        # 3. 只合并获授键；失败回滚消息、序号与元数据，不覆盖其他 owner 的键。
        if session_metadata:
            current = self._log.reader(self._session_id).metadata()
            updated = dict(current) if current is not None else {}
            for key, value in session_metadata.items():
                if value is None:
                    _ = updated.pop(key, None)
                else:
                    updated[key] = value
            if updated != (current if current is not None else {}):
                payload = json.dumps(json_value(freeze_json(updated)), ensure_ascii=False, allow_nan=False)
                _ = connection.execute(
                    "UPDATE sessions SET metadata=? WHERE key=?", (payload, self._session_id),
                )

        self._log._changed(type(body))
        return message

    def _check_parts(self, body: Body) -> tuple[set[str], tuple[str, ...]]:
        bindings, artifacts = self._references(body)
        self._check_artifacts(artifacts)
        return bindings, artifacts

    def _references(self, body: Body) -> tuple[set[str], tuple[str, ...]]:
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
        return bindings, tuple(artifacts)

    def _check_artifacts(self, artifacts: tuple[str, ...]) -> None:
        """在提交事务核对引用仍指向已发布资源和已迁移 schema。"""
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

    def _check_call_result(self, body: ToolResult) -> None:
        """在提交事务内校验调用地址与唯一结果，结果 writer 不得跨来源写入。"""
        connection = self._log._connection
        index = body.call_ref.part_index
        # 调用消息在写入时已通过完整解码校验；这里只核对地址归属与目标 part
        # 类型，定向提取替代整行物化，不把完整 body 拉回 Python 重新解码。
        call = connection.execute(
            "SELECT session_key, source,"
            " json_extract(body, '$.kind') AS body_kind,"
            " json_extract(body, ?) AS part_kind"
            " FROM messages WHERE id=?",
            (f"$.parts[{index}].kind", body.call_ref.message_id),
        ).fetchone()
        if call is None or (call["session_key"], call["source"]) != (
            self._session_id,
            self._source,
        ):
            raise ValueError("调用不在 writer 获授的 Session/source 内")
        if call["body_kind"] != "output" or call["part_kind"] != "tool_call":
            raise ValueError("call_ref 未指向真实工具调用")
        previous = connection.execute(
            "SELECT id FROM messages WHERE json_extract(body, '$.kind')='tool_result' "
            "AND json_extract(body, '$.call_ref.message_id')=? "
            "AND json_extract(body, '$.call_ref.part_index')=?",
            (body.call_ref.message_id, index),
        ).fetchone()
        if previous is not None:
            raise MessageConflict("该工具调用已经有结果消息")


@dataclass(frozen=True, slots=True, weakref_slot=True)
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
        """每次读取都以当前快照的 SQL 事实为准；行内容解码复用由 _decode 承担。

        收敛说明：旧版曾按 (提交序号, writer data_version) 双键共享查询结果，
        但版本号捕获自 writer 连接、数据来自只读连接的旧快照，旧结果会被挂在
        新版本键下持续返回（评审 #1140）。版本与数据必须来自同一快照，该机制
        撤除；减少读取的正确手段是调用方合并查询，不是跨连接共享结果。
        """
        with self._log._read(snapshot=False):
            row = self._log._connection.execute(
                "SELECT version,value FROM owner_records WHERE owner=? AND key=?",
                (self._owner, key),
            ).fetchone()
        return None if row is None else self._decode(row)

    def _decode(self, row: sqlite3.Row) -> OwnerRecord:
        """复用当前 SQL 行相同的不可变记录；有界强缓存不受调用者引用周期影响。"""
        key = (row["version"], row["value"])
        with self._log._decode_lock:
            record = self._log._decoded_owners.get(key)
        if record is None:
            record = _owner_record(row)
            if len(row["value"]) <= _DECODE_STRONG_BODY:
                with self._log._decode_lock:
                    if len(self._log._decoded_owners) >= _DECODE_STRONG_LIMIT:
                        self._log._decoded_owners.pop(next(iter(self._log._decoded_owners)))
                    self._log._decoded_owners[key] = record
        return record

    def list(self) -> tuple[tuple[str, OwnerRecord], ...]:
        with self._log._read(snapshot=False):
            rows = self._log._connection.execute(
                "SELECT key,version,value FROM owner_records WHERE owner=? ORDER BY key",
                (self._owner,),
            ).fetchall()
        return tuple((row["key"], self._decode(row)) for row in rows)

    def scan(self, *, start: str, stop: str, limit: int = 100) -> tuple[tuple[str, OwnerRecord], ...]:
        """按 key 倒序读取有界区间 [start, stop)，供 owner 分页读取自身索引。"""
        if not isinstance(start, str) or not isinstance(stop, str) or start >= stop:
            raise ValueError("状态扫描需要递增的 key 区间")
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("状态扫描 limit 必须介于 1 和 1000")
        with self._log._read(snapshot=False):
            rows = self._log._connection.execute(
                "SELECT key,version,value FROM owner_records "
                "WHERE owner=? AND key>=? AND key<? ORDER BY key DESC LIMIT ?",
                (self._owner, start, stop, limit),
            ).fetchall()
        return tuple((row["key"], self._decode(row)) for row in rows)

    def snapshot(self, callback: Callable[[], _T]) -> _T:
        """在同一只读快照内完成同步分页，不授予 SQL 或新的写入权限。"""
        with self._log._read():
            result = callback()
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("存储快照回调必须同步，不能跨 await")
            return result

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

    async def transact_async(
        self, callback: Callable[[OwnerTransaction], _T], *,
        on_commit: Callable[[_T], None] | None = None,
    ) -> _T:
        """纯 SQL owner 工作离开 loop；Context 校验应在调用者 scope 内完成。"""
        self._log._check_async_operation()
        return await _run_commit(self._log, lambda: self.transact(callback), on_commit)


class OwnerTransaction:
    def __init__(self, store: OwnerStore):
        self._store = store
        self._active = True
        self._failed = False

    def source_changed(
        self, reader: MessageReader, source: str, through_seq: int,
    ) -> bool:
        """同一 Core 事务内读来源前提；不借出 SQL 或复制来源状态。"""
        self._check_active()
        if reader._log is not self._store._log:
            raise ValueError("来源前提与 owner transaction 不属于同一 authority")
        return self._perform(lambda: self._store._log._connection.execute(
            "SELECT 1 FROM messages WHERE session_key=? AND source=? AND seq>? "
            "AND json_extract(body, '$.kind') IN ('input','control') LIMIT 1",
            (reader.session_id, source, through_seq),
        ).fetchone() is not None)

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
        version = 0 if expected_version is None else expected_version + 1
        payload = json.dumps(
            frozen,
            default=dict,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        # SQL 直接比较版本；保存新值不需要读取或解码旧正文。
        if expected_version is None:
            cursor = self._store._log._connection.execute(
                "INSERT INTO owner_records VALUES (?,?,?,?) "
                "ON CONFLICT(owner,key) DO NOTHING",
                (self._store._owner, key, version, payload),
            )
        else:
            cursor = self._store._log._connection.execute(
                "UPDATE owner_records SET version=?,value=? "
                "WHERE owner=? AND key=? AND version=?",
                (version, payload, self._store._owner, key, expected_version),
            )
        if cursor.rowcount != 1:
            raise MessageConflict("owner 记录版本已变化")
        record = OwnerRecord(version, cast(Mapping[str, object], frozen))
        # 同轮读回不再重复解码：写入方已持有冻结值与编码原文。
        if len(payload) <= _DECODE_STRONG_BODY:
            with self._store._log._decode_lock:
                owners = self._store._log._decoded_owners
                if len(owners) >= _DECODE_STRONG_LIMIT:
                    owners.pop(next(iter(owners)))
                owners[(version, payload)] = record
        return record

    def append(
        self,
        writer: MessageWriter,
        message_id: str,
        body: Body,
        *,
        expected_source_head: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Message:
        self._check_active()
        if writer._log is not self._store._log:
            raise ValueError("原子提交不能跨存储 authority")
        return self._perform(
            lambda: writer._append(
                message_id, body, expected_source_head=expected_source_head, metadata=metadata
            )
        )

    def append_prepared(
        self, prepared: PreparedAppend, *, expected_source_head: int | None = None,
    ) -> Message:
        """在同一事务核对已准备消息的 grant、引用、身份和来源 head。"""
        self._check_active()
        writer = prepared._writer
        if writer._log is not self._store._log:
            raise ValueError("原子提交不能跨存储 authority")
        return self._perform(lambda: prepared._append(expected_source_head)[0])


def _json_object(raw: str, label: str) -> Mapping[str, object]:
    """持久 JSON 对象在读取边界校验并深冻结，错误带所属记录。"""
    def fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
        values = dict(pairs)
        if len(values) != len(pairs):
            raise ValueError("包含重复键")
        return values
    try:
        if not isinstance(raw, str):
            raise ValueError("JSON 原文必须是文本")
        value = freeze_json(json.loads(raw, object_pairs_hook=fields))
        if not isinstance(value, Mapping):
            raise ValueError("必须是 JSON 对象")
    except (ValueError, TypeError) as error:
        raise ValueError(f"{label} 损坏: {error}") from error
    return cast(Mapping[str, object], value)


def _timestamp(raw: str, label: str) -> datetime:
    try:
        value = datetime.fromisoformat(raw)
        if value.utcoffset() is None:
            raise ValueError("时间缺少时区")
    except (ValueError, TypeError) as error:
        raise ValueError(f"{label} 时间无效: {raw!r}") from error
    return value


def _session_entry(row: sqlite3.Row) -> SessionEntry:
    """目录列与首条 Message 共用数据库快照，标题留给表示边界生成。"""
    key = row["key"]
    first = None if row["first_id"] is None else Message(
        row["first_id"], key, row["first_seq"], _timestamp(row["first_ts"], key),
        row["first_author"], row["first_source"], decode_body(row["first_body"]),
        _message_metadata(row["first_metadata"], key, row["first_id"]),
    )
    return SessionEntry(
        key, _timestamp(row["created_at"], key), _timestamp(row["updated_at"], key),
        decode_attributes(row["attributes"]),
        None if row["metadata"] is None else _json_object(row["metadata"], f"Session {key} metadata"),
        row["head_seq"], row["message_count"], first, row["title"],
    )


def _page_references(
    connection: sqlite3.Connection, messages: tuple[Message, ...],
) -> tuple[Mapping[str, tuple[AttachmentRef, ...]], Mapping[str, Mapping[str, object]]]:
    """批量加载本页已保存的资源关系；损坏引用不能降级为缺少附件或工具名。"""
    attachments: dict[str, list[AttachmentRef]] = {message.message_id: [] for message in messages}
    bindings: dict[str, Mapping[str, object]] = {}
    if messages:
        ids = tuple(message.message_id for message in messages)
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            "SELECT ma.message_id,ma.ordinal,a.* FROM message_attachments ma "
            "LEFT JOIN attachments a ON a.artifact_id=ma.artifact_id "
            f"WHERE ma.message_id IN ({placeholders}) ORDER BY ma.message_id,ma.ordinal", ids,
        ).fetchall()
        for row in rows:
            refs = attachments[row["message_id"]]
            if row["ordinal"] != len(refs) or row["artifact_id"] is None:
                raise ValueError(f"Message {row['message_id']} 附件引用损坏")
            refs.append(_artifact_ref(row))
        rows = connection.execute(
            "SELECT DISTINCT mb.binding_id,b.descriptor FROM message_bindings mb "
            "LEFT JOIN bindings b ON b.binding_id=mb.binding_id "
            f"WHERE mb.message_id IN ({placeholders}) ORDER BY mb.binding_id", ids,
        ).fetchall()
        for row in rows:
            bindings[row["binding_id"]] = _json_object(row["descriptor"], f"binding {row['binding_id']}")
    return MappingProxyType({key: tuple(refs) for key, refs in attachments.items()}), MappingProxyType(bindings)


def _owner_record(row: sqlite3.Row) -> OwnerRecord:
    if type(row["version"]) is not int or row["version"] < 0:
        raise ValueError("owner 记录版本无效")
    value = freeze_json(json.loads(row["value"]))
    if not isinstance(value, Mapping):
        raise ValueError("owner 记录不是 JSON 对象")
    return OwnerRecord(row["version"], cast(Mapping[str, object], value))


def _message_metadata(raw: str, session_id: str, message_id: str) -> Mapping[str, object]:
    """持久附加信息损坏时保留消息定位，不依赖解释它的插件。"""
    try:
        return _json_object(raw, "metadata")
    except (ValueError, TypeError) as error:
        raise ValueError(f"Session {session_id} Message {message_id} metadata 损坏: {error}") from error


def _message(row: sqlite3.Row) -> Message:
    return Message(
        row["id"],
        row["session_key"],
        row["seq"],
        datetime.fromisoformat(row["ts"]),
        row["author"],
        row["source"],
        decode_body(row["body"]),
        _message_metadata(row["metadata"], row["session_key"], row["id"]) if "metadata" in row.keys() else {},
    )


def _artifact_ref(row: sqlite3.Row) -> AttachmentRef:
    """数据库边界只接受已发布的完整附件元数据。"""
    if row["state"] != "ready":
        raise ValueError("附件尚未发布")
    return AttachmentRef(
        row["artifact_id"], AttachmentKind(row["kind"]), row["filename"],
        row["media_type"], row["size_bytes"], row["sha256"],
    )

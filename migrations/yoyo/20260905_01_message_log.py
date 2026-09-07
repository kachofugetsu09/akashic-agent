"""把已有消息改为不可变 body；旧执行记录留给其 owner 的后续迁移。"""

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260831_01_migrate_compaction_plugin_config"}
__transactional__ = False

_NAME = "message-log-v1"
_OLD_SCHEMA = {
    "messages": """CREATE TABLE messages (
        id TEXT PRIMARY KEY, session_key TEXT NOT NULL, seq INTEGER NOT NULL,
        role TEXT NOT NULL, content TEXT, tool_chain TEXT, extra TEXT,
        ts TEXT NOT NULL, UNIQUE(session_key, seq))""",
    "sessions": """CREATE TABLE sessions (
        key TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        last_consolidated INTEGER NOT NULL DEFAULT 0, metadata TEXT,
        last_user_at TEXT, last_proactive_at TEXT, next_seq INTEGER NOT NULL DEFAULT 0)""",
}
_OLD_COLUMNS = (
    ("id", "TEXT", 0, None, 1),
    ("session_key", "TEXT", 1, None, 0),
    ("seq", "INTEGER", 1, None, 0),
    ("role", "TEXT", 1, None, 0),
    ("content", "TEXT", 0, None, 0),
    ("tool_chain", "TEXT", 0, None, 0),
    ("extra", "TEXT", 0, None, 0),
    ("ts", "TEXT", 1, None, 0),
)
_NEW_COLUMNS = (
    ("id", "TEXT", 0, None, 1),
    ("session_key", "TEXT", 1, None, 0),
    ("seq", "INTEGER", 1, None, 0),
    ("ts", "TEXT", 1, None, 0),
    ("author", "TEXT", 1, None, 0),
    ("source", "TEXT", 1, None, 0),
    ("body", "TEXT", 1, None, 0),
)
_NEW_MESSAGE_SCHEMA = """CREATE TABLE messages (
    id TEXT PRIMARY KEY, session_key TEXT NOT NULL, seq INTEGER NOT NULL,
    ts TEXT NOT NULL, author TEXT NOT NULL, source TEXT NOT NULL,
    body TEXT NOT NULL, UNIQUE(session_key, seq))"""
_RESOURCE_SCHEMA = {
    "bindings": "CREATE TABLE bindings (binding_id TEXT PRIMARY KEY, descriptor TEXT NOT NULL)",
    "message_bindings": "CREATE TABLE message_bindings (message_id TEXT NOT NULL REFERENCES messages(id), binding_id TEXT NOT NULL REFERENCES bindings(binding_id), PRIMARY KEY(message_id, binding_id))",
    "message_call_result": "CREATE UNIQUE INDEX message_call_result ON messages (json_extract(body, '$.call_ref.message_id'), json_extract(body, '$.call_ref.part_index')) WHERE json_extract(body, '$.kind')='tool_result'",
}


def _json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value):
    """SQLite BLOB 保留类型及全部字节，不能丢掉引用表中的二进制事实。"""

    def blob(item):
        if isinstance(item, bytes):
            return {"sqlite_blob": item.hex()}
        raise TypeError("SQLite digest 遇到未知值类型")

    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=blob,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _columns(connection, table):
    return tuple(
        tuple(row)[1:] for row in connection.execute(f'PRAGMA table_info("{table}")')
    )


def _rows(connection, table):
    return [
        dict(row)
        for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
    ]


def _check(connection):
    """转换前后都检查完整性；现有坏引用不靠迁移静默修补。"""
    if [row[0] for row in connection.execute("PRAGMA integrity_check")] != ["ok"]:
        raise RuntimeError("Message 迁移发现 SQLite 完整性错误")
    if list(connection.execute("PRAGMA foreign_key_check")):
        raise RuntimeError("Message 迁移发现外键错误")


def _convert(row):
    """保留每个旧标量及原始 JSON 字节，不伪造旧工具的独立 seq。"""
    if not all(
        isinstance(row[key], str) and row[key] for key in ("id", "session_key", "ts")
    ):
        raise ValueError("旧消息缺少稳定身份或时间")
    if type(row["seq"]) is not int or row["seq"] < 0:
        raise ValueError("旧消息 seq 无效")
    if datetime.fromisoformat(row["ts"]).utcoffset() is None:
        raise ValueError("旧消息时间没有时区，请先完成时间迁移")
    if row["role"] not in ("user", "assistant"):
        raise ValueError("旧消息包含未知角色")
    if row["content"] is not None and not isinstance(row["content"], str):
        raise ValueError("旧消息正文不是文本或 null")
    parts = []
    if row["content"] is not None:
        parts.append({"kind": "text", "value": row["content"]})
    for key, expected in (("extra", dict), ("tool_chain", list)):
        raw = row[key]
        if raw is not None and (
            not isinstance(raw, str) or not isinstance(json.loads(raw), expected)
        ):
            raise ValueError(f"旧消息 {key} 结构未知")
    parts.append(
        {
            "kind": "history.provenance",
            "value": {
                "schema": "sessions.messages.v0",
                "role": row["role"],
                "content_was_null": row["content"] is None,
                "extra": row["extra"],
                "extra_sha256": (
                    None
                    if row["extra"] is None
                    else hashlib.sha256(row["extra"].encode("utf-8")).hexdigest()
                ),
            },
        }
    )
    if row["tool_chain"] is not None:
        parts.append(
            {
                "kind": "history.transcript",
                "value": {
                    "schema": "sessions.messages.tool_chain.v0",
                    "raw": row["tool_chain"],
                    "sha256": hashlib.sha256(
                        row["tool_chain"].encode("utf-8")
                    ).hexdigest(),
                    "completeness": "unknown",
                },
            }
        )
    body = {"kind": "input" if row["role"] == "user" else "output", "parts": parts}
    if row["role"] == "assistant":
        # 旧表保存的是已经提交的回复；不把嵌套工具轨迹再次提交执行。
        body.update(finish="complete")
    return {
        "id": row["id"],
        "session_key": row["session_key"],
        "seq": row["seq"],
        "ts": row["ts"],
        "author": "legacy-attribution-unknown",
        "source": "legacy-unattributed",
        "body": _json(body),
    }


def _check_old_schema(connection):
    """只接纳已知消息表及唯一约束，拒绝同版本异构结构。"""
    if _columns(connection, "messages") != _OLD_COLUMNS:
        raise RuntimeError("未知 messages schema，拒绝转换")
    for name, expected_sql in _OLD_SCHEMA.items():
        actual = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (name,)
        ).fetchone()
        if actual is None or _sql(actual[0]) != _sql(expected_sql):
            raise RuntimeError(f"未知 {name} CREATE TABLE identity")
    for name in (*_RESOURCE_SCHEMA, "message_log_migration", "messages_new"):
        if (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name=?", (name,)
            ).fetchone()
            is not None
        ):
            raise RuntimeError(f"迁移目标 {name} 已被占用")
    indexes = list(connection.execute("PRAGMA index_list(messages)"))
    identities = set()
    for index in indexes:
        if (
            not index["unique"]
            or index["partial"]
            or index["origin"] not in ("pk", "u")
        ):
            raise RuntimeError("messages 存在未登记索引")
        identities.add(
            tuple(
                row["name"]
                for row in connection.execute(f'PRAGMA index_info("{index["name"]}")')
            )
        )
    if identities != {("id",), ("session_key", "seq")}:
        raise RuntimeError("messages 缺少身份或 seq 唯一约束")
    # sessions 的旧统计列由后续消费者层处理；此层保留其原值。
    session_columns = {row[0]: row[1:] for row in _columns(connection, "sessions")}
    expected = {
        "key": ("TEXT", 0, None, 1),
        "created_at": ("TEXT", 1, None, 0),
        "updated_at": ("TEXT", 1, None, 0),
        "metadata": ("TEXT", 0, None, 0),
        "last_consolidated": ("INTEGER", 1, "0", 0),
        "last_user_at": ("TEXT", 0, None, 0),
        "last_proactive_at": ("TEXT", 0, None, 0),
        "next_seq": ("INTEGER", 1, "0", 0),
    }
    if session_columns != expected:
        raise RuntimeError("未知 sessions schema，请先完成此前迁移及旧 schema 初始化")
    if list(
        connection.execute(
            "SELECT id FROM messages WHERE session_key NOT IN (SELECT key FROM sessions)"
        )
    ):
        raise RuntimeError("消息缺少 Session")
    if list(
        connection.execute(
            "SELECT key FROM sessions WHERE next_seq <= (SELECT MAX(seq) FROM messages WHERE session_key=sessions.key)"
        )
    ):
        raise RuntimeError("Session next_seq 会复用已有消息序号")
    triggers = list(
        connection.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type='trigger' AND lower(sql) LIKE '%messages%'"
        )
    )
    if any(
        row["tbl_name"] != "messages"
        or row["name"] not in ("messages_ai", "messages_ad", "messages_au")
        for row in triggers
    ):
        raise RuntimeError("messages 存在未登记触发器")
    expected_triggers = {
        "messages_ai": "CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content); END",
        "messages_ad": "CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.rowid, old.content); END",
        "messages_au": "CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.rowid, old.content); INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content); END",
    }
    for name, expected_sql in expected_triggers.items():
        actual = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (name,)
        ).fetchone()
        if actual is not None and _sql(actual[0]) != _sql(expected_sql):
            raise RuntimeError("消息索引触发器定义未知")
    fts = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name='messages_fts'"
    ).fetchone()
    if fts is not None and _sql(fts[0]) != _sql(
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content, content='messages', content_rowid='rowid', tokenize='trigram')"
    ):
        raise RuntimeError("messages_fts 不是已登记的派生索引")
    if list(
        connection.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND lower(sql) LIKE '%messages%'"
        )
    ):
        raise RuntimeError("存在未登记的 messages 视图")
    for row in connection.execute(
        "SELECT name,sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ):
        if (
            "createvirtualtable" in _sql(row["sql"])
            and row["name"] != "messages_fts"
            and re.search(r"\bmessages\b", row["sql"], re.IGNORECASE)
        ):
            raise RuntimeError("存在未登记的 messages 虚拟表引用")


def _reference_digests(connection):
    """保全所有指向原 Message ID 的外键表，拒绝引用退休列。"""
    digests = {}
    tables = list(
        connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    )
    for row in tables:
        table = row["name"]
        quoted = '"' + table.replace('"', '""') + '"'
        foreign_keys = [
            item
            for item in connection.execute(f"PRAGMA foreign_key_list({quoted})")
            if item["table"].casefold() == "messages"
        ]
        if not foreign_keys:
            continue
        if any(
            item["to"] is not None and item["to"].casefold() != "id"
            for item in foreign_keys
        ):
            raise RuntimeError("外部 FK 指向即将退休的 Message 列")
        # 支持 WITHOUT ROWID 引用表；按完整列排序后摘要，而非猜 rowid。
        columns = [
            item["name"] for item in connection.execute(f"PRAGMA table_info({quoted})")
        ]
        order = ",".join('"' + column.replace('"', '""') + '"' for column in columns)
        rows = [
            tuple(item)
            for item in connection.execute(f"SELECT * FROM {quoted} ORDER BY {order}")
        ]
        digests[table] = _digest(rows)
    return digests


def _check_session_values(connection):
    """SQLite affinity 不保证单元类型；拒绝无法继续分配序号的旧状态。"""
    for session in connection.execute("SELECT * FROM sessions"):
        if not isinstance(session["key"], str) or not session["key"]:
            raise ValueError("Session key 为空或类型无效")
        if type(session["next_seq"]) is not int or session["next_seq"] < 0:
            raise ValueError("Session next_seq 类型或值域无效")
        for name in ("created_at", "updated_at"):
            if (
                not isinstance(session[name], str)
                or datetime.fromisoformat(session[name]).utcoffset() is None
            ):
                raise ValueError("Session 时间缺少时区或类型无效")
        raw = session["metadata"]
        if raw is not None and (
            not isinstance(raw, str) or not isinstance(json.loads(raw), dict)
        ):
            raise ValueError("Session metadata 不是 JSON 对象")


def _check_attachment_bindings(connection, messages, tables):
    """核对旧 extra 的附件投影与真实绑定，不让两份矛盾事实进入新日志。"""
    bindings = {}
    if "message_attachments" in tables:
        for row in connection.execute(
            "SELECT * FROM message_attachments ORDER BY message_id,ordinal"
        ):
            bindings.setdefault(row["message_id"], []).append(row)
    for message in messages:
        extra = {} if message["extra"] is None else json.loads(message["extra"])
        ids = extra.get("attachment_ids", [])
        if not isinstance(ids, list) or any(
            not isinstance(item, str) or not item for item in ids
        ):
            raise ValueError("Message attachment_ids 类型无效")
        actual = bindings.get(message["id"], [])
        if ids != [row["artifact_id"] for row in actual]:
            raise ValueError("Message attachment projection 已漂移")
        direction = "inbound" if message["role"] == "user" else "outbound"
        if any(
            row["ordinal"] != ordinal or row["direction"] != direction
            for ordinal, row in enumerate(actual)
        ):
            raise ValueError("Message attachment 顺序或方向已漂移")


def _sql(value):
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


def _verify_receipt(connection):
    """SQLite 已提交而 yoyo ledger 未落账时核对原转换，不再次导入。"""
    for name, statement in {
        "messages": _NEW_MESSAGE_SCHEMA,
        **_RESOURCE_SCHEMA,
    }.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (name,)
        ).fetchone()
        if row is None or _sql(row[0]) != _sql(statement):
            raise RuntimeError(f"迁移后的 {name} schema 已变化")
    row = connection.execute(
        "SELECT receipt FROM message_log_migration WHERE name=?", (_NAME,)
    ).fetchone()
    if row is None:
        raise RuntimeError("新 Message schema 缺少迁移 receipt")
    receipt = json.loads(row[0])
    converted = []
    for message_id, rowid in zip(
        receipt["message_ids"], receipt["rowids"], strict=True
    ):
        message = connection.execute(
            "SELECT * FROM messages WHERE id=? AND rowid=?", (message_id, rowid)
        ).fetchone()
        if message is None:
            raise RuntimeError("迁移后原消息缺失，不能确认重放")
        converted.append(dict(message))
    if _digest(converted) != receipt["target_digest"]:
        raise RuntimeError("迁移后原消息发生变化，不能确认重放")


def migrate_message_log(_connection):
    """备份后在单库事务内转换消息，保留附件、旧执行表和恢复证据。"""
    workspace = current_migration_context().workspace.resolve()
    path = workspace / "sessions.db"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        columns = _columns(connection, "messages")
        if columns == _NEW_COLUMNS:
            _check(connection)
            _verify_receipt(connection)
            return
        # 1. 固定写入边界，所有来源验证先于备份和 DDL。
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("BEGIN IMMEDIATE")
        try:
            _check_old_schema(connection)
            _check_session_values(connection)
            old = _rows(connection, "messages")
            converted = [_convert(row) for row in old]
            sessions = _rows(connection, "sessions")
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            _check_attachment_bindings(connection, old, tables)
            preserved = {
                name: _digest(_rows(connection, name))
                for name in ("turns", "message_attachments")
                if name in tables
            }
            references = _reference_digests(connection)
            _check(connection)
            backup = backup_sqlite_database(
                path, workspace / "backups" / _NAME / uuid4().hex, migration=_NAME
            )
            with closing(sqlite3.connect(backup)) as saved:
                saved.row_factory = sqlite3.Row
                if _digest(_rows(saved, "messages")) != _digest(old):
                    raise RuntimeError("备份与迁移来源不一致")

            # 2. 只替换已验证的消息表示；旧 FTS 是可重建索引，不是正文 owner。
            for name in ("messages_ai", "messages_ad", "messages_au"):
                connection.execute(f'DROP TRIGGER IF EXISTS "{name}"')
            connection.execute("DROP TABLE IF EXISTS messages_fts")
            connection.execute(
                _NEW_MESSAGE_SCHEMA.replace(
                    "CREATE TABLE messages (", "CREATE TABLE messages_new (", 1
                )
            )
            rowids = [
                row[0]
                for row in connection.execute(
                    "SELECT rowid FROM messages ORDER BY rowid"
                )
            ]
            for rowid, row in zip(rowids, converted, strict=True):
                connection.execute(
                    "INSERT INTO messages_new (rowid,id,session_key,seq,ts,author,source,body) VALUES (?,?,?,?,?,?,?,?)",
                    (rowid, *row.values()),
                )
            connection.execute("DROP TABLE messages")
            connection.execute("ALTER TABLE messages_new RENAME TO messages")
            for statement in _RESOURCE_SCHEMA.values():
                connection.execute(statement)
            connection.execute(
                "CREATE TABLE message_log_migration (name TEXT PRIMARY KEY, receipt TEXT NOT NULL)"
            )
            receipt = {
                "backup": str(backup),
                "source_digest": _digest(old),
                "target_digest": _digest(converted),
                "message_ids": [row["id"] for row in old],
                "rowids": rowids,
            }
            connection.execute(
                "INSERT INTO message_log_migration VALUES (?,?)",
                (_NAME, _json(receipt)),
            )

            # 3. 提交前核对完整消息、旧执行证据、序号与附件引用。
            if (
                _columns(connection, "messages") != _NEW_COLUMNS
                or _rows(connection, "messages") != converted
            ):
                raise RuntimeError("Message 转换结果不一致")
            if _rows(connection, "sessions") != sessions:
                raise RuntimeError("Message 转换改变了 Session 元数据或序号")
            if any(
                _digest(_rows(connection, name)) != digest
                for name, digest in preserved.items()
            ):
                raise RuntimeError("Message 转换改变了受保护引用或旧执行记录")
            if _reference_digests(connection) != references | {
                "message_bindings": _digest([])
            }:
                raise RuntimeError("Message 转换改变了外部 FK 引用表")
            _check(connection)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise


steps = [step(migrate_message_log)]

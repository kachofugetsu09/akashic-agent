"""消息附件成为中立引用；删除可由已保留角色精确重建的旧 direction。"""
import json
import re
import sqlite3
from contextlib import closing
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260905_05_message_embeddings"}
__transactional__ = False

_ARTIFACTS = """CREATE TABLE attachments (
    artifact_id TEXT PRIMARY KEY, storage_key TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('image', 'file')),
    filename TEXT, media_type TEXT, size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    state TEXT NOT NULL CHECK (state = 'ready'), created_at TEXT NOT NULL
)"""
_LINKS = """CREATE TABLE message_attachments (
    message_id TEXT NOT NULL, ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    artifact_id TEXT NOT NULL, PRIMARY KEY (message_id, ordinal),
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY (artifact_id) REFERENCES attachments(artifact_id)
)"""
_OLD_LINKS = """CREATE TABLE message_attachments (
    message_id TEXT NOT NULL, ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    artifact_id TEXT NOT NULL, direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    PRIMARY KEY (message_id, ordinal),
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY (artifact_id) REFERENCES attachments(artifact_id)
)"""
_INDEX = "CREATE INDEX idx_message_attachments_artifact ON message_attachments(artifact_id, message_id, ordinal)"


def _sql(value):
    tokens = re.findall(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|[A-Za-z_][A-Za-z_0-9]*|[^\s]", value)
    words = [token if token.startswith("'") else token.strip('"').lower() for token in tokens]
    for index in range(len(words) - 2):
        if words[index:index + 3] == ["if", "not", "exists"]:
            del words[index:index + 3]
            break
    return "".join(words).rstrip(";")


def _check(connection):
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise RuntimeError("附件迁移发现完整性错误")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise RuntimeError("附件迁移发现外键错误")


def migrate_message_artifacts(_ledger):
    """逐行证明旧方向冗余，备份后原子替换关系表，不修改消息或附件。"""
    workspace = current_migration_context().workspace
    path = workspace / "sessions.db"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        _check(connection)
        old = False
        missing = []
        for name, wanted in (("attachments", _ARTIFACTS), ("message_attachments", _LINKS),
                             ("idx_message_attachments_artifact", _INDEX)):
            row = connection.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()
            if row is None:
                missing.append((name, wanted))
            elif name == "message_attachments" and _sql(row[0]) == _sql(_OLD_LINKS):
                old = True
            elif _sql(row[0]) != _sql(wanted):
                raise RuntimeError(f"{name} schema lineage 不匹配")
        if not old and not missing:
            return

        # 1. 方向必须已能从不可变 provenance 唯一重建；未知内容在备份和写入前拒绝。
        rows = []
        if old:
            for message_id, ordinal, artifact_id, direction, body in connection.execute(
                "SELECT ma.message_id,ma.ordinal,ma.artifact_id,ma.direction,m.body "
                "FROM message_attachments ma JOIN messages m ON m.id=ma.message_id "
                "ORDER BY ma.message_id,ma.ordinal"
            ):
                provenance = [part["value"] for part in json.loads(body)["parts"]
                              if part.get("kind") == "history.provenance"]
                if len(provenance) != 1 or provenance[0].get("schema") != "sessions.messages.v0":
                    raise RuntimeError("旧附件方向缺少可重建的消息出处")
                role = provenance[0].get("role")
                if role not in {"user", "assistant"} or direction != {"user": "inbound", "assistant": "outbound"}[role]:
                    raise RuntimeError("旧附件方向与已保留消息角色不一致")
                rows.append((message_id, ordinal, artifact_id))
            for name, in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                quoted = '"' + name.replace('"', '""') + '"'
                if any(row[2] == "message_attachments" for row in connection.execute(f"PRAGMA foreign_key_list({quoted})")):
                    raise RuntimeError("存在未知表引用附件关系，不能重建")

        # 2. 备份保留旧完整表；新表只保留不可推导的消息、位置和附件身份。
        backup_sqlite_database(path, workspace / "backups/message-artifacts-v1" / uuid4().hex,
                               migration="20260905_06_message_artifacts")
        for name, statement in missing:
            if name != "idx_message_attachments_artifact":
                connection.execute(statement)
        if old:
            connection.execute(_LINKS.replace("CREATE TABLE message_attachments", "CREATE TABLE message_attachments_next"))
            connection.executemany("INSERT INTO message_attachments_next VALUES (?,?,?)", rows)
            connection.execute("DROP TABLE message_attachments")
            connection.execute("ALTER TABLE message_attachments_next RENAME TO message_attachments")
            connection.execute(_INDEX)
            if connection.execute("SELECT * FROM message_attachments ORDER BY message_id,ordinal").fetchall() != rows:
                raise RuntimeError("附件关系迁移前后不一致")
        elif any(name == "idx_message_attachments_artifact" for name, _ in missing):
            connection.execute(_INDEX)
        _check(connection)


steps = [step(migrate_message_artifacts)]

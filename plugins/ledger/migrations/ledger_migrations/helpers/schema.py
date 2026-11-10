"""已发布迁移的冻结 SQL 与结构校验；不导入运行时实现。"""
import re
import sqlite3
from collections.abc import Mapping

ARTIFACT_SCHEMA = {
    "attachments": """CREATE TABLE IF NOT EXISTS attachments (
    artifact_id TEXT PRIMARY KEY,
    storage_key TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('image', 'file')),
    filename TEXT,
    media_type TEXT,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    sha256 TEXT NOT NULL CHECK (
        length(sha256) = 64
        AND sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    state TEXT NOT NULL CHECK (state = 'ready'),
    created_at TEXT NOT NULL
)""",
    "attachment_imports": """CREATE TABLE IF NOT EXISTS attachment_imports (
    artifact_id TEXT PRIMARY KEY,
    storage_key TEXT NOT NULL UNIQUE,
    expected_size_bytes INTEGER NOT NULL
        CHECK (expected_size_bytes >= 0),
    expected_sha256 TEXT NOT NULL CHECK (
        length(expected_sha256) = 64
        AND expected_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    phase TEXT NOT NULL CHECK (
        phase IN (
            'prepared', 'file_published', 'artifact_committed'
        )
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error TEXT
)""",
}


MESSAGE_SOURCE_INDEX_SCHEMA = """CREATE INDEX IF NOT EXISTS message_source_seq
    ON messages (session_key, source, seq);"""


MESSAGE_BODY_KIND_INDEX_SCHEMA = """CREATE INDEX IF NOT EXISTS message_source_kind_seq
    ON messages (session_key, source, json_extract(body, '$.kind'), seq,
                 json_extract(body, '$.finish'));"""


_OLD_SESSION_SCHEMA = """CREATE TABLE sessions (
    key TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    metadata TEXT, next_seq INTEGER NOT NULL DEFAULT 0
);"""


_SESSION_ATTRIBUTES_COLUMN = (
    "attributes TEXT NOT NULL DEFAULT '{\"learning\": \"eligible\", \"visibility\": \"listed\"}'"
)


_SESSION_DELETED_COLUMN = "deleted_at TEXT"


_SESSION_TITLE_COLUMN = "title TEXT"


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

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from session.artifacts import AttachmentKind, AttachmentRef

_ATTACHMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_ATTACHMENT_SHA256_RE = re.compile(r"[0-9a-f]{64}")

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

@dataclass(frozen=True)
class AttachmentArtifactRecord:
    """存储行只在已校验的附件引用之外保留物理位置和发布时间。"""

    ref: AttachmentRef
    storage_key: str
    created_at: str


@dataclass(frozen=True)
class AttachmentImportRecord:
    """记录一次 attachment 文件发布与 metadata commit 的恢复状态。"""

    artifact_id: str
    storage_key: str
    expected_size_bytes: int
    expected_sha256: str
    phase: str
    created_at: str
    updated_at: str
    error: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or _ATTACHMENT_ID_RE.fullmatch(self.artifact_id) is None:
            raise ValueError("attachment artifact_id 必须是 1..256 字符安全 identity")
        _ = _attachment_storage_key(self.artifact_id, self.storage_key)
        if type(self.expected_size_bytes) is not int or self.expected_size_bytes < 0:
            raise ValueError("attachment expected_size_bytes 必须是非负整数")
        if not isinstance(self.expected_sha256, str) or _ATTACHMENT_SHA256_RE.fullmatch(self.expected_sha256) is None:
            raise ValueError("attachment expected_sha256 必须是 64 位小写十六进制")
        if self.phase not in ("prepared", "file_published", "artifact_committed"):
            raise ValueError("attachment import phase 无效")
        if any(not isinstance(value, str) or not value for value in (self.created_at, self.updated_at)):
            raise ValueError("attachment import 时间不得为空")
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise ValueError("attachment import error 必须是非空字符串或 None")


@dataclass(frozen=True)
class AttachmentIntegrityReport:
    """汇总附件 metadata 与导入记录的只读完整性证据。"""

    artifact_count: int
    incomplete_import_ids: tuple[str, ...]


def _attachment_storage_key(artifact_id: str, storage_key: str) -> PurePosixPath:
    """校验 Core artifact 的唯一 workspace-relative storage identity。"""

    storage_path = PurePosixPath(storage_key)
    if storage_path.parts != (
        "uploads",
        "artifacts",
        f"{artifact_id}.bin",
    ):
        raise ValueError(
            "attachment storage_key 必须是 uploads/artifacts/<artifact_id>.bin"
        )
    return storage_path


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


class ArtifactStore:
    """独占附件元数据与导入记录；不持有消息读写或删除接口。"""

    def __init__(self, path: str | Path):
        self.db_path = Path(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        _ = self._conn.execute("PRAGMA foreign_keys=ON")
        try:
            with self._conn:
                # 1. 两个表只有一条已知 lineage，拒绝部分损坏或未知约束。
                _ = self._conn.execute("BEGIN IMMEDIATE")
                existing: set[str] = set()
                for name, expected in ARTIFACT_SCHEMA.items():
                    row = self._conn.execute(
                        "SELECT sql FROM sqlite_master WHERE name=?", (name,)
                    ).fetchone()
                    if row is not None:
                        if _sql(row["sql"]) != _sql(expected):
                            raise RuntimeError(f"{name} schema 不匹配，请先完成对应迁移")
                        existing.add(name)
                if "attachment_imports" in existing and "attachments" not in existing:
                    raise RuntimeError("attachment_imports 存在但 attachments 缺失")
                if "attachments" in existing and "attachment_imports" not in existing:
                    if self._conn.execute("SELECT 1 FROM attachments LIMIT 1").fetchone():
                        raise RuntimeError("已有附件缺少 attachment_imports，不能猜测导入事实")
                # 2. 只初始化空存储，不改写既有 metadata 或消息表。
                for statement in ARTIFACT_SCHEMA.values():
                    _ = self._conn.execute(statement)
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def begin_attachment_import(
        self,
        *,
        artifact_id: str,
        storage_key: str,
        expected_size_bytes: int,
        expected_sha256: str,
        created_at: str,
    ) -> AttachmentImportRecord:
        """在写文件前持久化或复用完全相同的 attachment import intent。"""

        intent = AttachmentImportRecord(
            artifact_id, storage_key, expected_size_bytes, expected_sha256,
            "prepared", created_at, created_at, None,
        )
        with self._lock:
            with self._conn:
                _ = self._conn.execute("BEGIN IMMEDIATE")
                existing = self._conn.execute(
                    """
                    SELECT artifact_id, storage_key, expected_size_bytes,
                           expected_sha256, phase, created_at, updated_at, error
                    FROM attachment_imports
                    WHERE artifact_id = ?
                    """,
                    (artifact_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["storage_key"]) != storage_key
                        or int(existing["expected_size_bytes"])
                        != expected_size_bytes
                        or str(existing["expected_sha256"]) != expected_sha256
                    ):
                        raise RuntimeError(
                            f"attachment import identity 已漂移: {artifact_id}"
                        )
                    return _import_record(existing)
                _ = self._conn.execute(
                    """
                    INSERT INTO attachment_imports (
                        artifact_id, storage_key, expected_size_bytes,
                        expected_sha256, phase, created_at, updated_at, error
                    ) VALUES (?, ?, ?, ?, 'prepared', ?, ?, NULL)
                    """,
                    (
                        artifact_id,
                        storage_key,
                        expected_size_bytes,
                        expected_sha256,
                        created_at,
                        created_at,
                    ),
                )
        return intent

    def mark_attachment_import_file_published(
        self,
        artifact_id: str,
        *,
        updated_at: str,
    ) -> None:
        """在目录 fsync 后持久化 file_published 恢复边界。"""

        if not isinstance(updated_at, str) or not updated_at:
            raise ValueError("attachment import updated_at 不得为空")
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE attachment_imports
                    SET phase = 'file_published', updated_at = ?, error = NULL
                    WHERE artifact_id = ? AND phase = 'prepared'
                    """,
                    (updated_at, artifact_id),
                )
        if cur.rowcount != 1:
            raise RuntimeError(
                f"attachment import 不在 prepared: {artifact_id}"
            )

    def record_attachment_import_error(
        self,
        artifact_id: str,
        *,
        error: str,
        updated_at: str,
    ) -> None:
        """记录非终态 import 错误，不删除或伪终结已有 bytes。"""

        if not isinstance(error, str) or not error:
            raise ValueError("attachment import error 不得为空")
        if not isinstance(updated_at, str) or not updated_at:
            raise ValueError("attachment import updated_at 不得为空")
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE attachment_imports
                    SET updated_at = ?, error = ?
                    WHERE artifact_id = ? AND phase != 'artifact_committed'
                    """,
                    (updated_at, error, artifact_id),
                )
        if cur.rowcount != 1:
            raise RuntimeError(
                f"attachment import 不存在或已经 committed: {artifact_id}"
            )

    def attachment_import(self, artifact_id: str) -> AttachmentImportRecord | None:
        """读取一个 attachment import 的 durable 恢复状态。"""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT artifact_id, storage_key, expected_size_bytes,
                       expected_sha256, phase, created_at, updated_at, error
                FROM attachment_imports
                WHERE artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return _import_record(row)

    def incomplete_attachment_imports(self) -> tuple[AttachmentImportRecord, ...]:
        """同一查询读取所有非终态导入，不静默丢失中途变化的记录。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM attachment_imports WHERE phase != 'artifact_committed' "
                "ORDER BY created_at, artifact_id"
            ).fetchall()
        return tuple(_import_record(row) for row in rows)

    def validate_attachment_metadata_integrity(self) -> AttachmentIntegrityReport:
        """验证附件 metadata 与 import 终态，不解释消息正文或绑定。"""

        with self._lock, self._conn:
            # 1. 同一读快照核对两个 owner 表中的真实记录。
            _ = self._conn.execute("BEGIN")
            artifacts = self._conn.execute(
                """
                SELECT *
                FROM attachments
                ORDER BY artifact_id
                """
            ).fetchall()
            for artifact in artifacts:
                _ = _artifact_record(artifact)
            import_rows = self._conn.execute(
                """
                SELECT *
                FROM attachment_imports
                ORDER BY artifact_id
                """
            ).fetchall()
            for row in import_rows:
                _ = _import_record(row)
            imports = {str(row["artifact_id"]): row for row in import_rows}
            artifact_ids = {str(row["artifact_id"]) for row in artifacts}
            for artifact in artifacts:
                artifact_id = str(artifact["artifact_id"])
                intent = imports.get(artifact_id)
                if (
                    intent is None
                    or str(intent["phase"]) != "artifact_committed"
                    or str(intent["storage_key"]) != str(artifact["storage_key"])
                    or int(intent["expected_size_bytes"])
                    != int(artifact["size_bytes"])
                    or str(intent["expected_sha256"]) != str(artifact["sha256"])
                ):
                    raise ValueError(
                        f"attachment committed intent 已漂移: {artifact_id}"
                    )
            terminal_without_artifact = sorted(
                artifact_id
                for artifact_id, intent in imports.items()
                if str(intent["phase"]) == "artifact_committed"
                and artifact_id not in artifact_ids
            )
            if terminal_without_artifact:
                raise ValueError(
                    "attachment committed intent 缺少 artifact: "
                    + ", ".join(terminal_without_artifact)
                )

            incomplete_rows = self._conn.execute(
                """
                SELECT artifact_id
                FROM attachment_imports
                WHERE phase != 'artifact_committed'
                ORDER BY created_at, artifact_id
                """
            ).fetchall()
        return AttachmentIntegrityReport(
            artifact_count=len(artifacts),
            incomplete_import_ids=tuple(
                str(row["artifact_id"]) for row in incomplete_rows
            ),
        )

    def register_ready_attachment(
        self, *, ref: AttachmentRef, storage_key: str, created_at: str,
    ) -> AttachmentArtifactRecord:
        """在同一事务登记 ready 附件并完成对应的导入记录。"""

        # 1. 引用已经由领域类型校验，存储只校验物理位置与导入身份。
        _ = _attachment_storage_key(ref.artifact_id, storage_key)
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("attachment created_at 不得为空")
        artifact_id = ref.artifact_id
        size_bytes = ref.size_bytes
        sha256 = ref.sha256

        # 2. INSERT-only publication 禁止复用 identity 或 storage path。
        with self._lock:
            with self._conn:
                _ = self._conn.execute("BEGIN IMMEDIATE")
                intent = self._conn.execute(
                    """
                    SELECT storage_key, expected_size_bytes, expected_sha256, phase
                    FROM attachment_imports
                    WHERE artifact_id = ?
                    """,
                    (artifact_id,),
                ).fetchone()
                if intent is None:
                    raise RuntimeError(
                        f"attachment 缺少 durable import intent: {artifact_id}"
                    )
                if str(intent["phase"]) != "file_published":
                    raise RuntimeError(
                        f"attachment file 尚未发布: {artifact_id}:{intent['phase']}"
                    )
                if (
                    str(intent["storage_key"]) != storage_key
                    or int(intent["expected_size_bytes"]) != size_bytes
                    or str(intent["expected_sha256"]) != sha256
                ):
                    raise RuntimeError(
                        f"attachment publication 与 durable intent 不一致: {artifact_id}"
                    )
                _ = self._conn.execute(
                    """
                    INSERT INTO attachments (
                        artifact_id, storage_key, kind, filename, media_type,
                        size_bytes, sha256, state, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?)
                    """,
                    (
                        artifact_id,
                        storage_key,
                        ref.kind.value,
                        ref.filename,
                        ref.media_type,
                        size_bytes,
                        sha256,
                        created_at,
                    ),
                )
                _ = self._conn.execute(
                    """
                    UPDATE attachment_imports
                    SET phase = 'artifact_committed', updated_at = ?, error = NULL
                    WHERE artifact_id = ? AND phase = 'file_published'
                    """,
                    (created_at, artifact_id),
                )
        record = self.get_attachment(artifact_id)
        if record is None:
            raise RuntimeError(f"attachment publication 未返回记录: {artifact_id}")
        return record

    def get_attachment(self, artifact_id: str) -> AttachmentArtifactRecord | None:
        """读取一个 artifact 的权威 metadata。"""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT artifact_id, storage_key, kind, filename, media_type,
                       size_bytes, sha256, state, created_at
                FROM attachments
                WHERE artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return _artifact_record(row)

    def list_attachments(self) -> tuple[AttachmentArtifactRecord, ...]:
        """按 identity 列出 ready artifact metadata，不暴露任意 SQL。"""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT artifact_id, storage_key, kind, filename, media_type,
                       size_bytes, sha256, state, created_at
                FROM attachments
                ORDER BY artifact_id
                """
            ).fetchall()
        return tuple(_artifact_record(row) for row in rows)


def _artifact_record(row: sqlite3.Row) -> AttachmentArtifactRecord:
    """反序列化边界校验实际记录，拒绝路径或 ready 状态漂移。"""
    ref = AttachmentRef(
        artifact_id=row["artifact_id"], kind=AttachmentKind(row["kind"]),
        filename=row["filename"], media_type=row["media_type"],
        size_bytes=row["size_bytes"], sha256=row["sha256"],
    )
    _ = _attachment_storage_key(ref.artifact_id, row["storage_key"])
    if row["state"] != "ready" or not isinstance(row["created_at"], str) or not row["created_at"]:
        raise ValueError(f"attachment metadata 非法: {ref.artifact_id}")
    return AttachmentArtifactRecord(ref, row["storage_key"], row["created_at"])


def _import_record(row: sqlite3.Row) -> AttachmentImportRecord:
    return AttachmentImportRecord(
        row["artifact_id"], row["storage_key"], row["expected_size_bytes"],
        row["expected_sha256"], row["phase"], row["created_at"], row["updated_at"], row["error"],
    )

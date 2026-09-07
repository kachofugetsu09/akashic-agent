from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from datetime import UTC, datetime
from uuid import uuid4

_SCHEMA = {
    "session_admissions": """CREATE TABLE IF NOT EXISTS session_admissions (
    admission_id TEXT PRIMARY KEY, session_key TEXT NOT NULL, created_at TEXT NOT NULL
)""",
    "idx_session_admissions_session": """CREATE INDEX IF NOT EXISTS idx_session_admissions_session ON session_admissions(session_key)""",
}


def init_session_admissions(connection: sqlite3.Connection) -> None:
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
        raise RuntimeError("session_admissions schema 不完整")
    # 2. DDL 与调用方的写事务一起提交，不修改既有行。
    for statement in _SCHEMA.values():
        _ = connection.execute(statement)


class SessionAdmissions:
    """独占会话处理租约；存在校验与接纳在同一写事务内完成。"""

    def __init__(self, path: str | Path):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            with self._conn:
                _ = self._conn.execute("BEGIN IMMEDIATE")
                init_session_admissions(self._conn)
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def acquire(self, key: str, *, require_existing: bool = True) -> str:
        """保留 Session ID 的输入租约；新会话仍由首条 Message 提交创建。"""
        if not isinstance(key, str) or not key or type(require_existing) is not bool:
            raise ValueError("Session admission 需要非空身份和明确的存在条件")

        admission_id = uuid4().hex
        # 1. 用写事务串行化“存在校验”和租约创建
        with self._lock:
            _ = self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT 1 FROM sessions WHERE key = ?",
                    (key,),
                ).fetchone()
                if require_existing and row is None:
                    self._conn.rollback()
                    raise KeyError(f"session 不存在: {key}")

                # 2. 租约落库后，其他连接的删除操作才能继续竞争写锁
                _ = self._conn.execute(
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
        return admission_id

    def release_admission(self, admission_id: str) -> None:
        """释放已完成入站消息持有的会话处理租约。"""

        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM session_admissions WHERE admission_id = ?",
                (admission_id,),
            )
            self._conn.commit()
        if cur.rowcount != 1:
            raise RuntimeError(f"session admission 不存在: {admission_id}")

    def clear_stale(self) -> None:
        """在唯一 runtime 启动时清理上次异常退出遗留的处理租约。"""

        with self._lock:
            _ = self._conn.execute("DELETE FROM session_admissions")
            self._conn.commit()


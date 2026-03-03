"""
task_note.py — 任务进度持久化工具

提供两个工具：
  task_note   — 记录一条语义笔记（upsert）
  task_recall — 查询已记录的笔记

SQLite 存储，按 namespace（通常为 action_id）隔离。
设计原则：agent 自己决定写什么，而不是机械记录所有 tool call。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.tools.base import Tool

logger = logging.getLogger(__name__)

_DONE_FILENAME = ".done"

_DDL = """
CREATE TABLE IF NOT EXISTS task_notes (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(_DDL)
    conn.commit()
    return conn


class TaskNoteTool(Tool):
    """记录任务进度笔记（跨次运行持久化）。"""

    name = "task_note"
    description = (
        "记录任务进度笔记，供下次运行时读取。"
        "用 namespace 区分不同任务（通常传任务ID），key 是短标识，value 是内容。"
        "例如：记录已找到的论文列表、已完成的步骤、中间结果等。"
        "同一 namespace+key 再次写入会覆盖旧值。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "任务命名空间，通常使用任务ID（如 'research-agent-papers'）",
            },
            "key": {
                "type": "string",
                "description": "笔记键名，如 'found_papers'、'demo_path'、'progress'",
            },
            "value": {
                "type": "string",
                "description": "要记录的内容（字符串，可以是 JSON 格式）",
            },
        },
        "required": ["namespace", "key", "value"],
    }

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _connect(self._db_path)
        return self._conn

    async def execute(self, **kwargs: Any) -> str:
        namespace = str(kwargs.get("namespace", "")).strip()
        key = str(kwargs.get("key", "")).strip()
        value = str(kwargs.get("value", ""))

        if not namespace or not key:
            return json.dumps({"error": "namespace 和 key 不能为空"}, ensure_ascii=False)

        now = datetime.now(timezone.utc).isoformat()
        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO task_notes (namespace, key, value, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (namespace, key, value, now),
            )
            conn.commit()
            logger.debug("[task_note] 写入 namespace=%s key=%s", namespace, key)
            return json.dumps({"ok": True, "namespace": namespace, "key": key}, ensure_ascii=False)
        except Exception as e:
            logger.warning("[task_note] 写入失败: %s", e)
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class TaskRecallTool(Tool):
    """查询任务进度笔记。"""

    name = "task_recall"
    description = (
        "查询之前记录的任务笔记。"
        "传 namespace 列出该任务所有笔记；同时传 key 则查询单条。"
        "任务开始前调用，了解上次做到哪一步。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "任务命名空间（与 task_note 保持一致）",
            },
            "key": {
                "type": "string",
                "description": "可选。不传则返回该 namespace 下所有笔记",
            },
        },
        "required": ["namespace"],
    }

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _connect(self._db_path)
        return self._conn

    async def execute(self, **kwargs: Any) -> str:
        namespace = str(kwargs.get("namespace", "")).strip()
        key = str(kwargs.get("key", "")).strip()

        if not namespace:
            return json.dumps({"error": "namespace 不能为空"}, ensure_ascii=False)

        try:
            conn = self._get_conn()
            if key:
                row = conn.execute(
                    "SELECT value, updated_at FROM task_notes WHERE namespace=? AND key=?",
                    (namespace, key),
                ).fetchone()
                if row is None:
                    return json.dumps({"found": False}, ensure_ascii=False)
                return json.dumps(
                    {"found": True, "key": key, "value": row[0], "updated_at": row[1]},
                    ensure_ascii=False,
                )
            else:
                rows = conn.execute(
                    "SELECT key, value, updated_at FROM task_notes WHERE namespace=? "
                    "ORDER BY updated_at",
                    (namespace,),
                ).fetchall()
                if not rows:
                    return json.dumps({"notes": [], "count": 0}, ensure_ascii=False)
                notes = [{"key": r[0], "value": r[1], "updated_at": r[2]} for r in rows]
                return json.dumps({"notes": notes, "count": len(notes)}, ensure_ascii=False)
        except Exception as e:
            logger.warning("[task_recall] 查询失败: %s", e)
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class TaskDoneTool(Tool):
    """标记当前任务为永久完成，之后不再自动触发。"""

    name = "task_done"
    description = (
        "标记当前任务为已完成（DONE）。"
        "调用后该任务将永久停止自动触发，不会再被执行。"
        "只在任务真正全部完成时调用，未完成时请用 task_note 记录进度等下次继续。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "任务完成摘要，记录做了什么、结果在哪里",
            },
        },
        "required": ["summary"],
    }

    def __init__(self, action_dir: Path) -> None:
        self._done_file = action_dir / _DONE_FILENAME

    async def execute(self, **kwargs: Any) -> str:
        summary = str(kwargs.get("summary", "")).strip()
        try:
            self._done_file.parent.mkdir(parents=True, exist_ok=True)
            self._done_file.write_text(summary, encoding="utf-8")
            logger.info("[task_done] 任务已标记完成 file=%s", self._done_file)
            return json.dumps({"ok": True, "done": True}, ensure_ascii=False)
        except Exception as e:
            logger.warning("[task_done] 标记失败: %s", e)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

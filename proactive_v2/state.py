from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.common.timekit import parse_iso as _parse_iso, utcnow as _utcnow

logger = logging.getLogger(__name__)


def _parse_stored_datetime(raw: object, *, table: str, key: str) -> datetime:
    """解析 SQLite 中已存在的时间，损坏时阻止主动链路继续。"""

    # 1. 读取持久化原文并沿用统一 ISO 解析规则。
    text = str(raw or "").strip()
    timestamp = _parse_iso(text)
    if timestamp is None:
        raise ValueError(
            f"主动状态时间损坏 table={table} key={key} value={text!r}"
        )
    return timestamp


class ProactiveStateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.workspace_dir = self.db_path.parent
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._closed = False
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()
        logger.info(
            "[proactive.state] 初始化完成 db=%s deliveries=%d context_only=%d tick_logs=%d",
            self.db_path,
            self._count_rows("deliveries"),
            self._count_rows("context_only_timestamps"),
            self._count_rows("tick_log"),
        )

    def __del__(self) -> None:
        if not self._closed:
            try:
                self.close()
            except sqlite3.Error as cleanup_error:
                logger.warning(
                    "ProactiveStateStore 析构关闭失败 db=%s err=%s",
                    self.db_path,
                    cleanup_error,
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._db.close()

    def record_tick_log_start(
        self,
        *,
        tick_id: str,
        session_key: str,
        started_at: str,
        gate_exit: str | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO tick_log(tick_id, session_key, started_at, gate_exit)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(tick_id) DO UPDATE SET
                    session_key = excluded.session_key,
                    started_at = excluded.started_at,
                    gate_exit = excluded.gate_exit
                """,
                (tick_id, session_key, started_at, gate_exit),
            )
            self._commit_locked()

    def record_tick_log_finish(
        self,
        *,
        tick_id: str,
        session_key: str,
        started_at: str,
        finished_at: str,
        gate_exit: str | None,
        terminal_action: str | None,
        skip_reason: str,
        steps_taken: int,
        alert_count: int,
        content_count: int,
        context_count: int,
        interesting_ids: list[str],
        discarded_ids: list[str],
        cited_ids: list[str],
        drift_entered: bool,
        final_message: str,
        proactive_effects: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO tick_log(
                    tick_id, session_key, started_at, finished_at, gate_exit,
                    terminal_action, skip_reason, steps_taken, alert_count,
                    content_count, context_count, interesting_ids, discarded_ids,
                    cited_ids, drift_entered, final_message, proactive_effects_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tick_id) DO UPDATE SET
                    session_key = excluded.session_key,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    gate_exit = excluded.gate_exit,
                    terminal_action = excluded.terminal_action,
                    skip_reason = excluded.skip_reason,
                    steps_taken = excluded.steps_taken,
                    alert_count = excluded.alert_count,
                    content_count = excluded.content_count,
                    context_count = excluded.context_count,
                    interesting_ids = excluded.interesting_ids,
                    discarded_ids = excluded.discarded_ids,
                    cited_ids = excluded.cited_ids,
                    drift_entered = excluded.drift_entered,
                    final_message = excluded.final_message,
                    proactive_effects_json = excluded.proactive_effects_json
                """,
                (
                    tick_id,
                    session_key,
                    started_at,
                    finished_at,
                    gate_exit,
                    terminal_action,
                    skip_reason,
                    steps_taken,
                    alert_count,
                    content_count,
                    context_count,
                    json.dumps(interesting_ids, ensure_ascii=False),
                    json.dumps(discarded_ids, ensure_ascii=False),
                    json.dumps(cited_ids, ensure_ascii=False),
                    int(drift_entered),
                    final_message,
                    json.dumps(proactive_effects or [], ensure_ascii=False),
                ),
            )
            self._commit_locked()

    def record_tick_step_log(
        self,
        *,
        tick_id: str,
        step_index: int,
        phase: str,
        tool_name: str,
        tool_call_id: str,
        tool_args: dict[str, Any],
        tool_result_text: str,
        terminal_action_after: str | None,
        skip_reason_after: str,
        interesting_ids_after: list[str],
        discarded_ids_after: list[str],
        cited_ids_after: list[str],
        final_message_after: str,
    ) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO tick_step_log(
                    tick_id, step_index, phase, tool_name, tool_call_id,
                    tool_args_json, tool_result_text, terminal_action_after,
                    skip_reason_after, interesting_ids_after, discarded_ids_after,
                    cited_ids_after, final_message_after
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tick_id,
                    step_index,
                    phase,
                    tool_name,
                    tool_call_id,
                    json.dumps(tool_args, ensure_ascii=False),
                    tool_result_text,
                    terminal_action_after,
                    skip_reason_after,
                    json.dumps(interesting_ids_after, ensure_ascii=False),
                    json.dumps(discarded_ids_after, ensure_ascii=False),
                    json.dumps(cited_ids_after, ensure_ascii=False),
                    final_message_after,
                ),
            )
            self._commit_locked()

    def is_delivery_duplicate(
        self,
        session_key: str,
        delivery_key: str,
        window_hours: int,
        now: datetime | None = None,
    ) -> bool:
        now = now or _utcnow()
        cutoff = now - timedelta(hours=max(window_hours, 1))
        with self._lock:
            row = self._db.execute(
                """
                SELECT sent_at
                FROM deliveries
                WHERE session_key = ? AND delivery_key = ?
                """,
                (session_key, delivery_key),
            ).fetchone()
        if row is None:
            return False
        ts = _parse_stored_datetime(
            row["sent_at"],
            table="deliveries",
            key=f"{session_key}:{delivery_key}",
        )
        if ts < cutoff:
            return False
        logger.info(
            "[proactive.state] 命中发送去重 session=%s delivery_key=%s ts=%s window_hours=%d",
            session_key,
            delivery_key[:16],
            row["sent_at"],
            window_hours,
        )
        return True

    def mark_delivery(
        self,
        session_key: str,
        delivery_key: str,
        now: datetime | None = None,
    ) -> None:
        now = now or _utcnow()
        ts = now.isoformat()
        with self._lock:
            self._db.execute(
                """
                INSERT INTO deliveries(session_key, delivery_key, sent_at)
                VALUES(?, ?, ?)
                ON CONFLICT(session_key, delivery_key) DO UPDATE SET sent_at = excluded.sent_at
                """,
                (session_key, delivery_key, ts),
            )
            self._commit_locked()
        logger.info(
            "[proactive.state] 已记录发送 session=%s delivery_key=%s ts=%s",
            session_key,
            delivery_key[:16],
            ts,
        )

    def count_deliveries_in_window(
        self,
        session_key: str,
        window_hours: int,
        now: datetime | None = None,
    ) -> int:
        now = now or _utcnow()
        cutoff = now - timedelta(hours=window_hours)
        with self._lock:
            row = self._db.execute(
                """
                SELECT COUNT(*)
                FROM deliveries
                WHERE session_key = ? AND sent_at >= ?
                """,
                (session_key, cutoff.isoformat()),
            ).fetchone()
        if row is None:
            raise RuntimeError("deliveries COUNT 查询未返回结果行")
        return int(row[0])

    def get_last_drift_at(self, session_key: str) -> datetime | None:
        return self._get_session_datetime(session_key, "drift_last_at")

    def mark_drift_run(self, session_key: str, now: datetime | None = None) -> None:
        now = now or _utcnow()
        ts = now.isoformat()
        self._set_session_state(session_key, "drift_last_at", ts)
        logger.info("[proactive.state] drift 已记录 session=%s ts=%s", session_key, ts)

    def get_last_context_only_at(self, session_key: str) -> datetime | None:
        return self._get_session_datetime(session_key, "context_only_last_at")

    def mark_context_only_send(
        self, session_key: str, now: datetime | None = None
    ) -> None:
        now = now or _utcnow()
        ts = now.isoformat()
        with self._lock:
            try:
                self._db.execute(
                    """
                    INSERT INTO session_state(session_key, key, value)
                    VALUES(?, ?, ?)
                    ON CONFLICT(session_key, key) DO UPDATE SET value = excluded.value
                    """,
                    (session_key, "context_only_last_at", ts),
                )
                self._db.execute(
                    """
                    INSERT INTO context_only_timestamps(session_key, ts)
                    VALUES(?, ?)
                    """,
                    (session_key, ts),
                )
                self._commit_locked()
            except sqlite3.Error:
                self._db.rollback()
                raise
        logger.info(
            "[proactive.state] context-only 发送已记录 session=%s ts=%s",
            session_key,
            ts,
        )

    def count_context_only_in_window(
        self, session_key: str, window_hours: int, now: datetime | None = None
    ) -> int:
        now = now or _utcnow()
        cutoff = now - timedelta(hours=window_hours)
        with self._lock:
            row = self._db.execute(
                """
                SELECT COUNT(*)
                FROM context_only_timestamps
                WHERE session_key = ? AND ts >= ?
                """,
                (session_key, cutoff.isoformat()),
            ).fetchone()
        if row is None:
            raise RuntimeError("context_only_timestamps COUNT 查询未返回结果行")
        return int(row[0])

    def _init_schema(self) -> None:
        """创建当前表结构并迁移旧版 tick 日志。"""

        # 1. 创建当前版本所需的表与索引
        _ = self._db.executescript("""
            CREATE TABLE IF NOT EXISTS deliveries (
                session_key TEXT NOT NULL,
                delivery_key TEXT NOT NULL,
                sent_at TEXT NOT NULL,
                PRIMARY KEY (session_key, delivery_key)
            );
            CREATE INDEX IF NOT EXISTS idx_deliveries_session_sent
            ON deliveries(session_key, sent_at);

            CREATE TABLE IF NOT EXISTS session_state (
                session_key TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (session_key, key)
            );

            CREATE TABLE IF NOT EXISTS context_only_timestamps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                ts TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_context_only_session_ts
            ON context_only_timestamps(session_key, ts);

            CREATE TABLE IF NOT EXISTS tick_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick_id TEXT NOT NULL UNIQUE,
                session_key TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                gate_exit TEXT,
                terminal_action TEXT,
                skip_reason TEXT,
                steps_taken INTEGER,
                alert_count INTEGER,
                content_count INTEGER,
                context_count INTEGER,
                interesting_ids TEXT,
                discarded_ids TEXT,
                cited_ids TEXT,
                drift_entered INTEGER DEFAULT 0,
                final_message TEXT,
                proactive_effects_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_tick_log_session_started
            ON tick_log(session_key, started_at);

            CREATE TABLE IF NOT EXISTS tick_step_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                phase TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_call_id TEXT NOT NULL,
                tool_args_json TEXT NOT NULL,
                tool_result_text TEXT NOT NULL,
                terminal_action_after TEXT,
                skip_reason_after TEXT,
                interesting_ids_after TEXT NOT NULL,
                discarded_ids_after TEXT NOT NULL,
                cited_ids_after TEXT NOT NULL,
                final_message_after TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tick_step_log_tick_step
            ON tick_step_log(tick_id, step_index);
            """)

        # 2. 补齐 CREATE TABLE IF NOT EXISTS 无法新增的旧库列
        self._ensure_tick_log_effects_column()
        self._commit_locked()

    def _ensure_tick_log_effects_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self._db.execute("PRAGMA table_info(tick_log)").fetchall()
        }
        if "proactive_effects_json" not in columns:
            _ = self._db.execute(
                "ALTER TABLE tick_log ADD COLUMN proactive_effects_json TEXT"
            )

    def _get_session_datetime(self, session_key: str, key: str) -> datetime | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT value
                FROM session_state
                WHERE session_key = ? AND key = ?
                """,
                (session_key, key),
            ).fetchone()
        if row is None:
            return None
        return _parse_stored_datetime(
            row["value"],
            table="session_state",
            key=f"{session_key}:{key}",
        )

    def _set_session_state(self, session_key: str, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO session_state(session_key, key, value)
                VALUES(?, ?, ?)
                ON CONFLICT(session_key, key) DO UPDATE SET value = excluded.value
                """,
                (session_key, key, value),
            )
            self._commit_locked()

    def _commit_locked(self) -> None:
        """提交当前事务，提交失败时回滚未完成的写入。"""

        try:
            self._db.commit()
        except sqlite3.Error:
            self._db.rollback()
            raise

    def _count_rows(self, table: str) -> int:
        with self._lock:
            row = self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        if row is None:
            raise RuntimeError(f"{table} COUNT 查询未返回结果行")
        return int(row[0])

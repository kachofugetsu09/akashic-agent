"""将旧命令与模型调用失败收敛为终态，不改写 Message 正文。"""
from __future__ import annotations

import json
import sqlite3
import tomllib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from .support.session_db_backup import backup_sqlite_database
from .support.plugin_data import builtin_plugin_data_dir

__depends__ = {"20260909_01_tool_provider_views", "20260909_01_close_empty_wake_responses"}
__transactional__ = False

_MOBILE_OLD = """CREATE TABLE mobile_command_receipts (
    device_id TEXT NOT NULL,
    command_id TEXT NOT NULL,
    command_type TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(
        status IN ('processing', 'completed', 'outcome_unknown')
    ),
    reply_type TEXT,
    reply_payload_json TEXT,
    handoff_pending INTEGER NOT NULL DEFAULT 0 CHECK(handoff_pending IN (0, 1)),
    session_id TEXT,
    turn_id TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    PRIMARY KEY(device_id, command_id),
    CHECK(
        ((status IN ('processing', 'outcome_unknown'))
         AND reply_type IS NULL
         AND reply_payload_json IS NULL AND completed_at IS NULL)
        OR
        (status = 'completed' AND reply_type IS NOT NULL
         AND reply_payload_json IS NOT NULL AND completed_at IS NOT NULL)
    ),
    FOREIGN KEY(device_id) REFERENCES mobile_devices(device_id)
        ON DELETE CASCADE
)"""
_MODEL_OLD = """CREATE TABLE model_calls (
    id TEXT PRIMARY KEY NOT NULL,
    binding_json TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('started','success','unknown')),
    usage_json TEXT,
    failure TEXT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    first_token_ms REAL CHECK (first_token_ms >= 0),
    duration_ms REAL CHECK (duration_ms >= 0 AND (first_token_ms IS NULL OR duration_ms >= first_token_ms))
)"""
_MOBILE_NEW = _MOBILE_OLD.replace("'processing', 'completed', 'outcome_unknown'", "'processing', 'completed'").replace(
    "((status IN ('processing', 'outcome_unknown'))", "((status = 'processing')")
_MODEL_NEW = _MODEL_OLD.replace("'started','success','unknown'", "'started','success','error'")

_WAKE_OLD = """
    CREATE TABLE wake_attempts(
        attempt_id TEXT PRIMARY KEY,
        timer_id TEXT NOT NULL,
        scheduled_for TEXT NOT NULL,
        fired_at TEXT NOT NULL,
        mail_watermark INTEGER,
        outcome TEXT NOT NULL CHECK(outcome IN (
            'checking', 'no_due', 'content_insufficient', 'admission_rejected',
            'shared', 'model_skip', 'deferred', 'cancelled_after_fire',
            'delivery_unknown', 'failed'
        )),
        owner TEXT CHECK(owner IN ('alert', 'content', 'drift')),
        detail TEXT,
        completed_at TEXT
    )
"""
_WAKE_NEW = _WAKE_OLD.replace("'delivery_unknown', ", "")
_LEDGER_OLD = """CREATE TABLE deliveries(
    logical_delivery_id TEXT PRIMARY KEY,
    accepted_session_id TEXT NOT NULL,
    accepted_turn_id TEXT NOT NULL,
    target_service TEXT NOT NULL,
    channel TEXT NOT NULL,
    recipient TEXT NOT NULL,
    projection_session_id TEXT NOT NULL,
    body TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN (
        'prepared', 'provider_started', 'delivered', 'projected', 'settled',
        'rejected', 'uncertain'
    )),
    attempt_id TEXT,
    snapshot_id TEXT,
    generation_id TEXT,
    binding_token TEXT,
    provider_receipt_json TEXT,
    projection_message_id TEXT,
    domain_receipt TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(accepted_session_id, accepted_turn_id)
)"""
_LEDGER_NEW = _LEDGER_OLD.replace("'uncertain'", "'failed'")
_LEDGER_INDEX = "CREATE INDEX idx_deliveries_recoverable\nON deliveries(state, created_at, logical_delivery_id)"


def _sql(value: str) -> str:
    return "".join(value.replace('"mobile_command_receipts"', "mobile_command_receipts").replace('"model_calls"', "model_calls").lower().split()).replace("ifnotexists", "").rstrip(";")


def _migrate(path: Path, table: str, old_schema: str, new_schema: str, backups: Path,
             *, versions: tuple[int, int] | None = None, index: str | None = None) -> None:
    """只重建已知回执表；备份、原字段对账和完整性检查先于提交。"""
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        # 1. 核对准确 lineage，拒绝无法保留的自定义索引或触发器。
        row = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if row is None:
            raise RuntimeError(f"{table} schema 缺失")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if _sql(row[0]) == _sql(new_schema) and (versions is None or version == versions[1]):
            return
        if versions is not None and version != versions[0]:
            raise RuntimeError(f"{table} schema version 不匹配: {version}")
        if _sql(row[0]) != _sql(old_schema):
            raise RuntimeError(f"{table} schema 不匹配")
        extras = [row[0] for row in connection.execute("SELECT sql FROM sqlite_master WHERE tbl_name=? AND type IN ('index','trigger') AND sql IS NOT NULL", (table,))]
        if [_sql(value) for value in extras] != ([] if index is None else [_sql(index)]):
            raise RuntimeError(f"{table} 有未知索引或触发器")
        _check_integrity(connection)
        before = [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        backup_sqlite_database(path, backups / uuid4().hex, migration="20260909_02_execution_failures")
        # 2. 仅解释旧终态；started/processing 的 owner 不由迁移推断。
        expected = []
        completed_at = datetime.now(timezone.utc).isoformat()
        for original in before:
            value = dict(original)
            if table == "model_calls" and value["state"] == "unknown":
                value["state"] = "error"
            elif table == "mobile_command_receipts" and value["status"] == "outcome_unknown":
                value.update(status="completed", reply_type=f"{value['command_type']}.error",
                    reply_payload_json=json.dumps({
                        "code": "command_interrupted",
                        "message": "上次命令在终态记录前中断，外部效果可能已经发生，请先核对状态；不要自动重试",
                    }, ensure_ascii=False, separators=(",", ":"), sort_keys=True), completed_at=completed_at)
            elif table == "wake_attempts" and value["outcome"] == "delivery_unknown":
                value["outcome"] = "failed"
            elif table == "deliveries" and value["state"] == "uncertain":
                value["state"] = "failed"
            expected.append(value)
        temporary = table + "_failure_states"
        connection.execute(new_schema.replace(f"CREATE TABLE {table}", f"CREATE TABLE {temporary}", 1))
        for value in expected:
            connection.execute(f"INSERT INTO {temporary} ({','.join(value)}) VALUES ({','.join('?' for _ in value)})", tuple(value.values()))
        after = [dict(row) for row in connection.execute(f"SELECT * FROM {temporary} ORDER BY rowid")]
        if after != expected:
            raise RuntimeError(f"{table} 迁移字段对账失败")
        connection.execute(f"DROP TABLE {table}")
        connection.execute(new_schema)
        connection.execute(f"INSERT INTO {table} SELECT * FROM {temporary}")
        connection.execute(f"DROP TABLE {temporary}")
        if index is not None:
            connection.execute(index)
        if versions is not None:
            connection.execute(f"PRAGMA user_version = {versions[1]}")
        actual = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
        if _sql(actual) != _sql(new_schema):
            raise RuntimeError(f"{table} 迁移后 schema 不匹配")
        _check_integrity(connection)


def _check_integrity(connection: sqlite3.Connection) -> None:
    if [tuple(row) for row in connection.execute("PRAGMA integrity_check")] != [("ok",)]:
        raise RuntimeError("执行回执数据库完整性错误")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise RuntimeError("执行回执数据库外键错误")


def migrate_execution_failures(_ledger: object) -> None:
    """按正式配置定位各自数据库，先备份再进行可重入迁移。"""
    context = current_migration_context()
    config = tomllib.loads(context.config_path.read_text()) if context.config_path.is_file() else {}
    mobile = config.get("mobile_realtime", {})
    if not isinstance(mobile, dict):
        raise TypeError("mobile_realtime 配置不是 table")
    value = mobile.get("database", "data/mobile_realtime.db")
    if not isinstance(value, str) or not value:
        raise TypeError("mobile_realtime.database 必须为非空路径")
    configured = Path(value)
    path = configured if configured.is_absolute() else context.workspace / configured
    _migrate(path, "mobile_command_receipts", _MOBILE_OLD, _MOBILE_NEW,
             context.workspace / "backups/mobile-command-errors")
    _migrate(context.workspace / "model-registry.sqlite3", "model_calls", _MODEL_OLD, _MODEL_NEW,
             context.workspace / "backups/model-call-errors")
    _migrate(builtin_plugin_data_dir("wake", context.workspace) / "wake.sqlite3", "wake_attempts", _WAKE_OLD, _WAKE_NEW,
             context.workspace / "backups/wake-attempt-errors", versions=(8, 9))
    _migrate(context.workspace / "runtime/deliveries/settlements.sqlite", "deliveries", _LEDGER_OLD, _LEDGER_NEW,
             context.workspace / "backups/delivery-errors", versions=(1, 2), index=_LEDGER_INDEX)


steps = [step(migrate_execution_failures)]

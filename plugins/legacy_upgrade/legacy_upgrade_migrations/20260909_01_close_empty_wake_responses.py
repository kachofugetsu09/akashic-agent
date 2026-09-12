"""把旧 React 空响应留下的 Wake 阶段追加为可重试失败。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from .support.session_db_backup import backup_sqlite_database
from .support.wake_messages import finished
from .support.wake_request import Stage, WakeFailure, read_phase, read_request
from .support.legacy_message_log import MessageLog
from agent.plugin_contracts.message import Control

__depends__ = {"20260908_01_legacy_summaries"}
__transactional__ = False

_MIGRATION = "close-empty-wake-responses"
_ERROR = "模型没有产生内容或工具调用；空响应不是 quiet"
_DETAIL = f"ValueError: {_ERROR}"
_WAKE_DB = Path("plugin-data/wake-builtin/wake.sqlite3")
_MESSAGE_OBJECTS = {
    "attachments",
    "bindings",
    "idx_message_attachments_artifact",
    "ix_message_embeddings_hash",
    "message_attachments",
    "message_bindings",
    "message_call_result",
    "message_embeddings",
    "messages",
    "owner_records",
    "sessions",
}


class _Candidate(NamedTuple):
    flow_id: str
    session_id: str
    stage: Stage


def _failed_flows(path: Path) -> tuple[tuple[str, str], ...]:
    """读取 Wake 已确认的旧空响应 attempt，不推断其他失败。"""

    if not path.is_file():
        return ()
    with closing(sqlite3.connect(path)) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='wake_attempts'"
        ).fetchone()
        if table is None:
            raise RuntimeError("Wake 空响应迁移发现 wake_attempts 表缺失")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("Wake 空响应迁移发现 wake.sqlite3 损坏")
        columns = tuple(
            str(row[1]) for row in connection.execute("PRAGMA table_info(wake_attempts)")
        )
        if columns != (
            "attempt_id", "timer_id", "scheduled_for", "fired_at", "mail_watermark",
            "outcome", "owner", "detail", "completed_at",
        ):
            raise RuntimeError("Wake 空响应迁移发现 wake_attempts schema 不匹配")
        rows = connection.execute(
            "SELECT attempt_id, owner FROM wake_attempts "
            "WHERE outcome='delivery_unknown' AND detail=? ORDER BY attempt_id",
            (_DETAIL,),
        ).fetchall()
    return tuple((str(row[0]), str(row[1])) for row in rows)


def _check_sessions(path: Path) -> None:
    """只读确认 MessageLog 打开时不会补建缺失对象。"""

    uri = f"{path.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        objects = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
            )
        }
        missing = sorted(_MESSAGE_OBJECTS - objects)
        if missing:
            raise RuntimeError(
                "Wake 空响应迁移发现 sessions.db schema 不完整: " + ", ".join(missing)
            )
        if (
            connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]
            or connection.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise RuntimeError("Wake 空响应迁移发现 sessions.db 完整性检查失败")


def _candidates(workspace: Path) -> tuple[_Candidate, ...]:
    """只选择仍 pending、已有阶段 Input 且没有终态的精确旧 flow。"""

    sessions = workspace / "sessions.db"
    flow_ids = _failed_flows(workspace / _WAKE_DB)
    if not flow_ids:
        return ()
    if not sessions.is_file():
        raise RuntimeError("Wake 空响应迁移发现 sessions.db 缺失")
    _check_sessions(sessions)
    log = MessageLog(sessions)
    try:
        owner = log.owner("plugin:wake")
        selected: list[_Candidate] = []
        for flow_id, attempt_owner in flow_ids:
            row = owner.read("flow:" + flow_id)
            if row is None:
                raise RuntimeError(f"Wake flow {flow_id} 缺少恢复指针")
            pointer = dict(row.value)
            if set(pointer) != {"session_id", "input_id", "settled"}:
                raise RuntimeError(f"Wake flow {flow_id} pointer 结构不匹配")
            if pointer["settled"] is True:
                continue
            if pointer["settled"] is not False:
                raise RuntimeError(f"Wake flow {flow_id} settled 不是布尔值")
            reader = log.reader(str(pointer["session_id"]))
            request = read_request(reader.snapshot())
            if (
                request.flow_id,
                request.session_id,
                request.input_id,
                request.owner,
            ) != (
                flow_id,
                pointer["session_id"],
                pointer["input_id"],
                attempt_owner,
            ):
                raise RuntimeError(f"Wake flow {flow_id} pointer 与请求不一致")
            _, phase = read_phase(reader.snapshot(), request)
            if finished(reader, request, phase.stage) is None:
                selected.append(_Candidate(flow_id, request.session_id, phase.stage))
        return tuple(selected)
    finally:
        log.close()


def migrate(workspace: Path) -> int:
    """备份后只追加失败 Control；Source 在启动时继续领域结算。"""

    candidates = _candidates(workspace)
    if not candidates:
        return 0
    sessions = workspace / "sessions.db"
    backup_sqlite_database(
        sessions,
        workspace / "backups" / _MIGRATION / uuid4().hex,
        migration=_MIGRATION,
    )

    log = MessageLog(sessions)
    try:
        owner = log.owner("plugin:wake")
        reason = WakeFailure(message=_ERROR, retryable=True).model_dump_json()
        appended = 0
        for candidate in candidates:
            row = owner.read("flow:" + candidate.flow_id)
            if row is None or dict(row.value).get("settled") is not False:
                raise RuntimeError(f"Wake flow {candidate.flow_id} 在迁移提交前发生变化")
            reader = log.reader(candidate.session_id)
            request = read_request(reader.snapshot())
            _, phase = read_phase(reader.snapshot(), request)
            if (
                request.flow_id != candidate.flow_id
                or request.session_id != candidate.session_id
                or phase.stage != candidate.stage
            ):
                raise RuntimeError(f"Wake flow {candidate.flow_id} 在迁移提交前发生变化")
            if finished(reader, request, phase.stage) is not None:
                continue
            source_head = reader.head(source="wake")
            writer = log.writer(
                candidate.session_id,
                author="wake",
                source="wake",
                body_types=(Control,),
                content={},
            )
            try:
                terminal = writer.append(
                    request.phase_id(phase.stage) + ":failure",
                    Control("failure", source_head, reason),
                    expected_source_head=source_head,
                )
            finally:
                writer.expire()
            if finished(reader, request, phase.stage) != terminal:
                raise RuntimeError(f"Wake flow {candidate.flow_id} 失败终态未提交")
            appended += 1
    finally:
        log.close()
    with closing(sqlite3.connect(sessions)) as connection:
        if (
            connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]
            or connection.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise RuntimeError("Wake 空响应迁移后 sessions.db 完整性检查失败")
    return appended


def apply(_ledger: object) -> int:
    return migrate(current_migration_context().workspace)


steps = [step(apply)]

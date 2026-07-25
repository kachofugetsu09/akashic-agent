from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from agent.policies.tool_approval import canonical_args_hash, summarize_arguments
from agent.policies.tool_approval_decision import (
    ApprovalDecisionAction,
    ToolApprovalDecision,
)

ApprovalStatus = Literal[
    "pending",
    "approved",
    "denied",
    "expired",
    "consumed",
    "executed",
    "execution_failed",
]

_TERMINAL_STATUSES = frozenset(
    {"denied", "expired", "consumed", "executed", "execution_failed"}
)

_CREATE_APPROVAL_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tool_approval_requests (
    approval_request_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT '',
    chat_id TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL,
    risk TEXT NOT NULL,
    approval_scope TEXT NOT NULL,
    policy_reason TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    args_summary_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK (
        status IN (
            'pending', 'approved', 'denied', 'expired',
            'consumed', 'executed', 'execution_failed'
        )
    ),
    requested_by TEXT NOT NULL DEFAULT 'model',
    decided_by TEXT NOT NULL DEFAULT '',
    decision_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    decided_at TEXT NOT NULL DEFAULT '',
    consumed_at TEXT NOT NULL DEFAULT '',
    executed_at TEXT NOT NULL DEFAULT '',
    execution_status TEXT NOT NULL DEFAULT ''
)
"""


@dataclass(frozen=True)
class ToolApprovalRequestRecord:
    approval_request_id: str
    request_id: str
    session_key: str
    channel: str
    chat_id: str
    source: str
    tool_name: str
    risk: str
    approval_scope: str
    policy_reason: str
    args_hash: str
    args_summary: dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = "pending"
    requested_by: str = "model"
    decided_by: str = ""
    decision_reason: str = ""
    created_at: str = ""
    expires_at: str = ""
    decided_at: str = ""
    consumed_at: str = ""
    executed_at: str = ""
    execution_status: str = ""


class ToolApprovalStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def create_or_get_pending_request(
        self,
        *,
        request_id: str,
        session_key: str,
        channel: str,
        chat_id: str,
        source: str,
        tool_name: str,
        risk: str,
        approval_scope: str,
        policy_reason: str,
        arguments: Mapping[str, object],
        now: datetime,
        ttl: timedelta,
    ) -> ToolApprovalRequestRecord:
        self.expire_pending_requests(now=now)
        args_hash = canonical_args_hash(arguments)
        args_summary_json = _json_dumps(summarize_arguments(arguments))
        created_at = _to_iso(now)
        expires_at = _to_iso(now + ttl)
        approval_request_id = uuid.uuid4().hex
        with self._immediate_transaction() as conn:
            row = _select_pending_by_binding(
                conn,
                session_key=session_key,
                request_id=request_id,
                tool_name=tool_name,
                approval_scope=approval_scope or "tool_call",
                args_hash=args_hash,
            )
            if row is not None:
                return _record_from_row(row)
            try:
                conn.execute(
                    """
                    INSERT INTO tool_approval_requests (
                        approval_request_id, request_id, session_key, channel,
                        chat_id, source, tool_name, risk, approval_scope,
                        policy_reason, args_hash, args_summary_json, status,
                        requested_by, created_at, expires_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending',
                            'model', ?, ?)
                    """,
                    (
                        approval_request_id,
                        request_id,
                        session_key,
                        channel,
                        chat_id,
                        source,
                        tool_name,
                        risk,
                        approval_scope or "tool_call",
                        policy_reason,
                        args_hash,
                        args_summary_json,
                        created_at,
                        expires_at,
                    ),
                )
            except sqlite3.IntegrityError:
                row = _select_pending_by_binding(
                    conn,
                    session_key=session_key,
                    request_id=request_id,
                    tool_name=tool_name,
                    approval_scope=approval_scope or "tool_call",
                    args_hash=args_hash,
                )
                if row is not None:
                    return _record_from_row(row)
                raise
            row = _select_for_update(conn, approval_request_id)
        if row is None:
            raise RuntimeError("failed to create or load approval request")
        return _record_from_row(row)

    def get_request(
        self, approval_request_id: str
    ) -> ToolApprovalRequestRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tool_approval_requests WHERE approval_request_id = ?",
                (approval_request_id,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def list_pending_requests(
        self, *, session_key: str, now: datetime
    ) -> list[ToolApprovalRequestRecord]:
        self.expire_pending_requests(now=now)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tool_approval_requests
                WHERE session_key = ? AND status = 'pending'
                ORDER BY created_at ASC, approval_request_id ASC
                """,
                (session_key,),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def expire_pending_requests(self, *, now: datetime) -> list[ToolApprovalDecision]:
        now_iso = _to_iso(now)
        with self._immediate_transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tool_approval_requests
                WHERE status = 'pending' AND expires_at <= ?
                ORDER BY created_at ASC, approval_request_id ASC
                """,
                (now_iso,),
            ).fetchall()
            conn.executemany(
                """
                UPDATE tool_approval_requests
                SET status = 'expired',
                    decision_reason = 'approval_expired',
                    decided_at = ?
                WHERE approval_request_id = ? AND status = 'pending'
                """,
                [(now_iso, row["approval_request_id"]) for row in rows],
            )
        return [
            _decision_from_record(
                _record_from_row(row),
                action="expired",
                reason="approval_expired",
                decided_at=now_iso,
            )
            for row in rows
        ]

    def approve_request(
        self,
        *,
        approval_request_id: str,
        request_id: str,
        session_key: str,
        tool_name: str,
        approval_scope: str,
        args_hash: str,
        actor: str,
        now: datetime,
    ) -> ToolApprovalDecision:
        return self._decide_request(
            approval_request_id=approval_request_id,
            request_id=request_id,
            session_key=session_key,
            tool_name=tool_name,
            approval_scope=approval_scope,
            args_hash=args_hash,
            actor=actor,
            now=now,
            target_status="approved",
            reason="approval_approved",
        )

    def deny_request(
        self,
        *,
        approval_request_id: str,
        request_id: str,
        session_key: str,
        tool_name: str,
        approval_scope: str,
        args_hash: str,
        actor: str,
        reason: str,
        now: datetime,
    ) -> ToolApprovalDecision:
        return self._decide_request(
            approval_request_id=approval_request_id,
            request_id=request_id,
            session_key=session_key,
            tool_name=tool_name,
            approval_scope=approval_scope,
            args_hash=args_hash,
            actor=actor,
            now=now,
            target_status="denied",
            reason=reason or "approval_denied",
        )

    def consume_approved_request(
        self,
        *,
        approval_request_id: str,
        request_id: str,
        session_key: str,
        tool_name: str,
        approval_scope: str,
        args_hash: str,
        actor: str,
        now: datetime,
    ) -> ToolApprovalDecision:
        now_iso = _to_iso(now)
        with self._immediate_transaction() as conn:
            row = _select_for_update(conn, approval_request_id)
            if row is None:
                return ToolApprovalDecision(
                    action="not_found",
                    reason="approval_request_not_found",
                    approval_request_id=approval_request_id,
                    request_id=request_id,
                    session_key=session_key,
                    tool_name=tool_name,
                    approval_scope=approval_scope,
                    args_hash=args_hash,
                )
            record = _record_from_row(row)
            mismatch = _binding_mismatch(
                record, request_id, session_key, tool_name, approval_scope, args_hash
            )
            if mismatch:
                return _decision_from_record(
                    record, action="mismatch", reason="approval_binding_mismatch"
                )
            if record.status == "approved" and record.expires_at <= now_iso:
                conn.execute(
                    """
                    UPDATE tool_approval_requests
                    SET status = 'expired',
                        decision_reason = 'approval_expired',
                        decided_at = ?
                    WHERE approval_request_id = ? AND status = 'approved'
                    """,
                    (now_iso, approval_request_id),
                )
                return _decision_from_record(
                    record,
                    action="expired",
                    reason="approval_expired",
                    decided_at=now_iso,
                )
            if record.status == "approved":
                conn.execute(
                    """
                    UPDATE tool_approval_requests
                    SET status = 'consumed', consumed_at = ?
                    WHERE approval_request_id = ? AND status = 'approved'
                    """,
                    (now_iso, approval_request_id),
                )
                return _decision_from_record(
                    record,
                    action="consumed",
                    reason="approval_consumed",
                    actor=actor,
                    consumed_at=now_iso,
                )
            if record.status in _TERMINAL_STATUSES:
                return _decision_from_record(
                    record,
                    action=record.status if record.status in {"denied", "expired"} else "mismatch",
                    reason=f"approval_already_{record.status}",
                )
            return _decision_from_record(
                record, action=record.status, reason=f"approval_status_{record.status}"
            )

    def finalize_consumed_request(
        self,
        *,
        approval_request_id: str,
        request_id: str,
        session_key: str,
        tool_name: str,
        approval_scope: str,
        args_hash: str,
        execution_status: str,
        now: datetime,
    ) -> ToolApprovalDecision:
        if execution_status not in {"executed", "execution_failed"}:
            raise ValueError("execution_status must be executed or execution_failed")
        now_iso = _to_iso(now)
        with self._immediate_transaction() as conn:
            row = _select_for_update(conn, approval_request_id)
            if row is None:
                return ToolApprovalDecision(
                    action="not_found",
                    reason="approval_request_not_found",
                    approval_request_id=approval_request_id,
                    request_id=request_id,
                    session_key=session_key,
                    tool_name=tool_name,
                    approval_scope=approval_scope,
                    args_hash=args_hash,
                )
            record = _record_from_row(row)
            mismatch = _binding_mismatch(
                record, request_id, session_key, tool_name, approval_scope, args_hash
            )
            if mismatch:
                return _decision_from_record(
                    record, action="mismatch", reason="approval_binding_mismatch"
                )
            if record.status != "consumed":
                return _decision_from_record(
                    record,
                    action=record.status,
                    reason=f"approval_status_{record.status}",
                )
            conn.execute(
                """
                UPDATE tool_approval_requests
                SET status = ?, executed_at = ?, execution_status = ?
                WHERE approval_request_id = ? AND status = 'consumed'
                """,
                (execution_status, now_iso, execution_status, approval_request_id),
            )
        return _decision_from_record(
            record,
            action=execution_status,
            reason=f"approval_{execution_status}",
            executed_at=now_iso,
        )

    def _decide_request(
        self,
        *,
        approval_request_id: str,
        request_id: str,
        session_key: str,
        tool_name: str,
        approval_scope: str,
        args_hash: str,
        actor: str,
        now: datetime,
        target_status: Literal["approved", "denied"],
        reason: str,
    ) -> ToolApprovalDecision:
        now_iso = _to_iso(now)
        with self._immediate_transaction() as conn:
            row = _select_for_update(conn, approval_request_id)
            if row is None:
                return ToolApprovalDecision(
                    action="not_found",
                    reason="approval_request_not_found",
                    approval_request_id=approval_request_id,
                    request_id=request_id,
                    session_key=session_key,
                    tool_name=tool_name,
                    approval_scope=approval_scope,
                    args_hash=args_hash,
                )
            record = _record_from_row(row)
            mismatch = _binding_mismatch(
                record, request_id, session_key, tool_name, approval_scope, args_hash
            )
            if mismatch:
                return _decision_from_record(
                    record, action="mismatch", reason="approval_binding_mismatch"
                )
            if record.status == "pending" and record.expires_at <= now_iso:
                conn.execute(
                    """
                    UPDATE tool_approval_requests
                    SET status = 'expired',
                        decision_reason = 'approval_expired',
                        decided_at = ?
                    WHERE approval_request_id = ? AND status = 'pending'
                    """,
                    (now_iso, approval_request_id),
                )
                return _decision_from_record(
                    record,
                    action="expired",
                    reason="approval_expired",
                    decided_at=now_iso,
                )
            if record.status != "pending":
                return _decision_from_record(
                    record,
                    action=record.status,
                    reason=f"approval_status_{record.status}",
                )
            conn.execute(
                """
                UPDATE tool_approval_requests
                SET status = ?,
                    decided_by = ?,
                    decision_reason = ?,
                    decided_at = ?
                WHERE approval_request_id = ? AND status = 'pending'
                """,
                (target_status, actor, reason, now_iso, approval_request_id),
            )
        return _decision_from_record(
            record,
            action=target_status,
            reason=reason,
            actor=actor,
            decided_at=now_iso,
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _immediate_transaction(self) -> _ImmediateTransaction:
        return _ImmediateTransaction(self._connect())

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_APPROVAL_TABLE_SQL)
            _migrate_legacy_binding_unique(conn)
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_approval_pending_binding
                ON tool_approval_requests (
                    session_key, request_id, tool_name, approval_scope, args_hash
                )
                WHERE status = 'pending'
                """
            )


class _ImmediateTransaction:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()


def _select_for_update(
    conn: sqlite3.Connection, approval_request_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM tool_approval_requests WHERE approval_request_id = ?",
        (approval_request_id,),
    ).fetchone()


def _select_pending_by_binding(
    conn: sqlite3.Connection,
    *,
    session_key: str,
    request_id: str,
    tool_name: str,
    approval_scope: str,
    args_hash: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM tool_approval_requests
        WHERE session_key = ?
          AND request_id = ?
          AND tool_name = ?
          AND approval_scope = ?
          AND args_hash = ?
          AND status = 'pending'
        ORDER BY created_at ASC, approval_request_id ASC
        LIMIT 1
        """,
        (session_key, request_id, tool_name, approval_scope, args_hash),
    ).fetchone()


def _migrate_legacy_binding_unique(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type = 'table' AND name = 'tool_approval_requests'
        """
    ).fetchone()
    if row is None:
        return
    sql = str(row[0] or "")
    if "UNIQUE(" not in sql or "approval_scope, args_hash" not in sql:
        return
    conn.executescript(
        """
        ALTER TABLE tool_approval_requests RENAME TO tool_approval_requests_legacy;
        """
    )
    conn.execute(_CREATE_APPROVAL_TABLE_SQL)
    conn.execute(
        """
        INSERT INTO tool_approval_requests (
            approval_request_id, request_id, session_key, channel, chat_id,
            source, tool_name, risk, approval_scope, policy_reason, args_hash,
            args_summary_json, status, requested_by, decided_by, decision_reason,
            created_at, expires_at, decided_at, consumed_at, executed_at,
            execution_status
        )
        SELECT
            approval_request_id, request_id, session_key, channel, chat_id,
            source, tool_name, risk, approval_scope, policy_reason, args_hash,
            args_summary_json, status, requested_by, decided_by, decision_reason,
            created_at, expires_at, decided_at, consumed_at, executed_at,
            execution_status
        FROM tool_approval_requests_legacy
        """
    )
    conn.execute("DROP TABLE tool_approval_requests_legacy")


def _record_from_row(row: sqlite3.Row) -> ToolApprovalRequestRecord:
    args_summary_raw = row["args_summary_json"] or "{}"
    try:
        args_summary = json.loads(args_summary_raw)
    except json.JSONDecodeError:
        args_summary = {}
    return ToolApprovalRequestRecord(
        approval_request_id=str(row["approval_request_id"]),
        request_id=str(row["request_id"]),
        session_key=str(row["session_key"]),
        channel=str(row["channel"]),
        chat_id=str(row["chat_id"]),
        source=str(row["source"]),
        tool_name=str(row["tool_name"]),
        risk=str(row["risk"]),
        approval_scope=str(row["approval_scope"] or "tool_call"),
        policy_reason=str(row["policy_reason"]),
        args_hash=str(row["args_hash"]),
        args_summary=args_summary if isinstance(args_summary, dict) else {},
        status=row["status"],
        requested_by=str(row["requested_by"]),
        decided_by=str(row["decided_by"]),
        decision_reason=str(row["decision_reason"]),
        created_at=str(row["created_at"]),
        expires_at=str(row["expires_at"]),
        decided_at=str(row["decided_at"]),
        consumed_at=str(row["consumed_at"]),
        executed_at=str(row["executed_at"]),
        execution_status=str(row["execution_status"]),
    )


def _decision_from_record(
    record: ToolApprovalRequestRecord,
    *,
    action: str | None = None,
    reason: str,
    actor: str = "",
    decided_at: str = "",
    consumed_at: str = "",
    executed_at: str = "",
) -> ToolApprovalDecision:
    return ToolApprovalDecision(
        action=cast(ApprovalDecisionAction, action or record.status),
        reason=reason,
        approval_request_id=record.approval_request_id,
        request_id=record.request_id,
        session_key=record.session_key,
        tool_name=record.tool_name,
        approval_scope=record.approval_scope,
        args_hash=record.args_hash,
    )


def _binding_mismatch(
    record: ToolApprovalRequestRecord,
    request_id: str,
    session_key: str,
    tool_name: str,
    approval_scope: str,
    args_hash: str,
) -> bool:
    return (
        record.request_id != request_id
        or record.session_key != session_key
        or record.tool_name != tool_name
        or record.approval_scope != (approval_scope or "tool_call")
        or record.args_hash != args_hash
    )


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()

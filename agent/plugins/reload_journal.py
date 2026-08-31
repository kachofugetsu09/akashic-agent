from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

ReloadPhase = Literal[
    "preparing",
    "prepared",
    "validating",
    "commit_started",
    "latest_ready",
    "discarding",
    "promoting",
    "committed",
    "draining",
    "cleanup_failed",
    "degraded",
    "complete",
    "aborted",
    "recovered",
]
RecoveryActionName = Literal[
    "discard_candidate",
    "restore_candidate",
    "restore_committed",
    "retry_generation_cleanup",
    "retry_runtime_recovery",
]
RecoveryTarget = Literal["base", "candidate"]
_TERMINAL_PHASES = frozenset({"complete", "aborted", "recovered"})
_FAILURE_PHASES = frozenset({"cleanup_failed", "degraded"})
_TRANSITIONS: dict[str, frozenset[str]] = {
    "preparing": frozenset({"prepared", "aborted", "cleanup_failed", "degraded"}),
    "prepared": frozenset({"validating", "aborted", "cleanup_failed", "degraded"}),
    "validating": frozenset({"commit_started", "aborted", "cleanup_failed", "degraded"}),
    "commit_started": frozenset(
        {"latest_ready", "committed", "aborted", "recovered", "cleanup_failed", "degraded"}
    ),
    "latest_ready": frozenset(
        {"discarding", "promoting", "aborted", "recovered", "cleanup_failed", "degraded"}
    ),
    "discarding": frozenset({"aborted", "cleanup_failed", "degraded"}),
    "promoting": frozenset(
        {"discarding", "committed", "aborted", "recovered", "cleanup_failed", "degraded"}
    ),
    "committed": frozenset({"draining", "complete", "recovered", "cleanup_failed", "degraded"}),
    "draining": frozenset({"complete", "recovered", "cleanup_failed", "degraded"}),
    "cleanup_failed": frozenset({"cleanup_failed", "degraded"}),
    "degraded": frozenset({"degraded"}),
}


@dataclass(frozen=True)
class ReloadTransactionRecord:
    tx_id: str
    plugin_id: str
    base_snapshot_id: str | None
    candidate_snapshot_id: str | None
    generation_id: str
    source_revision: str
    config_revision: str
    phase: ReloadPhase
    started_at: str
    updated_at: str
    error: str
    base_generation_id: str | None = None
    formal_effects: tuple[str, ...] = ()
    failure_resource: str | None = None
    recovery_action: RecoveryActionName | None = None
    attempt_count: int = 0
    runtime_owner_boot_id: str | None = None
    base_artifact_pointer: str | None = None
    candidate_artifact_pointer: str | None = None
    recovery_target: RecoveryTarget | None = None

    @property
    def old_snapshot_id(self) -> str | None:
        """Return the stable snapshot that the attempt started from."""

        return self.base_snapshot_id

    @property
    def new_snapshot_id(self) -> str | None:
        """Return the candidate snapshot produced by the attempt."""

        return self.candidate_snapshot_id

    @property
    def old_generation_id(self) -> str | None:
        """Return the stable generation that the attempt started from."""

        return self.base_generation_id

    @property
    def attempt_generation_id(self) -> str:
        """Return the generation being prepared by this attempt."""

        return self.generation_id

    @property
    def resource(self) -> str | None:
        """Return the retained failed resource owner, if any."""

        return self.failure_resource

    @property
    def attempt(self) -> int:
        """Return the durable retry attempt count."""

        return self.attempt_count


@dataclass(frozen=True)
class ReloadJournalEvent:
    sequence: int
    phase: ReloadPhase
    details: dict[str, object]
    created_at: str


@dataclass(frozen=True)
class ReloadRecoveryAction:
    tx_id: str
    plugin_id: str
    generation_id: str
    source_revision: str
    phase: ReloadPhase
    action: RecoveryActionName
    base_snapshot_id: str | None = None
    candidate_snapshot_id: str | None = None
    base_generation_id: str | None = None
    formal_effects: tuple[str, ...] = ()
    failure_resource: str | None = None
    error: str = ""
    attempt_count: int = 0
    runtime_owner_boot_id: str | None = None
    base_artifact_pointer: str | None = None
    candidate_artifact_pointer: str | None = None
    recovery_target: RecoveryTarget | None = None

class ReloadJournal:
    """Persist plugin reload phases and expose deterministic crash recovery work."""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace / "runtime" / "plugin-reloads.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def begin(
        self,
        *,
        plugin_id: str,
        base_snapshot_id: str | None,
        base_generation_id: str | None = None,
        generation_id: str,
        source_revision: str,
        config_revision: str,
        base_artifact_pointer: str | None = None,
        candidate_artifact_pointer: str | None = None,
    ) -> str:
        now = _now()
        tx_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reload_transactions (
                    tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
                    base_generation_id, generation_id, source_revision,
                    config_revision, phase, started_at, updated_at, error,
                    formal_effects_json, attempt_count, base_artifact_pointer,
                    candidate_artifact_pointer
                ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, 'preparing', ?, ?, '', '[]', 0, ?, ?)
                """,
                (
                    tx_id,
                    plugin_id,
                    base_snapshot_id,
                    base_generation_id,
                    generation_id,
                    source_revision,
                    config_revision,
                    now,
                    now,
                    base_artifact_pointer,
                    candidate_artifact_pointer,
                ),
            )
            self._append_event(conn, tx_id, "preparing", {}, now)
        return tx_id

    def mark_runtime_owner(self, tx_id: str, boot_id: str) -> None:
        """Persist the boot owner before starting any candidate or formal runtime."""

        if not boot_id.strip():
            raise ValueError("ReloadTransaction runtime boot id 不能为空")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT runtime_owner_boot_id, phase FROM reload_transactions WHERE tx_id = ?",
                (tx_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
            existing = _optional_string(row[0])
            if existing not in {None, boot_id}:
                raise RuntimeError(
                    "ReloadTransaction runtime boot owner 不可覆盖: "
                    f"{existing} -> {boot_id}"
                )
            now = _now()
            conn.execute(
                """
                UPDATE reload_transactions
                SET runtime_owner_boot_id = COALESCE(runtime_owner_boot_id, ?),
                    updated_at = ?
                WHERE tx_id = ?
                """,
                (boot_id, now, tx_id),
            )
            self._append_event(
                conn,
                tx_id,
                cast(ReloadPhase, str(row[1])),
                {"runtime_owner_boot_id": boot_id},
                now,
            )

    def advance(
        self,
        tx_id: str,
        phase: ReloadPhase,
        *,
        candidate_snapshot_id: str | None = None,
        details: dict[str, object] | None = None,
        error: str | None = None,
        resource: str | None = None,
        formal_effects: tuple[str, ...] | None = None,
        recovery_action: RecoveryActionName | None = None,
        attempt_count: int | None = None,
        recovery_target: RecoveryTarget | None = None,
    ) -> None:
        """Advance one reload transaction and append its durable evidence."""

        with self._connect() as conn:
            # 1. 读取当前状态并验证单向 phase contract。
            row = conn.execute(
                """
                SELECT phase, base_snapshot_id, candidate_snapshot_id,
                       base_generation_id, generation_id, formal_effects_json,
                       failure_resource, recovery_action, attempt_count, error,
                       runtime_owner_boot_id, base_artifact_pointer,
                       candidate_artifact_pointer, recovery_target
                FROM reload_transactions
                WHERE tx_id = ?
                """,
                (tx_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
            current = str(row[0])
            if phase not in _TRANSITIONS.get(current, frozenset()):
                raise RuntimeError(
                    f"ReloadTransaction 状态跳转无效: {current} -> {phase}"
                )
            # 2. 规范化 snapshot、resource、formal effect 与 recovery evidence。
            details_for_event = dict(details or {})
            current_candidate = _optional_string(row[2])
            current_base_generation = _optional_string(row[3])
            current_effects = _decode_effects(row[5])
            current_resource = _optional_string(row[6])
            current_action = _optional_action(row[7])
            current_attempt_count = int(row[8])
            next_candidate = candidate_snapshot_id
            if next_candidate is None:
                next_candidate = _detail_string(
                    details_for_event,
                    "new_snapshot_id",
                    "candidate_snapshot_id",
                )
            if next_candidate is None:
                next_candidate = current_candidate
            next_base_generation = current_base_generation
            detail_base_generation = _detail_string(
                details_for_event,
                "old_generation_id",
                "base_generation_id",
                "old_generation",
            )
            if detail_base_generation is not None:
                next_base_generation = detail_base_generation
            next_resource = resource
            if next_resource is None:
                next_resource = _detail_string(
                    details_for_event,
                    "resource",
                    "failure_resource",
                )
            if next_resource is None:
                next_resource = current_resource
            elif current in _FAILURE_PHASES and current_resource is not None:
                next_resource = _merge_resources(current_resource, next_resource)
            next_effects = current_effects
            if formal_effects is not None:
                next_effects = _merge_effects(
                    current_effects,
                    _validate_effects(formal_effects),
                )
            elif "formal_effects" in details_for_event:
                next_effects = _merge_effects(
                    current_effects,
                    _validate_effects(details_for_event["formal_effects"]),
                )
            next_action = recovery_action
            if next_action is None:
                next_action = _optional_action(details_for_event.get("recovery_action"))
            expected_action = _recovery_action(phase)
            if next_action is None and phase in _FAILURE_PHASES:
                next_action = expected_action
            elif next_action is None and current in _FAILURE_PHASES:
                next_action = current_action
            if next_action is None:
                next_action = expected_action
            if expected_action is not None:
                if next_action is None:
                    next_action = expected_action
                elif next_action != expected_action:
                    raise RuntimeError(
                        f"ReloadTransaction 恢复 action 与状态不一致: {phase} -> {next_action}"
                    )
            next_attempt_count = current_attempt_count
            if attempt_count is None and "attempt" in details_for_event:
                raw_attempt = details_for_event["attempt"]
                if isinstance(raw_attempt, int) and not isinstance(raw_attempt, bool):
                    attempt_count = raw_attempt
            if phase in _FAILURE_PHASES:
                if attempt_count is None:
                    next_attempt_count += 1
                else:
                    if attempt_count < current_attempt_count:
                        raise ValueError("ReloadTransaction attempt_count 不能减少")
                    next_attempt_count = attempt_count
            elif attempt_count is not None:
                if attempt_count < current_attempt_count:
                    raise ValueError("ReloadTransaction attempt_count 不能减少")
                next_attempt_count = attempt_count
            next_error = error
            if next_error is None:
                next_error = _detail_string(details_for_event, "error")
            if next_error is None:
                next_error = str(row[9])
            if phase in _FAILURE_PHASES:
                stored_target = _optional_recovery_target(row[13])
                if (
                    stored_target is not None
                    and recovery_target is not None
                    and recovery_target != stored_target
                ):
                    raise RuntimeError(
                        "ReloadTransaction recovery target 不可覆盖: "
                        f"{stored_target} -> {recovery_target}"
                    )
                next_recovery_target = (
                    recovery_target
                    if recovery_target is not None
                    else stored_target
                )
                if next_recovery_target is None:
                    raise ValueError(
                        "ReloadTransaction failure phase 必须保存 recovery target"
                    )
                if not next_resource:
                    raise ValueError(
                        "ReloadTransaction failure phase 必须保存 resource identity"
                    )
                if not next_error:
                    raise ValueError(
                        "ReloadTransaction failure phase 必须保存 error evidence"
                    )
                _add_failure_evidence(
                    details_for_event,
                    base_snapshot_id=_optional_string(row[1]),
                    candidate_snapshot_id=next_candidate,
                    base_generation_id=next_base_generation,
                    generation_id=str(row[4]),
                    formal_effects=next_effects,
                    resource=next_resource,
                    error=next_error,
                    action=next_action,
                    attempt_count=next_attempt_count,
                    runtime_owner_boot_id=_optional_string(row[10]),
                    base_artifact_pointer=_optional_string(row[11]),
                    candidate_artifact_pointer=_optional_string(row[12]),
                    recovery_target=next_recovery_target,
                )
            else:
                next_recovery_target = _optional_recovery_target(row[13])
            # 3. 在同一 SQLite transaction 内更新状态并追加事件。
            now = _now()
            conn.execute(
                """
                UPDATE reload_transactions
                SET phase = ?,
                    candidate_snapshot_id = COALESCE(?, candidate_snapshot_id),
                    base_generation_id = COALESCE(?, base_generation_id),
                    updated_at = ?,
                    error = ?,
                    formal_effects_json = ?,
                    failure_resource = ?,
                    recovery_action = ?,
                    attempt_count = ?,
                    recovery_target = COALESCE(?, recovery_target)
                WHERE tx_id = ?
                """,
                (
                    phase,
                    next_candidate,
                    next_base_generation,
                    now,
                    next_error,
                    json.dumps(next_effects, ensure_ascii=False),
                    next_resource,
                    next_action,
                    next_attempt_count,
                    next_recovery_target,
                    tx_id,
                ),
            )
            self._append_event(conn, tx_id, phase, details_for_event, now)

    def get(self, tx_id: str) -> ReloadTransactionRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
                       base_generation_id, generation_id, source_revision,
                       config_revision, phase, started_at, updated_at, error,
                       formal_effects_json, failure_resource, recovery_action,
                       attempt_count, runtime_owner_boot_id,
                       base_artifact_pointer, candidate_artifact_pointer,
                       recovery_target
                FROM reload_transactions
                WHERE tx_id = ?
                """,
                (tx_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
        return _record(row)

    def latest(
        self,
        *,
        plugin_id: str | None = None,
    ) -> ReloadTransactionRecord | None:
        """返回指定插件最后发生状态变化的 reload transaction。"""
        where = "" if plugin_id is None else "WHERE plugin_id = ?"
        values: tuple[object, ...] = () if plugin_id is None else (plugin_id,)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
                       base_generation_id, generation_id, source_revision,
                       config_revision, phase, started_at, updated_at, error,
                       formal_effects_json, failure_resource, recovery_action,
                       attempt_count, runtime_owner_boot_id,
                       base_artifact_pointer, candidate_artifact_pointer,
                       recovery_target
                FROM reload_transactions
                {where}
                ORDER BY updated_at DESC, rowid DESC
                LIMIT 1
                """,
                values,
            ).fetchone()
        return None if row is None else _record(row)

    def events(self, tx_id: str) -> tuple[ReloadJournalEvent, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT sequence, phase, details_json, created_at
                FROM reload_events
                WHERE tx_id = ?
                ORDER BY sequence
                """,
                (tx_id,),
            ).fetchall()
        return tuple(
            ReloadJournalEvent(
                sequence=int(row[0]),
                phase=cast(ReloadPhase, str(row[1])),
                details=cast(dict[str, object], json.loads(str(row[2]))),
                created_at=str(row[3]),
            )
            for row in rows
        )

    def annotate(self, tx_id: str, details: dict[str, object]) -> None:
        """Append evidence without inventing another public rollout phase."""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT phase, base_snapshot_id, candidate_snapshot_id,
                       base_generation_id, generation_id, formal_effects_json,
                       failure_resource, recovery_action, attempt_count, error
                FROM reload_transactions
                WHERE tx_id = ?
                """,
                (tx_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
            now = _now()
            phase = cast(ReloadPhase, str(row[0]))
            details_for_event = dict(details)
            candidate_snapshot_id = _detail_string(
                details_for_event,
                "new_snapshot_id",
                "candidate_snapshot_id",
            )
            base_generation_id = _detail_string(
                details_for_event,
                "old_generation_id",
                "base_generation_id",
                "old_generation",
            )
            resource = _detail_string(
                details_for_event,
                "resource",
                "failure_resource",
            )
            effects = _decode_effects(row[5])
            if "formal_effects" in details_for_event:
                effects = _merge_effects(
                    effects,
                    _validate_effects(details_for_event["formal_effects"]),
                )
            action = _optional_action(details_for_event.get("recovery_action"))
            expected_action = _recovery_action(phase)
            if expected_action is not None and action not in {None, expected_action}:
                raise RuntimeError(
                    f"ReloadTransaction 恢复 action 与状态不一致: {phase} -> {action}"
                )
            if action is None:
                action = expected_action or _optional_action(row[7])
            attempt_count = int(row[8])
            if "attempt_count" in details_for_event:
                attempt_count = _validate_attempt_count(
                    details_for_event["attempt_count"],
                    minimum=attempt_count,
                )
            error = _detail_string(details_for_event, "error")
            conn.execute(
                """
                UPDATE reload_transactions
                SET candidate_snapshot_id = COALESCE(?, candidate_snapshot_id),
                    base_generation_id = COALESCE(?, base_generation_id),
                    updated_at = ?,
                    error = COALESCE(?, error),
                    formal_effects_json = ?,
                    failure_resource = COALESCE(?, failure_resource),
                    recovery_action = COALESCE(?, recovery_action),
                    attempt_count = ?
                WHERE tx_id = ?
                """,
                (
                    candidate_snapshot_id,
                    base_generation_id,
                    now,
                    error,
                    json.dumps(effects, ensure_ascii=False),
                    resource,
                    action,
                    attempt_count,
                    tx_id,
                ),
            )
            self._append_event(conn, tx_id, phase, details_for_event, now)

    def pending_recovery(self) -> tuple[ReloadRecoveryAction, ...]:
        placeholders = ", ".join("?" for _ in _TERMINAL_PHASES)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
                       base_generation_id, generation_id, source_revision, phase,
                       formal_effects_json, failure_resource, recovery_action,
                       error, attempt_count, runtime_owner_boot_id,
                       base_artifact_pointer, candidate_artifact_pointer,
                       recovery_target
                FROM reload_transactions
                WHERE phase NOT IN ({placeholders})
                ORDER BY started_at, tx_id
                """,
                tuple(sorted(_TERMINAL_PHASES)),
            ).fetchall()
        actions: list[ReloadRecoveryAction] = []
        for row in rows:
            phase = cast(ReloadPhase, str(row[7]))
            action = _recovery_action(phase, _optional_action(row[10]))
            if action is None:
                raise RuntimeError(f"ReloadTransaction 无法恢复状态: {phase}")
            actions.append(
                ReloadRecoveryAction(
                    tx_id=str(row[0]),
                    plugin_id=str(row[1]),
                    generation_id=str(row[5]),
                    source_revision=str(row[6]),
                    phase=phase,
                    action=action,
                    base_snapshot_id=_optional_string(row[2]),
                    candidate_snapshot_id=_optional_string(row[3]),
                    base_generation_id=_optional_string(row[4]),
                    formal_effects=_decode_effects(row[8]),
                    failure_resource=_optional_string(row[9]),
                    error=str(row[11]),
                    attempt_count=int(row[12]),
                    runtime_owner_boot_id=_optional_string(row[13]),
                    base_artifact_pointer=_optional_string(row[14]),
                    candidate_artifact_pointer=_optional_string(row[15]),
                    recovery_target=_optional_recovery_target(row[16]),
                )
            )
        return tuple(
            sorted(
                actions,
                key=lambda item: (
                    _RECOVERY_ACTION_ORDER[item.action],
                    item.tx_id,
                ),
            )
        )

    def finish_recovery(
        self,
        action: ReloadRecoveryAction,
        *,
        retry_receipt: str | None = None,
    ) -> None:
        """Finish one recovery only after its owner supplied required evidence."""

        current = self.get(action.tx_id)
        expected_action = _recovery_action(current.phase, current.recovery_action)
        pointer_reset_discard = (
            action.action == "discard_candidate"
            and expected_action == "restore_committed"
            and current.phase in {"commit_started", "promoting"}
        )
        if current.phase != action.phase or (
            expected_action != action.action and not pointer_reset_discard
        ) or action.generation_id != current.generation_id or (
            action.source_revision != current.source_revision
        ) or action.attempt_count != current.attempt_count:
            raise RuntimeError(
                "ReloadTransaction recovery action 已失效: "
                f"phase={current.phase}, action={expected_action}"
            )
        if action.action in {
            "retry_generation_cleanup",
            "retry_runtime_recovery",
        }:
            if not retry_receipt:
                raise RuntimeError(
                    f"ReloadTransaction {action.action} 缺少 Host retry receipt"
                )
            self._finish_host_retry(action, retry_receipt=retry_receipt)
            return
        phase: ReloadPhase = (
            "aborted" if action.action == "discard_candidate" else "recovered"
        )
        self.advance(
            action.tx_id,
            phase,
            details={
                "recovery_action": action.action,
                "attempt_count": action.attempt_count,
            },
            error=None if current.phase in _FAILURE_PHASES else "startup recovery",
            recovery_action=expected_action,
        )

    def _finish_host_retry(
        self,
        action: ReloadRecoveryAction,
        *,
        retry_receipt: str,
    ) -> None:
        """Atomically persist a successful Host retry without reopening advance()."""

        terminal: ReloadPhase = (
            "aborted"
            if action.action == "retry_generation_cleanup"
            and action.recovery_target == "base"
            else "recovered"
        )
        details: dict[str, object] = {"retry_receipt": retry_receipt}
        _add_failure_evidence(
            details,
            base_snapshot_id=action.base_snapshot_id,
            candidate_snapshot_id=action.candidate_snapshot_id,
            base_generation_id=action.base_generation_id,
            generation_id=action.generation_id,
            formal_effects=action.formal_effects,
            resource=action.failure_resource,
            error=action.error,
            action=action.action,
            attempt_count=action.attempt_count,
            runtime_owner_boot_id=action.runtime_owner_boot_id,
            base_artifact_pointer=action.base_artifact_pointer,
            candidate_artifact_pointer=action.candidate_artifact_pointer,
            recovery_target=action.recovery_target,
        )
        now = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE reload_transactions
                SET phase = ?, updated_at = ?
                WHERE tx_id = ? AND phase = ? AND recovery_action = ?
                      AND attempt_count = ?
                """,
                (
                    terminal,
                    now,
                    action.tx_id,
                    action.phase,
                    action.action,
                    action.attempt_count,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "ReloadTransaction Host retry receipt 已失效: "
                    f"{action.tx_id}"
                )
            self._append_event(conn, action.tx_id, terminal, details, now)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS reload_transactions (
                    tx_id TEXT PRIMARY KEY,
                    plugin_id TEXT NOT NULL,
                    base_snapshot_id TEXT,
                    candidate_snapshot_id TEXT,
                    base_generation_id TEXT,
                    generation_id TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    config_revision TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT NOT NULL,
                    formal_effects_json TEXT NOT NULL DEFAULT '[]',
                    failure_resource TEXT,
                    recovery_action TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    runtime_owner_boot_id TEXT,
                    base_artifact_pointer TEXT,
                    candidate_artifact_pointer TEXT,
                    recovery_target TEXT
                );
                CREATE TABLE IF NOT EXISTS reload_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    tx_id TEXT NOT NULL REFERENCES reload_transactions(tx_id),
                    phase TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reload_transactions_phase
                ON reload_transactions(phase);
                CREATE INDEX IF NOT EXISTS idx_reload_events_tx
                ON reload_events(tx_id, sequence);
                """)
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(reload_transactions)")
            }
            additions = {
                "base_generation_id": "TEXT",
                "formal_effects_json": "TEXT NOT NULL DEFAULT '[]'",
                "failure_resource": "TEXT",
                "recovery_action": "TEXT",
                "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                "runtime_owner_boot_id": "TEXT",
                "base_artifact_pointer": "TEXT",
                "candidate_artifact_pointer": "TEXT",
                "recovery_target": "TEXT",
            }
            for name, definition in additions.items():
                if name not in columns:
                    conn.execute(
                        f"ALTER TABLE reload_transactions ADD COLUMN {name} {definition}"
                    )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _append_event(
        conn: sqlite3.Connection,
        tx_id: str,
        phase: ReloadPhase,
        details: dict[str, object],
        created_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO reload_events (tx_id, phase, details_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                tx_id,
                phase,
                json.dumps(details, ensure_ascii=False, sort_keys=True),
                created_at,
            ),
        )


def _record(row: sqlite3.Row | tuple[object, ...]) -> ReloadTransactionRecord:
    return ReloadTransactionRecord(
        tx_id=str(row[0]),
        plugin_id=str(row[1]),
        base_snapshot_id=None if row[2] is None else str(row[2]),
        candidate_snapshot_id=None if row[3] is None else str(row[3]),
        base_generation_id=_optional_string(row[4]),
        generation_id=str(row[5]),
        source_revision=str(row[6]),
        config_revision=str(row[7]),
        phase=cast(ReloadPhase, str(row[8])),
        started_at=str(row[9]),
        updated_at=str(row[10]),
        error=str(row[11]),
        formal_effects=_decode_effects(row[12]),
        failure_resource=_optional_string(row[13]),
        recovery_action=_optional_action(row[14]),
        attempt_count=_stored_attempt_count(row[15]),
        runtime_owner_boot_id=_optional_string(row[16]),
        base_artifact_pointer=_optional_string(row[17]),
        candidate_artifact_pointer=_optional_string(row[18]),
        recovery_target=_optional_recovery_target(row[19]),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_RECOVERY_ACTION_ORDER: dict[RecoveryActionName, int] = {
    "restore_committed": 0,
    "retry_runtime_recovery": 1,
    "retry_generation_cleanup": 2,
    "discard_candidate": 3,
    "restore_candidate": 4,
}
_RECOVERY_ACTIONS = frozenset(_RECOVERY_ACTION_ORDER)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"ReloadTransaction 字段必须是字符串: {value!r}")
    return value


def _detail_string(details: dict[str, object], *names: str) -> str | None:
    for name in names:
        if name in details:
            return _optional_string(details[name])
    return None


def _validate_effects(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise TypeError("ReloadTransaction formal_effects 必须是字符串序列")
    effects: list[str] = []
    for item in cast(list[object] | tuple[object, ...], value):
        if not isinstance(item, str):
            raise TypeError("ReloadTransaction formal_effects 必须只包含字符串")
        effects.append(item)
    return tuple(effects)


def _decode_effects(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return _validate_effects(json.loads(value))
    return _validate_effects(value)


def _merge_effects(
    current: tuple[str, ...], additions: tuple[str, ...]
) -> tuple[str, ...]:
    merged = list(current)
    seen = set(current)
    for effect in additions:
        if effect not in seen:
            merged.append(effect)
            seen.add(effect)
    return tuple(merged)


def _merge_resources(current: str, additions: str) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for resource in (*current.split(","), *additions.split(",")):
        item = resource.strip()
        if item and item not in seen:
            merged.append(item)
            seen.add(item)
    return ",".join(merged)


def _optional_action(value: object) -> RecoveryActionName | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _RECOVERY_ACTIONS:
        raise ValueError(f"ReloadTransaction recovery action 无效: {value!r}")
    return cast(RecoveryActionName, value)


def _optional_recovery_target(value: object) -> RecoveryTarget | None:
    if value is None:
        return None
    if value not in {"base", "candidate"}:
        raise ValueError(f"ReloadTransaction recovery target 无效: {value!r}")
    return cast(RecoveryTarget, value)


def _validate_attempt_count(value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("ReloadTransaction attempt_count 必须是整数")
    if value < minimum or value < 0:
        raise ValueError("ReloadTransaction attempt_count 不能减少或为负数")
    return value


def _stored_attempt_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("ReloadTransaction attempt_count 存储值必须是整数")
    if value < 0:
        raise ValueError("ReloadTransaction attempt_count 不能为负数")
    return value


def _add_failure_evidence(
    details: dict[str, object],
    *,
    base_snapshot_id: str | None,
    candidate_snapshot_id: str | None,
    base_generation_id: str | None,
    generation_id: str,
    formal_effects: tuple[str, ...],
    resource: str | None,
    error: str,
    action: RecoveryActionName | None,
    attempt_count: int,
    runtime_owner_boot_id: str | None,
    base_artifact_pointer: str | None,
    candidate_artifact_pointer: str | None,
    recovery_target: RecoveryTarget | None,
) -> None:
    _ = details.setdefault("old_snapshot_id", base_snapshot_id)
    _ = details.setdefault("new_snapshot_id", candidate_snapshot_id)
    _ = details.setdefault("old_generation_id", base_generation_id)
    _ = details.setdefault("attempt_generation_id", generation_id)
    _ = details.setdefault("formal_effects", formal_effects)
    _ = details.setdefault("resource", resource)
    _ = details.setdefault("error", error)
    _ = details.setdefault("recovery_action", action)
    _ = details.setdefault("attempt_count", attempt_count)
    _ = details.setdefault("attempt", attempt_count)
    _ = details.setdefault("runtime_owner_boot_id", runtime_owner_boot_id)
    _ = details.setdefault("base_artifact_pointer", base_artifact_pointer)
    _ = details.setdefault("candidate_artifact_pointer", candidate_artifact_pointer)
    _ = details.setdefault("recovery_target", recovery_target)


def _recovery_action(
    phase: str,
    persisted: RecoveryActionName | None = None,
) -> RecoveryActionName | None:
    if persisted is not None:
        return persisted
    if phase in {"latest_ready", "discarding", "preparing", "prepared", "validating"}:
        return "discard_candidate"
    if phase in {"commit_started", "promoting", "committed", "draining"}:
        return "restore_committed"
    if phase == "cleanup_failed":
        return "retry_generation_cleanup"
    if phase == "degraded":
        return "retry_runtime_recovery"
    return None

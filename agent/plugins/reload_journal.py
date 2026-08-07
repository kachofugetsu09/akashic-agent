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
    "complete",
    "aborted",
    "recovered",
]
RecoveryActionName = Literal[
    "discard_candidate",
    "restore_candidate",
    "restore_committed",
]
_TERMINAL_PHASES = frozenset({"complete", "aborted", "recovered"})
_TRANSITIONS: dict[str, frozenset[str]] = {
    "preparing": frozenset({"prepared", "aborted"}),
    "prepared": frozenset({"validating", "aborted"}),
    "validating": frozenset({"commit_started", "aborted"}),
    "commit_started": frozenset({"latest_ready", "committed", "aborted", "recovered"}),
    "latest_ready": frozenset({"discarding", "promoting", "aborted", "recovered"}),
    "discarding": frozenset({"aborted"}),
    "promoting": frozenset({"discarding", "committed", "aborted", "recovered"}),
    "committed": frozenset({"draining", "complete", "recovered"}),
    "draining": frozenset({"complete", "recovered"}),
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
        generation_id: str,
        source_revision: str,
        config_revision: str,
    ) -> str:
        now = _now()
        tx_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reload_transactions (
                    tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
                    generation_id, source_revision, config_revision, phase,
                    started_at, updated_at, error
                ) VALUES (?, ?, ?, NULL, ?, ?, ?, 'preparing', ?, ?, '')
                """,
                (
                    tx_id,
                    plugin_id,
                    base_snapshot_id,
                    generation_id,
                    source_revision,
                    config_revision,
                    now,
                    now,
                ),
            )
            self._append_event(conn, tx_id, "preparing", {}, now)
        return tx_id

    def advance(
        self,
        tx_id: str,
        phase: ReloadPhase,
        *,
        candidate_snapshot_id: str | None = None,
        details: dict[str, object] | None = None,
        error: str = "",
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT phase FROM reload_transactions WHERE tx_id = ?",
                (tx_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
            current = str(row[0])
            if phase not in _TRANSITIONS.get(current, frozenset()):
                raise RuntimeError(
                    f"ReloadTransaction 状态跳转无效: {current} -> {phase}"
                )
            now = _now()
            conn.execute(
                """
                UPDATE reload_transactions
                SET phase = ?,
                    candidate_snapshot_id = COALESCE(?, candidate_snapshot_id),
                    updated_at = ?,
                    error = ?
                WHERE tx_id = ?
                """,
                (phase, candidate_snapshot_id, now, error, tx_id),
            )
            self._append_event(conn, tx_id, phase, details or {}, now)

    def get(self, tx_id: str) -> ReloadTransactionRecord:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
                       generation_id, source_revision, config_revision, phase,
                       started_at, updated_at, error
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
                       generation_id, source_revision, config_revision, phase,
                       started_at, updated_at, error
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
                "SELECT phase FROM reload_transactions WHERE tx_id = ?",
                (tx_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"ReloadTransaction 不存在: {tx_id}")
            now = _now()
            phase = cast(ReloadPhase, str(row[0]))
            conn.execute(
                "UPDATE reload_transactions SET updated_at = ? WHERE tx_id = ?",
                (now, tx_id),
            )
            self._append_event(conn, tx_id, phase, details, now)

    def pending_recovery(self) -> tuple[ReloadRecoveryAction, ...]:
        placeholders = ", ".join("?" for _ in _TERMINAL_PHASES)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT tx_id, plugin_id, generation_id, source_revision, phase
                FROM reload_transactions
                WHERE phase NOT IN ({placeholders})
                ORDER BY started_at, tx_id
                """,
                tuple(sorted(_TERMINAL_PHASES)),
            ).fetchall()
        actions = [
            ReloadRecoveryAction(
                tx_id=str(row[0]),
                plugin_id=str(row[1]),
                generation_id=str(row[2]),
                source_revision=str(row[3]),
                phase=cast(ReloadPhase, str(row[4])),
                action=_recovery_action(str(row[4])),
            )
            for row in rows
        ]
        return tuple(
            sorted(
                actions,
                key=lambda item: (
                    item.action == "discard_candidate",
                    item.action == "restore_candidate",
                    item.tx_id,
                ),
            )
        )

    def finish_recovery(self, action: ReloadRecoveryAction) -> None:
        phase: ReloadPhase = (
            "aborted" if action.action == "discard_candidate" else "recovered"
        )
        self.advance(
            action.tx_id,
            phase,
            details={"recovery_action": action.action},
            error="startup recovery",
        )

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS reload_transactions (
                    tx_id TEXT PRIMARY KEY,
                    plugin_id TEXT NOT NULL,
                    base_snapshot_id TEXT,
                    candidate_snapshot_id TEXT,
                    generation_id TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    config_revision TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT NOT NULL
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
        generation_id=str(row[4]),
        source_revision=str(row[5]),
        config_revision=str(row[6]),
        phase=cast(ReloadPhase, str(row[7])),
        started_at=str(row[8]),
        updated_at=str(row[9]),
        error=str(row[10]),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _recovery_action(phase: str) -> RecoveryActionName:
    if phase == "latest_ready":
        return "discard_candidate"
    if phase in {"commit_started", "promoting", "committed", "draining"}:
        return "restore_committed"
    return "discard_candidate"

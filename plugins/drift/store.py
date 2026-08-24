from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator, Literal, NotRequired, TypedDict

_SCHEMA_VERSION = 1
_TABLE_SQL = """
    CREATE TABLE proposals(
        proposal_id TEXT NOT NULL,
        revision TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        status TEXT NOT NULL,
        due_at TEXT NOT NULL,
        next_due TEXT,
        state_version INTEGER NOT NULL,
        selection_token TEXT UNIQUE,
        selected_session_id TEXT,
        selected_turn_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(proposal_id, revision)
    )
"""
_INDEX_SQL = (
    """
    CREATE UNIQUE INDEX proposals_selected_turn_idx
    ON proposals(selected_session_id, selected_turn_id)
    WHERE selected_turn_id IS NOT NULL
    """,
    "CREATE INDEX proposals_due_idx ON proposals(status, due_at)",
)


class DriftRef(TypedDict):
    proposal_id: str
    revision: str
    state_version: int


class DriftProposal(TypedDict):
    ref: DriftRef
    payload: dict[str, object]
    due_at: str
    next_due: str | None
    due: bool


class DriftSnapshot(TypedDict):
    next_due: str | None
    proposals: tuple[DriftProposal, ...]


class DriftSelectResult(TypedDict):
    selected: bool
    reason: NotRequired[str]
    selection_token: str | None
    accepted_turn: dict[str, str] | None


class DriftSelectionReceipt(TypedDict):
    selection_token: str
    ref: dict[str, str]
    payload: dict[str, object]
    due_at: str
    next_due: str | None
    status: str
    accepted_turn: dict[str, str]


class DriftStore:
    """Persist Drift proposals and their Turn-bound lifecycle."""

    def __init__(
        self,
        path: Path,
        *,
        data_access: Literal["read_write", "read_only"] = "read_write",
    ) -> None:
        self.path = path
        self.data_access = data_access

    def initialize(self) -> None:
        """Create or validate the exact Drift schema."""

        with self._transaction(write=self.data_access == "read_write") as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != _SCHEMA_VERSION:
                raise RuntimeError(f"不支持的 Drift schema version: {version}")
            tables = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
                )
            )
            if tables != ("proposals",):
                raise RuntimeError("Drift owned table set 不匹配")
            table_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'proposals'"
            ).fetchone()
            if table_sql is None or _normalize_sql(str(table_sql[0])) != _normalize_sql(
                _TABLE_SQL
            ):
                raise RuntimeError("Drift constraint-bearing table SQL 不匹配")
            columns = tuple(
                (str(row[1]), str(row[2]), int(row[3]), int(row[5]))
                for row in connection.execute("PRAGMA table_info(proposals)")
            )
            if columns != _EXPECTED_COLUMNS:
                raise RuntimeError("Drift schema identity 不匹配")
            if _schema_indexes(connection) != _EXPECTED_INDEXES:
                raise RuntimeError("Drift index identity 不匹配")
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                raise RuntimeError("Drift SQLite integrity check failed")

    def propose(
        self,
        proposal_id: str,
        revision: str,
        payload: Mapping[str, object],
        due_at: datetime,
        *,
        next_due: datetime | None = None,
    ) -> dict[str, object]:
        """Append one idempotent proposal revision for its owning producer."""

        proposal = _identity("proposal_id", proposal_id)
        proposal_revision = _identity("revision", revision)
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        due = _aware_utc(due_at)
        retry = _aware_utc(next_due) if next_due is not None else None
        now = datetime.now(UTC).isoformat()
        with self._transaction(write=True) as connection:
            existing = connection.execute(
                "SELECT payload_json, due_at, next_due FROM proposals "
                "WHERE proposal_id = ? AND revision = ?",
                (proposal, proposal_revision),
            ).fetchone()
            if existing is not None:
                if tuple(existing) != (payload_json, due, retry):
                    raise RuntimeError(
                        f"Drift proposal identity conflict: {proposal}/{proposal_revision}"
                    )
                return {
                    "inserted": False,
                    "ref": {
                        "proposal_id": proposal,
                        "revision": proposal_revision,
                        "state_version": 1,
                    },
                }
            connection.execute(
                """
                INSERT INTO proposals(
                    proposal_id, revision, payload_json, status, due_at, next_due,
                    state_version, selection_token, selected_session_id,
                    selected_turn_id, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, ?, 1, NULL, NULL, NULL, ?, ?)
                """,
                (proposal, proposal_revision, payload_json, due, retry, now, now),
            )
            return {
                "inserted": True,
                "ref": {
                    "proposal_id": proposal,
                    "revision": proposal_revision,
                    "state_version": 1,
                },
            }

    def snapshot(self, now: datetime) -> DriftSnapshot:
        """Return the next durable deadline and frozen due proposals."""

        instant = _aware_utc(now)
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT proposal_id, revision, payload_json, due_at, next_due,
                       state_version
                FROM proposals
                WHERE status IN ('pending', 'deferred')
                ORDER BY due_at, proposal_id, revision
                """
            ).fetchall()
            proposals: tuple[DriftProposal, ...] = tuple(
                {
                    "ref": {
                        "proposal_id": row["proposal_id"],
                        "revision": row["revision"],
                        "state_version": int(row["state_version"]),
                    },
                    "payload": json.loads(row["payload_json"]),
                    "due_at": row["due_at"],
                    "next_due": row["next_due"],
                    "due": row["due_at"] <= instant,
                }
                for row in rows
            )
            return {
                "next_due": rows[0]["due_at"] if rows else None,
                "proposals": proposals,
            }

    def select(
        self,
        ref: Mapping[str, object],
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> DriftSelectResult:
        """CAS one due proposal into a Turn-bound selection."""

        proposal = _identity("proposal_id", ref.get("proposal_id"))
        revision = _identity("revision", ref.get("revision"))
        state_version = ref.get("state_version")
        if type(state_version) is not int or state_version <= 0:
            raise ValueError("Drift state_version 必须是正整数")
        accepted = _accepted_turn(accepted_turn)
        token = "drift-selection:" + secrets.token_hex(16)
        with self._transaction(write=True) as connection:
            existing_turn = connection.execute(
                """
                SELECT 1 FROM proposals
                WHERE selected_session_id = ? AND selected_turn_id = ?
                """,
                (accepted["session_id"], accepted["turn_id"]),
            ).fetchone()
            if existing_turn is not None:
                return {
                    "selected": False,
                    "reason": "turn_already_selected",
                    "selection_token": None,
                    "accepted_turn": None,
                }
            changed = connection.execute(
                """
                UPDATE proposals
                SET status = 'selected', selection_token = ?,
                    selected_session_id = ?, selected_turn_id = ?,
                    state_version = state_version + 1, updated_at = ?
                WHERE proposal_id = ? AND revision = ? AND state_version = ?
                  AND status IN ('pending', 'deferred') AND due_at <= ?
                """,
                (
                    token,
                    accepted["session_id"],
                    accepted["turn_id"],
                    datetime.now(UTC).isoformat(),
                    proposal,
                    revision,
                    state_version,
                    _aware_utc(now),
                ),
            )
            return {
                "selected": changed.rowcount == 1,
                "selection_token": token if changed.rowcount == 1 else None,
                "accepted_turn": accepted if changed.rowcount == 1 else None,
            }

    def selected(self, limit: int = 100) -> tuple[DriftSelectionReceipt, ...]:
        """List stable selected receipts for startup reconciliation."""

        if type(limit) is not int or limit <= 0:
            raise ValueError("limit 必须是正整数")
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT proposal_id, revision, payload_json, due_at, next_due,
                       selection_token, selected_session_id, selected_turn_id
                FROM proposals WHERE status = 'selected'
                ORDER BY due_at, proposal_id, revision LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return tuple(_selection_receipt(row) for row in rows)

    def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> DriftSelectionReceipt | None:
        """Read the exact Drift selection bound to one accepted Turn."""

        accepted = _accepted_turn(accepted_turn)
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """
                SELECT proposal_id, revision, payload_json, due_at, next_due,
                       selection_token, selected_session_id, selected_turn_id
                FROM proposals
                WHERE status = 'selected' AND selected_session_id = ?
                  AND selected_turn_id = ?
                """,
                (accepted["session_id"], accepted["turn_id"]),
            ).fetchone()
            return None if row is None else _selection_receipt(row)

    def transition(self, token: str, action: str) -> dict[str, object]:
        """Commit one selected proposal terminal or retry transition."""

        selection = _identity("selection_token", token)
        if action not in {
            "ready_for_delivery",
            "defer",
            "await_change",
            "invalidated",
        }:
            raise ValueError(f"未知 Drift transition: {action}")
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT status, next_due FROM proposals WHERE selection_token = ?",
                (selection,),
            ).fetchone()
            if row is None:
                return {"changed": False, "reason": "selection_missing"}
            if row["status"] != "selected":
                return {"changed": False, "reason": f"status:{row['status']}"}
            if action == "defer" and row["next_due"] is None:
                raise RuntimeError("Drift defer 缺少 proposal owner 提供的 next_due")
            status = "deferred" if action == "defer" else action
            due_at = row["next_due"] if action == "defer" else None
            connection.execute(
                """
                UPDATE proposals
                SET status = ?, due_at = COALESCE(?, due_at),
                    selection_token = NULL, selected_session_id = NULL,
                    selected_turn_id = NULL, state_version = state_version + 1,
                    updated_at = ?
                WHERE selection_token = ?
                """,
                (status, due_at, datetime.now(UTC).isoformat(), selection),
            )
            return {"changed": True, "status": status, "next_due": due_at}

    @contextmanager
    def _transaction(self, *, write: bool) -> Generator[sqlite3.Connection]:
        """Open one mode-aware SQLite transaction."""

        if write and self.data_access == "read_only":
            raise PermissionError("Drift read-only candidate cannot write shared data")
        if self.data_access == "read_write":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
        else:
            connection = sqlite3.connect(
                self.path.resolve(strict=False).as_uri() + "?mode=ro", uri=True
            )
        connection.row_factory = sqlite3.Row
        try:
            if self.data_access == "read_write":
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("BEGIN IMMEDIATE")
                self._ensure_schema(connection)
            else:
                connection.execute("PRAGMA query_only = ON")
                connection.execute("BEGIN")
            yield connection
            if self.data_access == "read_write":
                connection.commit()
            else:
                connection.rollback()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, _SCHEMA_VERSION):
            raise RuntimeError(f"不支持的 Drift schema version: {version}")
        if version == _SCHEMA_VERSION:
            return
        connection.execute(_TABLE_SQL)
        for statement in _INDEX_SQL:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


_EXPECTED_COLUMNS = (
    ("proposal_id", "TEXT", 1, 1),
    ("revision", "TEXT", 1, 2),
    ("payload_json", "TEXT", 1, 0),
    ("status", "TEXT", 1, 0),
    ("due_at", "TEXT", 1, 0),
    ("next_due", "TEXT", 0, 0),
    ("state_version", "INTEGER", 1, 0),
    ("selection_token", "TEXT", 0, 0),
    ("selected_session_id", "TEXT", 0, 0),
    ("selected_turn_id", "TEXT", 0, 0),
    ("created_at", "TEXT", 1, 0),
    ("updated_at", "TEXT", 1, 0),
)
_EXPECTED_INDEXES = (
    ("proposals_due_idx", 0, "c", 0, ("status", "due_at")),
    (
        "proposals_selected_turn_idx",
        1,
        "c",
        1,
        ("selected_session_id", "selected_turn_id"),
    ),
    ("sqlite_autoindex_proposals_1", 1, "u", 0, ("selection_token",)),
    (
        "sqlite_autoindex_proposals_2",
        1,
        "pk",
        0,
        ("proposal_id", "revision"),
    ),
)


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def _schema_indexes(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, int, str, int, tuple[str, ...]], ...]:
    indexes = []
    for row in connection.execute("PRAGMA index_list(proposals)"):
        name = str(row["name"])
        columns = tuple(
            str(column["name"])
            for column in connection.execute(
                "SELECT name FROM pragma_index_info(?) ORDER BY seqno",
                (name,),
            )
        )
        indexes.append(
            (name, int(row["unique"]), str(row["origin"]), int(row["partial"]), columns)
        )
    return tuple(sorted(indexes))


def _selection_receipt(row: sqlite3.Row) -> DriftSelectionReceipt:
    return {
        "selection_token": row["selection_token"],
        "ref": {"proposal_id": row["proposal_id"], "revision": row["revision"]},
        "payload": json.loads(row["payload_json"]),
        "due_at": row["due_at"],
        "next_due": row["next_due"],
        "status": "selected",
        "accepted_turn": {
            "session_id": row["selected_session_id"],
            "turn_id": row["selected_turn_id"],
        },
    }


def _accepted_turn(value: Mapping[str, object]) -> dict[str, str]:
    return {
        "session_id": _identity("accepted_turn.session_id", value.get("session_id")),
        "turn_id": _identity("accepted_turn.turn_id", value.get("turn_id")),
    }


def _identity(field: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} 必须是非空且无首尾空白的字符串")
    return value


def _aware_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Drift 时间必须带时区")
    return value.astimezone(UTC).isoformat()

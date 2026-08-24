from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator, Literal, NotRequired, TypedDict, cast

_SCHEMA_VERSION = 1
_IMMEDIATE_NOT_BEFORE = "1970-01-01T00:00:00+00:00"
_ColumnIdentity = tuple[int, str, str, int, str | None, int]
_IndexIdentity = tuple[str, int, str, int, tuple[str, ...]]

_TABLE_SQL = {
    "content_state": """
        CREATE TABLE content_state(
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            next_seq INTEGER NOT NULL,
            state_version INTEGER NOT NULL,
            wake_needed INTEGER NOT NULL CHECK(wake_needed IN (0, 1)),
            earliest_not_before TEXT
        )
    """,
    "items": """
        CREATE TABLE items(
            source_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            snapshot_seq INTEGER NOT NULL UNIQUE,
            status TEXT NOT NULL,
            not_before TEXT NOT NULL,
            requires_ack INTEGER NOT NULL CHECK(requires_ack IN (0, 1)),
            item_state_version INTEGER NOT NULL DEFAULT 1,
            selection_token TEXT UNIQUE,
            selected_session_id TEXT,
            selected_turn_id TEXT,
            settlement_ref TEXT UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(source_id, item_id, revision)
        )
    """,
    "submissions": """
        CREATE TABLE submissions(
            source_id TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            submitted_at TEXT NOT NULL,
            PRIMARY KEY(source_id, batch_id)
        )
    """,
}
_INDEX_SQL = (
    "CREATE INDEX items_wake_idx ON items(status, not_before, snapshot_seq)",
    "CREATE INDEX items_source_ack_idx ON items(source_id, status, snapshot_seq)",
    "CREATE UNIQUE INDEX items_selected_turn_idx "
    "ON items(selected_session_id, selected_turn_id) "
    "WHERE selected_turn_id IS NOT NULL",
)
_EXPECTED_COLUMNS: dict[str, tuple[_ColumnIdentity, ...]] = {
    "content_state": (
        (0, "singleton", "INTEGER", 0, None, 1),
        (1, "next_seq", "INTEGER", 1, None, 0),
        (2, "state_version", "INTEGER", 1, None, 0),
        (3, "wake_needed", "INTEGER", 1, None, 0),
        (4, "earliest_not_before", "TEXT", 0, None, 0),
    ),
    "items": (
        (0, "source_id", "TEXT", 1, None, 1),
        (1, "item_id", "TEXT", 1, None, 2),
        (2, "revision", "TEXT", 1, None, 3),
        (3, "payload_json", "TEXT", 1, None, 0),
        (4, "snapshot_seq", "INTEGER", 1, None, 0),
        (5, "status", "TEXT", 1, None, 0),
        (6, "not_before", "TEXT", 1, None, 0),
        (7, "requires_ack", "INTEGER", 1, None, 0),
        (8, "item_state_version", "INTEGER", 1, "1", 0),
        (9, "selection_token", "TEXT", 0, None, 0),
        (10, "selected_session_id", "TEXT", 0, None, 0),
        (11, "selected_turn_id", "TEXT", 0, None, 0),
        (12, "settlement_ref", "TEXT", 0, None, 0),
        (13, "created_at", "TEXT", 1, None, 0),
        (14, "updated_at", "TEXT", 1, None, 0),
    ),
    "submissions": (
        (0, "source_id", "TEXT", 1, None, 1),
        (1, "batch_id", "TEXT", 1, None, 2),
        (2, "fingerprint", "TEXT", 1, None, 0),
        (3, "receipt_json", "TEXT", 1, None, 0),
        (4, "submitted_at", "TEXT", 1, None, 0),
    ),
}
_EXPECTED_INDEXES: dict[str, tuple[_IndexIdentity, ...]] = {
    "content_state": (),
    "items": (
        (
            "items_selected_turn_idx",
            1,
            "c",
            1,
            ("selected_session_id", "selected_turn_id"),
        ),
        ("items_source_ack_idx", 0, "c", 0, ("source_id", "status", "snapshot_seq")),
        ("items_wake_idx", 0, "c", 0, ("status", "not_before", "snapshot_seq")),
        ("sqlite_autoindex_items_1", 1, "u", 0, ("snapshot_seq",)),
        ("sqlite_autoindex_items_2", 1, "u", 0, ("selection_token",)),
        ("sqlite_autoindex_items_3", 1, "u", 0, ("settlement_ref",)),
        (
            "sqlite_autoindex_items_4",
            1,
            "pk",
            0,
            ("source_id", "item_id", "revision"),
        ),
    ),
    "submissions": (
        (
            "sqlite_autoindex_submissions_1",
            1,
            "pk",
            0,
            ("source_id", "batch_id"),
        ),
    ),
}


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def _schema_indexes(
    connection: sqlite3.Connection, table: str
) -> tuple[_IndexIdentity, ...]:
    indexes: list[_IndexIdentity] = []
    for row in connection.execute(f"PRAGMA index_list({table})"):
        index_name = str(row["name"])
        columns = tuple(
            str(column["name"])
            for column in connection.execute(
                "SELECT name FROM pragma_index_info(?) ORDER BY seqno",
                (index_name,),
            )
        )
        indexes.append(
            (
                index_name,
                int(row["unique"]),
                str(row["origin"]),
                int(row["partial"]),
                columns,
            )
        )
    return tuple(sorted(indexes))


class ContentIdentityConflict(RuntimeError):
    """Report reuse of one stable identity with different canonical content."""


class _NormalizedItem(TypedDict):
    item_id: str
    revision: str
    payload_json: str
    not_before: str
    requires_ack: bool


class ContentRef(TypedDict):
    source_id: str
    item_id: str
    revision: str
    state_version: int


class AcceptedTurn(TypedDict):
    session_id: str
    turn_id: str


class SubmissionRef(TypedDict):
    source_id: str
    item_id: str
    revision: str


class ContentSnapshotItem(TypedDict):
    ref: ContentRef
    payload: dict[str, object]
    snapshot_seq: int
    status: str
    not_before: str
    due: bool


class ContentSelectionReceipt(TypedDict):
    ref: ContentRef
    payload: dict[str, object]
    snapshot_seq: int
    status: str
    not_before: str
    requires_ack: bool
    selection_token: str
    accepted_turn: AcceptedTurn


class ContentSnapshot(TypedDict):
    snapshot_seq: int
    state_version: int
    wake_needed: bool
    earliest_not_before: str | None
    items: tuple[ContentSnapshotItem, ...]


class SubmissionReceipt(TypedDict):
    receipt_id: str
    source_id: str
    batch_id: str
    inserted: list[SubmissionRef]
    duplicates: list[SubmissionRef]
    high_watermark: int
    state_version: int
    wake_needed: bool


class ContentSelectResult(TypedDict):
    selected: bool
    reason: NotRequired[str]
    selection_token: str | None
    accepted_turn: AcceptedTurn | None
    state_version: int
    wake_needed: bool
    earliest_not_before: str | None


class ContentTransitionResult(TypedDict):
    changed: bool
    reason: NotRequired[str]
    status: NotRequired[str]
    state_version: NotRequired[int]
    wake_needed: NotRequired[bool]
    earliest_not_before: NotRequired[str | None]


class ContentStore:
    """Persist Content revisions and expose source- and Wake-scoped transitions."""

    def __init__(
        self,
        path: Path,
        *,
        data_access: Literal["read_write", "read_only"] = "read_write",
    ) -> None:
        self.path = path
        self.data_access = data_access

    def initialize(self) -> None:
        """Create or validate the exact schema and SQLite file integrity."""

        with self._transaction(write=self.data_access == "read_write") as connection:
            self._validate_schema(connection)
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                detail = "missing result" if result is None else str(result[0])
                raise RuntimeError(f"Content SQLite integrity check failed: {detail}")

    def submit(
        self,
        source_id: str,
        batch_id: str,
        items: Sequence[Mapping[str, object]],
    ) -> SubmissionReceipt:
        """Commit one idempotent source batch before its cursor may advance."""

        # 1. Freeze and validate the external source batch before opening a transaction.
        source = _identity("source_id", source_id)
        batch = _identity("batch_id", batch_id)
        normalized = tuple(_normalize_item(item) for item in items)
        fingerprint = _batch_fingerprint(normalized)

        # 2. Reuse an exact durable receipt, or append each previously unseen revision.
        with self._transaction(write=True) as connection:
            previous = self._submission_receipt(connection, source, batch, fingerprint)
            if previous is not None:
                return previous
            now = _utc_now()
            inserted, duplicates = self._append_items(
                connection, source, normalized, now
            )
            self._recompute_wake(connection)
            current = self._state(connection)
            receipt: SubmissionReceipt = {
                "receipt_id": f"content-submit:{source}:{batch}",
                "source_id": source,
                "batch_id": batch,
                "inserted": inserted,
                "duplicates": duplicates,
                "high_watermark": int(current["next_seq"]),
                "state_version": int(current["state_version"]),
                "wake_needed": bool(current["wake_needed"]),
            }
            self._record_submission(
                connection, source, batch, fingerprint, receipt, now
            )
            return receipt

    def read_submission(
        self, source_id: str, batch_id: str
    ) -> dict[str, object] | None:
        """Read one exact durable submit receipt without changing Content state."""

        source = _identity("source_id", source_id)
        batch = _identity("batch_id", batch_id)
        with self._verification_transaction() as connection:
            row = connection.execute(
                "SELECT receipt_json FROM submissions WHERE source_id=? AND batch_id=?",
                (source, batch),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(str(row["receipt_json"]))
        if not isinstance(value, dict):
            raise RuntimeError("Content submission receipt must be an object")
        return cast(dict[str, object], value)

    def read_revision(
        self, source_id: str, item_id: str, revision: str
    ) -> dict[str, object] | None:
        """Read one exact source-owned revision without advancing its lifecycle."""

        source = _identity("source_id", source_id)
        item = _identity("item_id", item_id)
        item_revision = _identity("revision", revision)
        with self._verification_transaction() as connection:
            row = connection.execute(
                """
                SELECT payload_json, status, not_before, requires_ack, snapshot_seq
                FROM items WHERE source_id=? AND item_id=? AND revision=?
                """,
                (source, item, item_revision),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            raise RuntimeError("Content revision payload must be an object")
        return {
            "ref": {"source_id": source, "item_id": item, "revision": item_revision},
            "payload": payload,
            "status": str(row["status"]),
            "not_before": str(row["not_before"]),
            "requires_ack": bool(row["requires_ack"]),
            "snapshot_seq": int(row["snapshot_seq"]),
        }

    @staticmethod
    def _submission_receipt(
        connection: sqlite3.Connection,
        source: str,
        batch: str,
        fingerprint: str,
    ) -> SubmissionReceipt | None:
        row = connection.execute(
            """
            SELECT receipt_json, fingerprint FROM submissions
            WHERE source_id = ? AND batch_id = ?
            """,
            (source, batch),
        ).fetchone()
        if row is None:
            return None
        if row["fingerprint"] != fingerprint:
            raise ContentIdentityConflict(
                f"Content batch identity conflict: {source}/{batch}"
            )
        return cast(SubmissionReceipt, json.loads(row["receipt_json"]))

    def _append_items(
        self,
        connection: sqlite3.Connection,
        source: str,
        items: Sequence[_NormalizedItem],
        now: str,
    ) -> tuple[list[SubmissionRef], list[SubmissionRef]]:
        """Append unseen revisions and return inserted and duplicate refs."""

        next_seq = int(self._state(connection)["next_seq"])
        inserted: list[SubmissionRef] = []
        duplicates: list[SubmissionRef] = []
        for item in items:
            existing = connection.execute(
                """
                SELECT payload_json, not_before, requires_ack FROM items
                WHERE source_id = ? AND item_id = ? AND revision = ?
                """,
                (source, item["item_id"], item["revision"]),
            ).fetchone()
            item_ref = _item_ref(source, item)
            if existing is not None:
                _assert_same_revision(source, item, existing)
                duplicates.append(item_ref)
                continue
            next_seq += 1
            _ = connection.execute(
                """
                INSERT INTO items(
                    source_id, item_id, revision, payload_json, snapshot_seq,
                    status, not_before, requires_ack, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                """,
                (
                    source,
                    item["item_id"],
                    item["revision"],
                    item["payload_json"],
                    next_seq,
                    item["not_before"],
                    int(item["requires_ack"]),
                    now,
                    now,
                ),
            )
            inserted.append(item_ref)
        if inserted:
            _ = connection.execute(
                """
                UPDATE content_state
                SET next_seq = ?, state_version = state_version + 1
                WHERE singleton = 1
                """,
                (next_seq,),
            )
        return inserted, duplicates

    @staticmethod
    def _record_submission(
        connection: sqlite3.Connection,
        source: str,
        batch: str,
        fingerprint: str,
        receipt: Mapping[str, object],
        now: str,
    ) -> None:
        _ = connection.execute(
            """
            INSERT INTO submissions(
                source_id, batch_id, fingerprint, receipt_json, submitted_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                source,
                batch,
                fingerprint,
                json.dumps(receipt, sort_keys=True, separators=(",", ":")),
                now,
            ),
        )

    def snapshot(self, now: datetime) -> ContentSnapshot:
        """Return one immutable high-watermark view for a Wake proposal."""

        instant = _aware_utc(now)
        with self._transaction(write=False) as connection:
            state = self._state(connection)
            high_watermark = int(state["next_seq"])
            rows = connection.execute(
                """
                SELECT source_id, item_id, revision, payload_json, snapshot_seq,
                       status, not_before, item_state_version
                FROM items
                WHERE snapshot_seq <= ? AND status IN ('pending', 'deferred')
                ORDER BY snapshot_seq
                """,
                (high_watermark,),
            ).fetchall()
            items: tuple[ContentSnapshotItem, ...] = tuple(
                {
                    "ref": {
                        "source_id": row["source_id"],
                        "item_id": row["item_id"],
                        "revision": row["revision"],
                        "state_version": int(row["item_state_version"]),
                    },
                    "payload": json.loads(row["payload_json"]),
                    "snapshot_seq": int(row["snapshot_seq"]),
                    "status": row["status"],
                    "not_before": row["not_before"],
                    "due": row["not_before"] <= instant,
                }
                for row in rows
            )
            return {
                "snapshot_seq": high_watermark,
                "state_version": int(state["state_version"]),
                "wake_needed": bool(state["wake_needed"]),
                "earliest_not_before": state["earliest_not_before"],
                "items": items,
            }

    def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> ContentSelectionReceipt | None:
        """Recover Content's durable selection from one accepted Turn receipt."""

        accepted = _normalize_accepted_turn(accepted_turn)
        with self._transaction(write=False) as connection:
            row = self._selection_row(connection, accepted)
            return None if row is None else self._selection_receipt(row)

    def selected(self, limit: int = 100) -> tuple[ContentSelectionReceipt, ...]:
        """Return selected rows in stable inbox order for external recovery."""

        if type(limit) is not int or limit <= 0:
            raise ValueError("limit 必须是正整数")
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT source_id, item_id, revision, payload_json, snapshot_seq,
                       status, not_before, requires_ack, item_state_version,
                       selection_token, selected_session_id, selected_turn_id
                FROM items
                WHERE status = 'selected'
                ORDER BY snapshot_seq
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return tuple(self._selection_receipt(row) for row in rows)

    def select(
        self,
        item_ref: Mapping[str, object],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> ContentSelectResult:
        """CAS one frozen eligible revision into a Turn-bound selection."""

        ref = _normalize_ref(item_ref)
        if type(snapshot_seq) is not int or snapshot_seq < 0:
            raise ValueError("snapshot_seq 必须是非负整数")
        accepted = _normalize_accepted_turn(accepted_turn)
        instant = _aware_utc(now)
        with self._transaction(write=True) as connection:
            existing = self._selection_row(connection, accepted)
            if existing is not None:
                state = self._state(connection)
                return {
                    "selected": False,
                    "reason": "turn_already_selected",
                    "selection_token": None,
                    "accepted_turn": None,
                    "state_version": int(state["state_version"]),
                    "wake_needed": bool(state["wake_needed"]),
                    "earliest_not_before": state["earliest_not_before"],
                }
            token = f"content-selection:{secrets.token_hex(16)}"
            cursor = connection.execute(
                """
                UPDATE items
                SET status = 'selected', selection_token = ?,
                    selected_session_id = ?, selected_turn_id = ?,
                    item_state_version = item_state_version + 1, updated_at = ?
                WHERE source_id = ? AND item_id = ? AND revision = ?
                  AND snapshot_seq <= ?
                  AND item_state_version = ?
                  AND status IN ('pending', 'deferred')
                  AND not_before <= ?
                """,
                (
                    token,
                    accepted["session_id"],
                    accepted["turn_id"],
                    _utc_now(),
                    ref["source_id"],
                    ref["item_id"],
                    ref["revision"],
                    snapshot_seq,
                    ref["state_version"],
                    instant,
                ),
            )
            selected = cursor.rowcount == 1
            if selected:
                _ = connection.execute(
                    "UPDATE content_state SET state_version = state_version + 1 WHERE singleton = 1"
                )
            self._recompute_wake(connection)
            state = self._state(connection)
            return {
                "selected": selected,
                "selection_token": token if selected else None,
                "accepted_turn": accepted if selected else None,
                "state_version": int(state["state_version"]),
                "wake_needed": bool(state["wake_needed"]),
                "earliest_not_before": state["earliest_not_before"],
            }

    @staticmethod
    def _selection_row(
        connection: sqlite3.Connection, accepted_turn: AcceptedTurn
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT source_id, item_id, revision, payload_json, snapshot_seq,
                   status, not_before, requires_ack, item_state_version,
                   selection_token, selected_session_id, selected_turn_id
            FROM items
            WHERE selected_session_id = ? AND selected_turn_id = ?
            """,
            (accepted_turn["session_id"], accepted_turn["turn_id"]),
        ).fetchone()

    @staticmethod
    def _selection_receipt(row: sqlite3.Row) -> ContentSelectionReceipt:
        return {
            "selection_token": row["selection_token"],
            "ref": {
                "source_id": row["source_id"],
                "item_id": row["item_id"],
                "revision": row["revision"],
                "state_version": int(row["item_state_version"]),
            },
            "payload": json.loads(row["payload_json"]),
            "snapshot_seq": int(row["snapshot_seq"]),
            "status": row["status"],
            "not_before": row["not_before"],
            "requires_ack": bool(row["requires_ack"]),
            "accepted_turn": {
                "session_id": row["selected_session_id"],
                "turn_id": row["selected_turn_id"],
            },
        }

    def transition(
        self,
        selection_token: str,
        action: str,
        *,
        not_before: datetime | None = None,
        settlement_ref: str | None = None,
    ) -> ContentTransitionResult:
        """Commit one explicit domain transition without inferring Turn state."""

        token = _identity("selection_token", selection_token)
        if action not in {
            "ready_for_delivery",
            "defer",
            "await_change",
            "invalidated",
            "abandoned",
            "expired",
            "delivered",
        }:
            raise ValueError(f"未知 Content transition: {action}")
        if action == "defer" and not_before is None:
            raise ValueError("defer 必须提供 not_before")
        if action == "delivered" and settlement_ref is None:
            raise ValueError("delivered 必须提供 settlement_ref")
        if action != "delivered" and settlement_ref is not None:
            raise ValueError("只有 delivered transition 可以提供 settlement_ref")
        deadline = _aware_utc(not_before) if not_before is not None else None
        settlement = (
            _identity("settlement_ref", settlement_ref)
            if settlement_ref is not None
            else None
        )

        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT status, requires_ack FROM items WHERE selection_token = ?",
                (token,),
            ).fetchone()
            if row is None:
                return {"changed": False, "reason": "selection_missing"}
            allowed_statuses = (
                {"ready_for_delivery"}
                if action == "delivered"
                else (
                    {"selected", "ready_for_delivery"}
                    if action == "abandoned"
                    else {"selected"}
                )
            )
            if row["status"] not in allowed_statuses:
                return {"changed": False, "reason": f"status:{row['status']}"}
            status = "deferred" if action == "defer" else action
            if action == "delivered" and not bool(row["requires_ack"]):
                status = "settled"
            _ = connection.execute(
                """
                UPDATE items
                SET status = ?, not_before = COALESCE(?, not_before),
                    settlement_ref = COALESCE(?, settlement_ref),
                    item_state_version = item_state_version + 1, updated_at = ?
                WHERE selection_token = ?
                """,
                (status, deadline, settlement, _utc_now(), token),
            )
            _ = connection.execute(
                "UPDATE content_state SET state_version = state_version + 1 WHERE singleton = 1"
            )
            self._recompute_wake(connection)
            state = self._state(connection)
            return {
                "changed": True,
                "status": status,
                "state_version": int(state["state_version"]),
                "wake_needed": bool(state["wake_needed"]),
                "earliest_not_before": state["earliest_not_before"],
            }

    def unsettled(
        self, source_id: str, limit: int = 100
    ) -> tuple[dict[str, object], ...]:
        """Return only delivered rows owned by one bound source."""

        source = _identity("source_id", source_id)
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit 必须是正整数")
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT source_id, item_id, revision, settlement_ref, payload_json
                FROM items
                WHERE source_id = ? AND status = 'delivered' AND requires_ack = 1
                ORDER BY snapshot_seq
                LIMIT ?
                """,
                (source, limit),
            ).fetchall()
            return tuple(
                {
                    "ref": {
                        "source_id": row["source_id"],
                        "item_id": row["item_id"],
                        "revision": row["revision"],
                    },
                    "settlement_ref": row["settlement_ref"],
                    "payload": json.loads(row["payload_json"]),
                }
                for row in rows
            )

    def pending_delivery(self, limit: int = 100) -> tuple[dict[str, object], ...]:
        """Return body-free ready selections for the delivery composition owner."""

        if type(limit) is not int or limit <= 0:
            raise ValueError("limit 必须是正整数")
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT selection_token, selected_session_id, selected_turn_id,
                       snapshot_seq
                FROM items
                WHERE status = 'ready_for_delivery'
                ORDER BY snapshot_seq
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return tuple(
                {
                    "selection_token": _identity(
                        "selection_token", row["selection_token"]
                    ),
                    "accepted_turn": {
                        "session_id": _identity(
                            "selected_session_id", row["selected_session_id"]
                        ),
                        "turn_id": _identity(
                            "selected_turn_id", row["selected_turn_id"]
                        ),
                    },
                }
                for row in rows
            )

    def delivery(
        self, accepted_turn: Mapping[str, object]
    ) -> dict[str, object] | None:
        """Read a body-free delivery receipt by its accepted Turn identity."""

        accepted = _normalize_accepted_turn(accepted_turn)
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """
                SELECT selection_token, selected_session_id, selected_turn_id,
                       status, settlement_ref
                FROM items
                WHERE selected_session_id = ? AND selected_turn_id = ?
                """,
                (accepted["session_id"], accepted["turn_id"]),
            ).fetchone()
            if row is None:
                return None
            result: dict[str, object] = {
                "selection_token": row["selection_token"],
                "accepted_turn": dict(accepted),
                "status": row["status"],
                "settlement_ref": row["settlement_ref"],
            }
            if row["status"] in {"delivered", "settled"}:
                settlement = _identity("settlement_ref", row["settlement_ref"])
                result["receipt"] = _delivery_receipt(
                    str(row["selection_token"]), settlement
                )
            return result

    def settle_delivery(
        self,
        selection_token: str,
        settlement_ref: str,
    ) -> dict[str, object]:
        """Idempotently bind one projected logical delivery to its Content selection."""

        token = _identity("selection_token", selection_token)
        settlement = _identity("settlement_ref", settlement_ref)
        receipt = _delivery_receipt(token, settlement)

        with self._transaction(write=True) as connection:
            row = connection.execute(
                """
                SELECT status, requires_ack, settlement_ref
                FROM items WHERE selection_token = ?
                """,
                (token,),
            ).fetchone()
            if row is None:
                return {"settled": False, "reason": "selection_missing"}
            if row["status"] in {"delivered", "settled"}:
                if row["settlement_ref"] != settlement:
                    raise RuntimeError("Content delivery settlement identity conflict")
                return {
                    "settled": True,
                    "duplicate": True,
                    "status": row["status"],
                    "receipt": receipt,
                }
            if row["status"] != "ready_for_delivery":
                return {"settled": False, "reason": f"status:{row['status']}"}
            status = "delivered" if bool(row["requires_ack"]) else "settled"
            _ = connection.execute(
                """
                UPDATE items
                SET status = ?, settlement_ref = ?,
                    item_state_version = item_state_version + 1, updated_at = ?
                WHERE selection_token = ? AND status = 'ready_for_delivery'
                """,
                (status, settlement, _utc_now(), token),
            )
            _ = connection.execute(
                """
                UPDATE content_state
                SET state_version = state_version + 1 WHERE singleton = 1
                """
            )
            self._recompute_wake(connection)
            return {
                "settled": True,
                "duplicate": False,
                "status": status,
                "receipt": receipt,
            }

    def ack(self, source_id: str, settlement_ref: str) -> dict[str, object]:
        """Settle one delivered row only through its source-bound view."""

        source = _identity("source_id", source_id)
        settlement = _identity("settlement_ref", settlement_ref)
        with self._transaction(write=True) as connection:
            row = connection.execute(
                """
                SELECT status FROM items
                WHERE source_id = ? AND settlement_ref = ?
                """,
                (source, settlement),
            ).fetchone()
            if row is None:
                return {"settled": False, "reason": "settlement_missing"}
            if row["status"] == "settled":
                return {"settled": True, "duplicate": True}
            if row["status"] != "delivered":
                return {"settled": False, "reason": f"status:{row['status']}"}
            _ = connection.execute(
                """
                UPDATE items SET status = 'settled',
                    item_state_version = item_state_version + 1, updated_at = ?
                WHERE source_id = ? AND settlement_ref = ? AND status = 'delivered'
                """,
                (_utc_now(), source, settlement),
            )
            _ = connection.execute(
                "UPDATE content_state SET state_version = state_version + 1 WHERE singleton = 1"
            )
            return {"settled": True, "duplicate": False}

    def state_counts(self) -> dict[str, int]:
        """Expose deterministic state counts for tests and runtime inspection."""

        with self._transaction(write=False) as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM items GROUP BY status"
            ).fetchall()
            return {str(row["status"]): int(row["count"]) for row in rows}

    @contextmanager
    def _transaction(self, *, write: bool) -> Generator[sqlite3.Connection]:
        """Open one mode-aware SQLite transaction and close it at the boundary."""

        # 1. Reject every candidate write at the store's single transaction boundary.
        if write and self.data_access == "read_only":
            raise PermissionError("Content read-only candidate cannot write shared data")

        # 2. Preserve the formal store's serialized transaction and lazy schema setup.
        if self.data_access == "read_write":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
        else:
            database_uri = self.path.resolve(strict=False).as_uri() + "?mode=ro"
            connection = sqlite3.connect(database_uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            if self.data_access == "read_write":
                _ = connection.execute("PRAGMA journal_mode = WAL")
                _ = connection.execute("PRAGMA foreign_keys = ON")
                _ = connection.execute("BEGIN IMMEDIATE")
                self._ensure_schema(connection)
            else:
                _ = connection.execute("PRAGMA query_only = ON")
                _ = connection.execute("BEGIN")
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

    @contextmanager
    def _verification_transaction(self) -> Generator[sqlite3.Connection]:
        """Open an exact read-only view without lazy schema or WAL initialization."""

        # 1. Immutable SQLite ignores WAL, so offline handoff must see a checkpoint.
        wal_path = self.path.with_name(self.path.name + "-wal")
        if wal_path.is_file() and wal_path.stat().st_size > 0:
            raise RuntimeError(
                "Content handoff verification requires a checkpointed offline store"
            )

        # 2. Immutable mode proves the verification read itself creates no WAL/SHM.
        database_uri = (
            self.path.resolve(strict=False).as_uri() + "?mode=ro&immutable=1"
        )
        connection = sqlite3.connect(database_uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            _ = connection.execute("PRAGMA query_only = ON")
            _ = connection.execute("BEGIN")
            yield connection
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
            raise RuntimeError(f"不支持的 Content schema version: {version}")
        if version == _SCHEMA_VERSION:
            return
        statements = (
            _TABLE_SQL["content_state"],
            "INSERT INTO content_state VALUES(1, 0, 0, 0, NULL)",
            _TABLE_SQL["items"],
            *_INDEX_SQL,
            _TABLE_SQL["submissions"],
            f"PRAGMA user_version = {_SCHEMA_VERSION}",
        )
        _ = connection.executescript(";\n".join(statements) + ";")

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        """Reject every version-one database that is not this exact schema."""

        # 1. Exact identity includes the schema version, not only matching tables.
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != _SCHEMA_VERSION:
            raise RuntimeError(f"不支持的 Content schema version: {version}")

        # 2. Match the complete owned table set and each constraint-bearing DDL.
        tables = {
            str(row["name"]): str(row["sql"])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if set(tables) != set(_TABLE_SQL):
            raise RuntimeError(
                f"Content schema mismatch: tables expected={sorted(_TABLE_SQL)} "
                f"actual={sorted(tables)}"
            )
        for table, expected_sql in _TABLE_SQL.items():
            if _normalize_sql(tables[table]) != _normalize_sql(expected_sql):
                raise RuntimeError(f"Content schema mismatch: {table} table SQL")

        # 3. Match storage attributes and every explicit or constraint-owned index.
        for table, expected_columns in _EXPECTED_COLUMNS.items():
            actual_columns = tuple(
                (
                    int(row["cid"]),
                    str(row["name"]),
                    str(row["type"]),
                    int(row["notnull"]),
                    row["dflt_value"],
                    int(row["pk"]),
                )
                for row in connection.execute(f"PRAGMA table_info({table})")
            )
            if actual_columns != expected_columns:
                raise RuntimeError(f"Content schema mismatch: {table} columns")
            actual_indexes = _schema_indexes(connection, table)
            if actual_indexes != _EXPECTED_INDEXES[table]:
                raise RuntimeError(f"Content schema mismatch: {table} indexes")

        # 4. The state table owns one singleton row, never zero or a second row.
        state_rows = tuple(
            int(row["singleton"])
            for row in connection.execute("SELECT singleton FROM content_state")
        )
        if state_rows != (1,):
            raise RuntimeError("Content schema mismatch: content_state singleton row")

    @staticmethod
    def _state(connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM content_state WHERE singleton = 1"
        ).fetchone()
        assert row is not None
        return row

    @staticmethod
    def _recompute_wake(connection: sqlite3.Connection) -> None:
        row = connection.execute("""
            SELECT MIN(not_before) AS earliest
            FROM items WHERE status IN ('pending', 'deferred')
            """).fetchone()
        earliest = row["earliest"] if row is not None else None
        _ = connection.execute(
            """
            UPDATE content_state
            SET wake_needed = ?, earliest_not_before = ?
            WHERE singleton = 1
            """,
            (int(earliest is not None), earliest),
        )


def _normalize_item(item: Mapping[str, object]) -> _NormalizedItem:
    if not isinstance(item, Mapping):
        raise ValueError("Content item 必须是 Mapping")
    item_id = _identity("item_id", item.get("item_id"))
    revision = _identity("revision", item.get("revision"))
    payload = item.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Content payload 必须是 Mapping")
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    raw_not_before = item.get("not_before")
    not_before = (
        _IMMEDIATE_NOT_BEFORE
        if raw_not_before is None
        else _aware_utc(_parse_datetime(raw_not_before))
    )
    requires_ack = item.get("requires_ack", True)
    if type(requires_ack) is not bool:
        raise ValueError("requires_ack 必须是 bool")
    return {
        "item_id": item_id,
        "revision": revision,
        "payload_json": payload_json,
        "not_before": not_before,
        "requires_ack": requires_ack,
    }


def _normalize_ref(item_ref: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(item_ref, Mapping):
        raise ValueError("Content item_ref 必须是 Mapping")
    state_version = item_ref.get("state_version")
    if type(state_version) is not int or state_version <= 0:
        raise ValueError("Content item_ref state_version 必须是正整数")
    return {
        "source_id": _identity("source_id", item_ref.get("source_id")),
        "item_id": _identity("item_id", item_ref.get("item_id")),
        "revision": _identity("revision", item_ref.get("revision")),
        "state_version": state_version,
    }


def _normalize_accepted_turn(value: Mapping[str, object]) -> AcceptedTurn:
    if not isinstance(value, Mapping):
        raise ValueError("accepted_turn 必须是 Mapping")
    return {
        "session_id": _identity("accepted_turn.session_id", value.get("session_id")),
        "turn_id": _identity("accepted_turn.turn_id", value.get("turn_id")),
    }


def _item_ref(source_id: str, item: _NormalizedItem) -> SubmissionRef:
    return {
        "source_id": source_id,
        "item_id": item["item_id"],
        "revision": item["revision"],
    }


def _identity(field: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} 必须是非空且无首尾空白的字符串")
    return value


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError("not_before 必须是 datetime 或 ISO 字符串")


def _aware_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Content 时间必须带时区")
    return value.astimezone(UTC).isoformat()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _batch_fingerprint(items: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _delivery_receipt(selection_token: str, settlement_ref: str) -> str:
    payload = f"{selection_token}\x00{settlement_ref}".encode("utf-8")
    return "content-delivery:" + hashlib.sha256(payload).hexdigest()


def _assert_same_revision(
    source_id: str,
    item: _NormalizedItem,
    existing: sqlite3.Row,
) -> None:
    same = (
        existing["payload_json"] == item["payload_json"]
        and existing["not_before"] == item["not_before"]
        and bool(existing["requires_ack"]) is item["requires_ack"]
    )
    if not same:
        raise ContentIdentityConflict(
            "Content revision identity conflict: "
            f"{source_id}/{item['item_id']}/{item['revision']}"
        )

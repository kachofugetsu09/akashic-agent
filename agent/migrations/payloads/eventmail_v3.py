"""Frozen EventMail v3 payload used only by the 2026-08-28 Yoyo migration."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator, Literal, NotRequired, TypedDict, cast

_SCHEMA_VERSION = 3
_IMMEDIATE_NOT_BEFORE = "1970-01-01T00:00:00+00:00"
_ColumnIdentity = tuple[int, str, str, int, str | None, int]
_IndexIdentity = tuple[str, int, str, int, tuple[str, ...]]

_TABLE_SQL = {
    "mail_envelopes": """
        CREATE TABLE mail_envelopes(
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            mail_id TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL CHECK(kind IN ('content', 'alert', 'context')),
            source_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            not_before TEXT,
            expires_at TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(kind, source_id, item_id, revision)
        )
    """,
    "mail_transitions": """
        CREATE TABLE mail_transitions(
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            transition_id TEXT NOT NULL UNIQUE,
            mail_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('content', 'alert', 'context')),
            action TEXT NOT NULL,
            detail_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(mail_id) REFERENCES mail_envelopes(mail_id)
        )
    """,
    "alert_projection": """
        CREATE TABLE alert_projection(
            source_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            mail_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL CHECK(status IN ('pending', 'selected', 'delivered', 'skipped', 'expired')),
            not_before TEXT NOT NULL,
            expires_at TEXT,
            accepted_session TEXT,
            accepted_turn TEXT,
            PRIMARY KEY(source_id, event_id),
            FOREIGN KEY(mail_id) REFERENCES mail_envelopes(mail_id),
            CHECK((status = 'selected') = (accepted_session IS NOT NULL AND accepted_turn IS NOT NULL))
        )
    """,
    "context_projection": """
        CREATE TABLE context_projection(
            source_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            mail_id TEXT NOT NULL UNIQUE,
            expires_at TEXT,
            PRIMARY KEY(source_id, event_id),
            FOREIGN KEY(mail_id) REFERENCES mail_envelopes(mail_id)
        )
    """,
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
    "content_selections": """
        CREATE TABLE content_selections(
            selection_token TEXT PRIMARY KEY,
            accepted_session_id TEXT NOT NULL,
            accepted_turn_id TEXT NOT NULL,
            snapshot_seq INTEGER NOT NULL,
            status TEXT NOT NULL,
            driver_source_id TEXT NOT NULL,
            driver_item_id TEXT NOT NULL,
            driver_revision TEXT NOT NULL,
            decision_format TEXT NOT NULL,
            settlement_ref TEXT UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(accepted_session_id, accepted_turn_id)
        )
    """,
    "content_selection_members": """
        CREATE TABLE content_selection_members(
            selection_token TEXT NOT NULL,
            position INTEGER NOT NULL,
            source_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            selected_for_delivery INTEGER NOT NULL DEFAULT 0
                CHECK(selected_for_delivery IN (0, 1)),
            settlement_ref TEXT UNIQUE,
            PRIMARY KEY(selection_token, source_id, item_id, revision),
            FOREIGN KEY(selection_token) REFERENCES content_selections(selection_token)
        )
    """,
}
_INDEX_SQL = (
    "CREATE INDEX mail_envelopes_kind_seq_idx ON mail_envelopes(kind, seq)",
    "CREATE INDEX mail_transitions_mail_seq_idx ON mail_transitions(mail_id, seq)",
    "CREATE INDEX alert_projection_due_idx ON alert_projection(status, not_before)",
    "CREATE INDEX context_projection_expiry_idx ON context_projection(expires_at)",
    "CREATE INDEX items_wake_idx ON items(status, not_before, snapshot_seq)",
    "CREATE INDEX items_source_ack_idx ON items(source_id, status, snapshot_seq)",
    "CREATE UNIQUE INDEX items_selected_turn_idx "
    "ON items(selected_session_id, selected_turn_id) "
    "WHERE selected_turn_id IS NOT NULL",
    "CREATE INDEX content_selection_status_idx "
    "ON content_selections(status, snapshot_seq)",
    "CREATE INDEX content_selection_members_order_idx "
    "ON content_selection_members(selection_token, position)",
)
_EXPECTED_COLUMNS: dict[str, tuple[_ColumnIdentity, ...]] = {
    "mail_envelopes": (
        (0, "seq", "INTEGER", 0, None, 1),
        (1, "mail_id", "TEXT", 1, None, 0),
        (2, "kind", "TEXT", 1, None, 0),
        (3, "source_id", "TEXT", 1, None, 0),
        (4, "item_id", "TEXT", 1, None, 0),
        (5, "revision", "TEXT", 1, None, 0),
        (6, "payload_json", "TEXT", 1, None, 0),
        (7, "observed_at", "TEXT", 1, None, 0),
        (8, "not_before", "TEXT", 0, None, 0),
        (9, "expires_at", "TEXT", 0, None, 0),
        (10, "created_at", "TEXT", 1, None, 0),
    ),
    "mail_transitions": (
        (0, "seq", "INTEGER", 0, None, 1),
        (1, "transition_id", "TEXT", 1, None, 0),
        (2, "mail_id", "TEXT", 1, None, 0),
        (3, "kind", "TEXT", 1, None, 0),
        (4, "action", "TEXT", 1, None, 0),
        (5, "detail_json", "TEXT", 1, None, 0),
        (6, "created_at", "TEXT", 1, None, 0),
    ),
    "alert_projection": (
        (0, "source_id", "TEXT", 1, None, 1),
        (1, "event_id", "TEXT", 1, None, 2),
        (2, "mail_id", "TEXT", 1, None, 0),
        (3, "status", "TEXT", 1, None, 0),
        (4, "not_before", "TEXT", 1, None, 0),
        (5, "expires_at", "TEXT", 0, None, 0),
        (6, "accepted_session", "TEXT", 0, None, 0),
        (7, "accepted_turn", "TEXT", 0, None, 0),
    ),
    "context_projection": (
        (0, "source_id", "TEXT", 1, None, 1),
        (1, "event_id", "TEXT", 1, None, 2),
        (2, "mail_id", "TEXT", 1, None, 0),
        (3, "expires_at", "TEXT", 0, None, 0),
    ),
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
    "content_selections": (
        (0, "selection_token", "TEXT", 0, None, 1),
        (1, "accepted_session_id", "TEXT", 1, None, 0),
        (2, "accepted_turn_id", "TEXT", 1, None, 0),
        (3, "snapshot_seq", "INTEGER", 1, None, 0),
        (4, "status", "TEXT", 1, None, 0),
        (5, "driver_source_id", "TEXT", 1, None, 0),
        (6, "driver_item_id", "TEXT", 1, None, 0),
        (7, "driver_revision", "TEXT", 1, None, 0),
        (8, "decision_format", "TEXT", 1, None, 0),
        (9, "settlement_ref", "TEXT", 0, None, 0),
        (10, "created_at", "TEXT", 1, None, 0),
        (11, "updated_at", "TEXT", 1, None, 0),
    ),
    "content_selection_members": (
        (0, "selection_token", "TEXT", 1, None, 1),
        (1, "position", "INTEGER", 1, None, 0),
        (2, "source_id", "TEXT", 1, None, 2),
        (3, "item_id", "TEXT", 1, None, 3),
        (4, "revision", "TEXT", 1, None, 4),
        (5, "selected_for_delivery", "INTEGER", 1, "0", 0),
        (6, "settlement_ref", "TEXT", 0, None, 0),
    ),
}
_EXPECTED_INDEXES: dict[str, tuple[_IndexIdentity, ...]] = {
    "mail_envelopes": (
        ("mail_envelopes_kind_seq_idx", 0, "c", 0, ("kind", "seq")),
        ("sqlite_autoindex_mail_envelopes_1", 1, "u", 0, ("mail_id",)),
        (
            "sqlite_autoindex_mail_envelopes_2",
            1,
            "u",
            0,
            ("kind", "source_id", "item_id", "revision"),
        ),
    ),
    "mail_transitions": (
        ("mail_transitions_mail_seq_idx", 0, "c", 0, ("mail_id", "seq")),
        ("sqlite_autoindex_mail_transitions_1", 1, "u", 0, ("transition_id",)),
    ),
    "alert_projection": (
        ("alert_projection_due_idx", 0, "c", 0, ("status", "not_before")),
        ("sqlite_autoindex_alert_projection_1", 1, "u", 0, ("mail_id",)),
        ("sqlite_autoindex_alert_projection_2", 1, "pk", 0, ("source_id", "event_id")),
    ),
    "context_projection": (
        ("context_projection_expiry_idx", 0, "c", 0, ("expires_at",)),
        ("sqlite_autoindex_context_projection_1", 1, "u", 0, ("mail_id",)),
        (
            "sqlite_autoindex_context_projection_2",
            1,
            "pk",
            0,
            ("source_id", "event_id"),
        ),
    ),
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
    "content_selections": (
        (
            "content_selection_status_idx",
            0,
            "c",
            0,
            ("status", "snapshot_seq"),
        ),
        (
            "sqlite_autoindex_content_selections_1",
            1,
            "pk",
            0,
            ("selection_token",),
        ),
        (
            "sqlite_autoindex_content_selections_2",
            1,
            "u",
            0,
            ("settlement_ref",),
        ),
        (
            "sqlite_autoindex_content_selections_3",
            1,
            "u",
            0,
            ("accepted_session_id", "accepted_turn_id"),
        ),
    ),
    "content_selection_members": (
        (
            "content_selection_members_order_idx",
            0,
            "c",
            0,
            ("selection_token", "position"),
        ),
        (
            "sqlite_autoindex_content_selection_members_1",
            1,
            "u",
            0,
            ("settlement_ref",),
        ),
        (
            "sqlite_autoindex_content_selection_members_2",
            1,
            "pk",
            0,
            ("selection_token", "source_id", "item_id", "revision"),
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


class EventMailIdentityConflict(RuntimeError):
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
    items: tuple[ContentSnapshotItem, ...]
    decision_format: str


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


class EventMailV3MigrationStore:
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
            self._append_content_projections(
                connection,
                (
                    _mail_id(
                        "content", ref["source_id"], ref["item_id"], ref["revision"]
                    )
                    for ref in inserted
                ),
            )
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
            raise EventMailIdentityConflict(
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
                SELECT payload_json, not_before, requires_ack, item_state_version
                FROM items
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
            mail_id, mail_inserted = self._append_envelope(
                connection,
                kind="content",
                source_id=source,
                item_id=item["item_id"],
                revision=item["revision"],
                payload_json=item["payload_json"],
                observed_at=now,
                not_before=item["not_before"],
                expires_at=None,
            )
            if not mail_inserted:
                raise RuntimeError("新 Content projection 缺少新 EventMail envelope")
            self._append_transition(connection, mail_id, "content", "received", {})
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

    def report_alert(
        self,
        *,
        source_id: str,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]:
        """Append one Alert revision and publish it as the current projection."""

        source = _identity("source_id", source_id)
        event = _identity("event_id", event_id)
        observed = _aware_utc(observed_at)
        expiry = None if expires_at is None else _aware_utc(expires_at)
        if expiry is not None and expiry <= observed:
            raise ValueError("Alert expires_at 必须晚于 observed_at")
        encoded = _payload_json(payload)
        with self._transaction(write=True) as connection:
            mail_id, inserted = self._append_envelope(
                connection,
                kind="alert",
                source_id=source,
                item_id=event,
                revision=observed,
                payload_json=encoded,
                observed_at=observed,
                not_before=observed,
                expires_at=expiry,
            )
            current = connection.execute(
                """
                SELECT projection.mail_id, envelope.observed_at
                FROM alert_projection AS projection
                JOIN mail_envelopes AS envelope ON envelope.mail_id=projection.mail_id
                WHERE projection.source_id=? AND projection.event_id=?
                """,
                (source, event),
            ).fetchone()
            if current is not None and str(current["mail_id"]) == mail_id:
                return {
                    "accepted": False,
                    "source_id": source,
                    "event_id": event,
                    "mail_id": mail_id,
                }
            if current is not None and str(current["observed_at"]) > observed:
                if inserted:
                    self._append_transition(
                        connection, mail_id, "alert", "received", {}
                    )
                    self._append_transition(
                        connection,
                        mail_id,
                        "alert",
                        "superseded",
                        {"by": str(current["mail_id"])},
                    )
                return {
                    "accepted": inserted,
                    "projected": False,
                    "source_id": source,
                    "event_id": event,
                    "mail_id": mail_id,
                }
            if current is not None:
                self._append_transition(
                    connection,
                    str(current["mail_id"]),
                    "alert",
                    "superseded",
                    {"by": mail_id},
                )
            connection.execute(
                """
                INSERT INTO alert_projection(
                    source_id, event_id, mail_id, status, not_before, expires_at,
                    accepted_session, accepted_turn
                ) VALUES (?, ?, ?, 'pending', ?, ?, NULL, NULL)
                ON CONFLICT(source_id, event_id) DO UPDATE SET
                    mail_id=excluded.mail_id, status='pending',
                    not_before=excluded.not_before, expires_at=excluded.expires_at,
                    accepted_session=NULL, accepted_turn=NULL
                """,
                (source, event, mail_id, observed, expiry),
            )
            if inserted:
                self._append_transition(connection, mail_id, "alert", "received", {})
        return {
            "accepted": inserted,
            "source_id": source,
            "event_id": event,
            "mail_id": mail_id,
        }

    def alert_status(self, source_id: str, event_id: str) -> str | None:
        """Read the current Alert projection without changing its envelope."""

        source = _identity("source_id", source_id)
        event = _identity("event_id", event_id)
        with self._transaction(write=False) as connection:
            row = connection.execute(
                "SELECT status FROM alert_projection WHERE source_id=? AND event_id=?",
                (source, event),
            ).fetchone()
        return None if row is None else str(row["status"])

    def alert_deadline(self, now: datetime) -> datetime | None:
        """Expire old alerts and return the earliest pending deadline."""

        instant = _aware_utc(now)
        with self._transaction(write=True) as connection:
            self._expire_alerts(connection, instant)
            row = connection.execute(
                "SELECT MIN(not_before) AS deadline FROM alert_projection "
                "WHERE status='pending'"
            ).fetchone()
        return (
            None
            if row is None or row["deadline"] is None
            else _parse_datetime(str(row["deadline"]))
        )

    def select_alert(
        self, accepted_turn: Mapping[str, object], now: datetime
    ) -> Mapping[str, object] | None:
        """Claim the oldest due Alert for one accepted Turn."""

        accepted = _normalize_accepted_turn(accepted_turn)
        instant = _aware_utc(now)
        with self._transaction(write=True) as connection:
            self._expire_alerts(connection, instant)
            row = connection.execute(
                """
                SELECT projection.source_id, projection.event_id, projection.mail_id,
                       envelope.payload_json, envelope.observed_at,
                       projection.not_before, projection.expires_at
                FROM alert_projection AS projection
                JOIN mail_envelopes AS envelope ON envelope.mail_id=projection.mail_id
                WHERE projection.status='pending' AND projection.not_before <= ?
                ORDER BY projection.not_before, envelope.seq LIMIT 1
                """,
                (instant,),
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                "UPDATE alert_projection SET status='selected', accepted_session=?, "
                "accepted_turn=? WHERE source_id=? AND event_id=? AND status='pending'",
                (
                    accepted["session_id"],
                    accepted["turn_id"],
                    row["source_id"],
                    row["event_id"],
                ),
            )
            if cursor.rowcount != 1:
                return None
            self._append_transition(
                connection,
                str(row["mail_id"]),
                "alert",
                "selected",
                {"accepted_turn": accepted},
            )
            return _alert_view(row, accepted)

    def selected_alert(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        """Read the Alert selected by one accepted Turn."""

        accepted = _normalize_accepted_turn(accepted_turn)
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """
                SELECT projection.source_id, projection.event_id, projection.mail_id,
                       envelope.payload_json, envelope.observed_at,
                       projection.not_before, projection.expires_at
                FROM alert_projection AS projection
                JOIN mail_envelopes AS envelope ON envelope.mail_id=projection.mail_id
                WHERE projection.status='selected' AND accepted_session=?
                  AND accepted_turn=?
                """,
                (accepted["session_id"], accepted["turn_id"]),
            ).fetchone()
        return None if row is None else _alert_view(row, accepted)

    def selected_alerts(self) -> tuple[Mapping[str, object], ...]:
        """List selected Alerts for crash recovery."""

        with self._transaction(write=False) as connection:
            rows = connection.execute("""
                SELECT projection.source_id, projection.event_id, projection.mail_id,
                       envelope.payload_json, envelope.observed_at,
                       projection.not_before, projection.expires_at,
                       projection.accepted_session, projection.accepted_turn
                FROM alert_projection AS projection
                JOIN mail_envelopes AS envelope ON envelope.mail_id=projection.mail_id
                WHERE projection.status='selected' ORDER BY envelope.seq
                """).fetchall()
        return tuple(
            _alert_view(
                row,
                {
                    "session_id": str(row["accepted_session"]),
                    "turn_id": str(row["accepted_turn"]),
                },
            )
            for row in rows
        )

    def expire_alert(self, source_id: str, event_id: str, now: datetime) -> bool:
        """Expire one due Alert before an external provider call starts."""

        source = _identity("source_id", source_id)
        event = _identity("event_id", event_id)
        instant = _aware_utc(now)
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT mail_id, expires_at, status FROM alert_projection "
                "WHERE source_id=? AND event_id=?",
                (source, event),
            ).fetchone()
            if (
                row is None
                or row["expires_at"] is None
                or str(row["expires_at"]) > instant
                or str(row["status"]) not in {"pending", "selected"}
            ):
                return False
            connection.execute(
                "UPDATE alert_projection SET status='expired', accepted_session=NULL, "
                "accepted_turn=NULL WHERE source_id=? AND event_id=?",
                (source, event),
            )
            self._append_transition(
                connection, str(row["mail_id"]), "alert", "expired", {}
            )
            return True

    def defer_alert(self, source_id: str, event_id: str, not_before: datetime) -> None:
        self._transition_alert(
            source_id, event_id, "pending", not_before=_aware_utc(not_before)
        )

    def close_alert(self, source_id: str, event_id: str, status: str) -> None:
        if status not in {"delivered", "skipped"}:
            raise ValueError("Alert close status 必须是 delivered 或 skipped")
        self._transition_alert(source_id, event_id, status)

    def _transition_alert(
        self,
        source_id: str,
        event_id: str,
        status: str,
        *,
        not_before: str | None = None,
    ) -> None:
        source = _identity("source_id", source_id)
        event = _identity("event_id", event_id)
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT mail_id FROM alert_projection WHERE source_id=? AND event_id=? "
                "AND status='selected'",
                (source, event),
            ).fetchone()
            if row is None:
                raise RuntimeError("EventMail Alert transition 未命中 selected row")
            connection.execute(
                "UPDATE alert_projection SET status=?, not_before=COALESCE(?, not_before), "
                "accepted_session=NULL, accepted_turn=NULL WHERE source_id=? AND event_id=?",
                (status, not_before, source, event),
            )
            self._append_transition(
                connection,
                str(row["mail_id"]),
                "alert",
                status if status != "pending" else "deferred",
                {} if not_before is None else {"not_before": not_before},
            )

    def report_context(
        self,
        *,
        source_id: str,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None,
    ) -> Mapping[str, object]:
        """Append one Context revision and replace only its current projection."""

        source = _identity("source_id", source_id)
        event = _identity("event_id", event_id)
        observed = _aware_utc(observed_at)
        expiry = None if expires_at is None else _aware_utc(expires_at)
        if expiry is not None and expiry <= observed:
            raise ValueError("Context expires_at 必须晚于 observed_at")
        with self._transaction(write=True) as connection:
            mail_id, inserted = self._append_envelope(
                connection,
                kind="context",
                source_id=source,
                item_id=event,
                revision=observed,
                payload_json=_payload_json(payload),
                observed_at=observed,
                not_before=None,
                expires_at=expiry,
            )
            current = connection.execute(
                """
                SELECT projection.mail_id, envelope.observed_at
                FROM context_projection AS projection
                JOIN mail_envelopes AS envelope ON envelope.mail_id=projection.mail_id
                WHERE projection.source_id=? AND projection.event_id=?
                """,
                (source, event),
            ).fetchone()
            if current is not None and str(current["mail_id"]) == mail_id:
                return {
                    "accepted": False,
                    "source_id": source,
                    "event_id": event,
                    "mail_id": mail_id,
                }
            if current is not None and str(current["observed_at"]) > observed:
                if inserted:
                    self._append_transition(
                        connection, mail_id, "context", "received", {}
                    )
                    self._append_transition(
                        connection,
                        mail_id,
                        "context",
                        "superseded",
                        {"by": str(current["mail_id"])},
                    )
                return {
                    "accepted": inserted,
                    "projected": False,
                    "source_id": source,
                    "event_id": event,
                    "mail_id": mail_id,
                }
            if current is not None:
                self._append_transition(
                    connection,
                    str(current["mail_id"]),
                    "context",
                    "superseded",
                    {"by": mail_id},
                )
            connection.execute(
                """
                INSERT INTO context_projection(source_id, event_id, mail_id, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id, event_id) DO UPDATE SET
                    mail_id=excluded.mail_id, expires_at=excluded.expires_at
                """,
                (source, event, mail_id, expiry),
            )
            if inserted:
                self._append_transition(connection, mail_id, "context", "received", {})
        return {
            "accepted": inserted,
            "source_id": source,
            "event_id": event,
            "mail_id": mail_id,
        }

    def active_context(self, now: datetime) -> tuple[Mapping[str, object], ...]:
        """Read non-expired current Context without consuming it."""

        instant = _aware_utc(now)
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT projection.source_id, projection.event_id, projection.mail_id,
                       envelope.payload_json, envelope.observed_at, projection.expires_at
                FROM context_projection AS projection
                JOIN mail_envelopes AS envelope ON envelope.mail_id=projection.mail_id
                WHERE projection.expires_at IS NULL OR projection.expires_at > ?
                ORDER BY envelope.observed_at DESC, envelope.seq DESC
                """,
                (instant,),
            ).fetchall()
        return tuple(
            {
                "source_id": str(row["source_id"]),
                "event_id": str(row["event_id"]),
                "mail_id": str(row["mail_id"]),
                "payload": _decode_payload(row["payload_json"]),
                "observed_at": str(row["observed_at"]),
                "expires_at": (
                    None if row["expires_at"] is None else str(row["expires_at"])
                ),
            }
            for row in rows
        )

    def mail_watermark(self) -> int:
        """Return the immutable envelope high-watermark for Wake attempts."""

        with self._transaction(write=False) as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM mail_envelopes"
            ).fetchone()
        assert row is not None
        return int(row["seq"])

    def rebuild_mail_projections(self) -> None:
        """Rebuild every mutable mail projection from immutable mail history."""

        with self._transaction(write=True) as connection:
            self._rebuild_content_projections(connection)
            alerts = self._latest_envelopes(connection, "alert")
            contexts = self._latest_envelopes(connection, "context")
            connection.execute("DELETE FROM alert_projection")
            connection.execute("DELETE FROM context_projection")
            for envelope in alerts:
                status = "pending"
                not_before = str(envelope["not_before"])
                accepted_session: str | None = None
                accepted_turn: str | None = None
                transitions = connection.execute(
                    "SELECT action, detail_json FROM mail_transitions "
                    "WHERE mail_id=? ORDER BY seq",
                    (envelope["mail_id"],),
                ).fetchall()
                for transition in transitions:
                    action = str(transition["action"])
                    detail = json.loads(str(transition["detail_json"]))
                    if not isinstance(detail, dict):
                        raise RuntimeError("EventMail transition detail 必须是 object")
                    if action == "selected":
                        accepted = _normalize_accepted_turn(detail["accepted_turn"])
                        status = "selected"
                        accepted_session = accepted["session_id"]
                        accepted_turn = accepted["turn_id"]
                    elif action == "deferred":
                        status = "pending"
                        accepted_session = None
                        accepted_turn = None
                        not_before = _identity("not_before", detail["not_before"])
                    elif action in {"delivered", "skipped", "expired"}:
                        status = action
                        accepted_session = None
                        accepted_turn = None
                connection.execute(
                    """
                    INSERT INTO alert_projection(
                        source_id, event_id, mail_id, status, not_before, expires_at,
                        accepted_session, accepted_turn
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        envelope["source_id"],
                        envelope["item_id"],
                        envelope["mail_id"],
                        status,
                        not_before,
                        envelope["expires_at"],
                        accepted_session,
                        accepted_turn,
                    ),
                )
            for envelope in contexts:
                connection.execute(
                    "INSERT INTO context_projection(source_id, event_id, mail_id, expires_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        envelope["source_id"],
                        envelope["item_id"],
                        envelope["mail_id"],
                        envelope["expires_at"],
                    ),
                )

    @classmethod
    def _append_content_projections(
        cls, connection: sqlite3.Connection, mail_ids: Iterable[str]
    ) -> None:
        """Append complete Content query facts after one committed state change."""

        state = dict(cls._state(connection))
        for mail_id in dict.fromkeys(mail_ids):
            envelope = connection.execute(
                "SELECT source_id, item_id, revision FROM mail_envelopes "
                "WHERE mail_id=? AND kind='content'",
                (mail_id,),
            ).fetchone()
            if envelope is None:
                raise RuntimeError(f"Content projection 缺少 envelope: {mail_id}")
            item = connection.execute(
                "SELECT * FROM items WHERE source_id=? AND item_id=? AND revision=?",
                (envelope["source_id"], envelope["item_id"], envelope["revision"]),
            ).fetchone()
            if item is None:
                raise RuntimeError(f"Content projection 缺少 item: {mail_id}")
            selection_rows = connection.execute(
                "SELECT selection.* FROM content_selection_members AS member "
                "JOIN content_selections AS selection USING(selection_token) "
                "WHERE member.source_id=? AND member.item_id=? AND member.revision=? "
                "ORDER BY selection.created_at, selection.selection_token",
                (envelope["source_id"], envelope["item_id"], envelope["revision"]),
            ).fetchall()
            selections = []
            for selection in selection_rows:
                members = connection.execute(
                    "SELECT * FROM content_selection_members WHERE selection_token=? "
                    "ORDER BY position",
                    (selection["selection_token"],),
                ).fetchall()
                selections.append(
                    {
                        "selection": dict(selection),
                        "members": [dict(member) for member in members],
                    }
                )
            cls._append_transition(
                connection,
                mail_id,
                "content",
                "projected",
                {
                    "item": dict(item),
                    "content_state": state,
                    "selections": selections,
                },
            )

    @staticmethod
    def _rebuild_content_projections(connection: sqlite3.Connection) -> None:
        """Replace Content query tables with the latest immutable projection facts."""

        rows = connection.execute(
            "SELECT seq, detail_json FROM mail_transitions "
            "WHERE kind='content' AND action='projected' ORDER BY seq"
        ).fetchall()
        content_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM mail_envelopes WHERE kind='content'"
            ).fetchone()[0]
        )
        if content_count and not rows:
            raise RuntimeError("Content ledger 缺少 projected transition")
        items: dict[tuple[str, str, str], dict[str, object]] = {}
        selections: dict[str, dict[str, object]] = {}
        state: dict[str, object] | None = None
        for row in rows:
            detail = json.loads(str(row["detail_json"]))
            if not isinstance(detail, dict):
                raise RuntimeError("Content projected detail 必须是 object")
            item = detail.get("item")
            projected_state = detail.get("content_state")
            projected_selections = detail.get("selections")
            if (
                not isinstance(item, dict)
                or not isinstance(projected_state, dict)
                or not isinstance(projected_selections, list)
            ):
                raise RuntimeError("Content projected detail 结构无效")
            key = (
                _identity("source_id", item.get("source_id")),
                _identity("item_id", item.get("item_id")),
                _identity("revision", item.get("revision")),
            )
            items[key] = item
            state = projected_state
            for projected in projected_selections:
                if not isinstance(projected, dict):
                    raise RuntimeError("Content projected selection 必须是 object")
                selection = projected.get("selection")
                members = projected.get("members")
                if not isinstance(selection, dict) or not isinstance(members, list):
                    raise RuntimeError("Content projected selection 结构无效")
                token = _identity("selection_token", selection.get("selection_token"))
                selections[token] = {"selection": selection, "members": members}
        if len(items) != content_count:
            raise RuntimeError(
                "Content projected item 数量与 envelope 不一致: "
                f"items={len(items)} envelopes={content_count}"
            )
        connection.execute("DELETE FROM content_selection_members")
        connection.execute("DELETE FROM content_selections")
        connection.execute("DELETE FROM items")
        for item in sorted(
            items.values(),
            key=_projected_snapshot_seq,
        ):
            columns = tuple(item)
            connection.execute(
                f"INSERT INTO items({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                tuple(item[column] for column in columns),
            )
        for projected in selections.values():
            selection = cast(dict[str, object], projected["selection"])
            columns = tuple(selection)
            connection.execute(
                f"INSERT INTO content_selections({','.join(columns)}) "
                f"VALUES({','.join('?' for _ in columns)})",
                tuple(selection[column] for column in columns),
            )
            for member in cast(list[dict[str, object]], projected["members"]):
                member_columns = tuple(member)
                connection.execute(
                    f"INSERT INTO content_selection_members({','.join(member_columns)}) "
                    f"VALUES({','.join('?' for _ in member_columns)})",
                    tuple(member[column] for column in member_columns),
                )
        if state is not None:
            connection.execute("DELETE FROM content_state")
            columns = tuple(state)
            connection.execute(
                f"INSERT INTO content_state({','.join(columns)}) "
                f"VALUES({','.join('?' for _ in columns)})",
                tuple(state[column] for column in columns),
            )

    @staticmethod
    def _latest_envelopes(
        connection: sqlite3.Connection,
        kind: Literal["alert", "context"],
    ) -> tuple[sqlite3.Row, ...]:
        rows = connection.execute(
            """
            SELECT envelope.* FROM mail_envelopes AS envelope
            WHERE envelope.kind=? AND NOT EXISTS (
                SELECT 1 FROM mail_envelopes AS newer
                WHERE newer.kind=envelope.kind
                  AND newer.source_id=envelope.source_id
                  AND newer.item_id=envelope.item_id
                  AND (
                    newer.observed_at > envelope.observed_at
                    OR (newer.observed_at=envelope.observed_at AND newer.seq > envelope.seq)
                  )
            )
            ORDER BY envelope.seq
            """,
            (kind,),
        ).fetchall()
        return tuple(rows)

    @staticmethod
    def _append_envelope(
        connection: sqlite3.Connection,
        *,
        kind: Literal["content", "alert", "context"],
        source_id: str,
        item_id: str,
        revision: str,
        payload_json: str,
        observed_at: str,
        not_before: str | None,
        expires_at: str | None,
    ) -> tuple[str, bool]:
        """Append an immutable envelope or verify an exact replay."""

        mail_id = _mail_id(kind, source_id, item_id, revision)
        row = connection.execute(
            "SELECT payload_json, observed_at, not_before, expires_at FROM mail_envelopes "
            "WHERE mail_id=?",
            (mail_id,),
        ).fetchone()
        if row is not None:
            actual = (
                str(row["payload_json"]),
                str(row["observed_at"]),
                None if row["not_before"] is None else str(row["not_before"]),
                None if row["expires_at"] is None else str(row["expires_at"]),
            )
            expected = (payload_json, observed_at, not_before, expires_at)
            if actual != expected:
                raise EventMailIdentityConflict(
                    f"EventMail envelope identity conflict: {mail_id}"
                )
            return mail_id, False
        connection.execute(
            """
            INSERT INTO mail_envelopes(
                mail_id, kind, source_id, item_id, revision, payload_json,
                observed_at, not_before, expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mail_id,
                kind,
                source_id,
                item_id,
                revision,
                payload_json,
                observed_at,
                not_before,
                expires_at,
                _utc_now(),
            ),
        )
        return mail_id, True

    @staticmethod
    def _append_transition(
        connection: sqlite3.Connection,
        mail_id: str,
        kind: Literal["content", "alert", "context"],
        action: str,
        detail: Mapping[str, object],
    ) -> None:
        """Append one idempotent transition without rewriting its envelope."""

        encoded = json.dumps(
            dict(detail), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
        transition_id = f"{mail_id}:{action}:{fingerprint}"
        connection.execute(
            "INSERT OR IGNORE INTO mail_transitions("
            "transition_id, mail_id, kind, action, detail_json, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (transition_id, mail_id, kind, action, encoded, _utc_now()),
        )

    @classmethod
    def _expire_alerts(cls, connection: sqlite3.Connection, now: str) -> int:
        rows = connection.execute(
            "SELECT mail_id, source_id, event_id FROM alert_projection "
            "WHERE status IN ('pending', 'selected') AND expires_at IS NOT NULL "
            "AND expires_at <= ? ORDER BY source_id, event_id",
            (now,),
        ).fetchall()
        for row in rows:
            cls._append_transition(
                connection, str(row["mail_id"]), "alert", "expired", {}
            )
        if rows:
            connection.execute(
                "UPDATE alert_projection SET status='expired', accepted_session=NULL, "
                "accepted_turn=NULL WHERE status IN ('pending', 'selected') "
                "AND expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
        return len(rows)

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
                  AND NOT EXISTS (
                    SELECT 1 FROM content_selection_members AS member
                    JOIN content_selections AS selection
                      ON selection.selection_token = member.selection_token
                    WHERE member.source_id = items.source_id
                      AND member.item_id = items.item_id
                      AND member.revision = items.revision
                      AND (
                        selection.status = 'selected'
                        OR (
                          selection.status = 'ready_for_delivery'
                          AND member.selected_for_delivery = 1
                        )
                      )
                  )
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
        """Recover one durable Content batch from its accepted Turn receipt."""

        accepted = _normalize_accepted_turn(accepted_turn)
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """
                SELECT * FROM content_selections
                WHERE accepted_session_id = ? AND accepted_turn_id = ?
                """,
                (accepted["session_id"], accepted["turn_id"]),
            ).fetchone()
            return None if row is None else self._selection_receipt(connection, row)

    def selected(self, limit: int = 100) -> tuple[ContentSelectionReceipt, ...]:
        """Return selected batches in stable inbox order for external recovery."""

        if type(limit) is not int or limit <= 0:
            raise ValueError("limit 必须是正整数")
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT * FROM content_selections
                WHERE status = 'selected'
                ORDER BY snapshot_seq
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return tuple(self._selection_receipt(connection, row) for row in rows)

    def select(
        self,
        item_ref: Mapping[str, object],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> ContentSelectResult:
        """CAS one frozen eligible revision into a one-item batch."""

        return self.select_batch((item_ref,), snapshot_seq, accepted_turn, now)

    def select_batch(
        self,
        item_refs: Sequence[Mapping[str, object]],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> ContentSelectResult:
        """CAS one frozen candidate page into a Turn-bound durable batch."""

        refs = tuple(_normalize_ref(item_ref) for item_ref in item_refs)
        if not refs or len(refs) > 100:
            raise ValueError("Content batch selection 必须包含 1..100 个候选")
        identities = {
            (ref["source_id"], ref["item_id"], ref["revision"]) for ref in refs
        }
        if len(identities) != len(refs):
            raise ValueError("Content batch selection 不允许重复候选")
        if type(snapshot_seq) is not int or snapshot_seq < 0:
            raise ValueError("snapshot_seq 必须是非负整数")
        accepted = _normalize_accepted_turn(accepted_turn)
        instant = _aware_utc(now)
        with self._transaction(write=True) as connection:
            existing = connection.execute(
                """
                SELECT selection_token FROM content_selections
                WHERE accepted_session_id = ? AND accepted_turn_id = ?
                """,
                (accepted["session_id"], accepted["turn_id"]),
            ).fetchone()
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
            candidates: list[sqlite3.Row] = []
            for ref in refs:
                row = connection.execute(
                    """
                    SELECT * FROM items
                    WHERE source_id = ? AND item_id = ? AND revision = ?
                      AND snapshot_seq <= ? AND item_state_version = ?
                      AND status IN ('pending', 'deferred') AND not_before <= ?
                      AND NOT EXISTS (
                        SELECT 1 FROM content_selection_members AS member
                        JOIN content_selections AS selection
                          ON selection.selection_token = member.selection_token
                        WHERE member.source_id = items.source_id
                          AND member.item_id = items.item_id
                          AND member.revision = items.revision
                          AND (
                            selection.status = 'selected'
                            OR (
                              selection.status = 'ready_for_delivery'
                              AND member.selected_for_delivery = 1
                            )
                          )
                      )
                    """,
                    (
                        ref["source_id"],
                        ref["item_id"],
                        ref["revision"],
                        snapshot_seq,
                        ref["state_version"],
                        instant,
                    ),
                ).fetchone()
                if row is None:
                    state = self._state(connection)
                    return {
                        "selected": False,
                        "reason": "batch_candidate_changed",
                        "selection_token": None,
                        "accepted_turn": None,
                        "state_version": int(state["state_version"]),
                        "wake_needed": bool(state["wake_needed"]),
                        "earliest_not_before": state["earliest_not_before"],
                    }
                candidates.append(row)
            driver = candidates[0]
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
                    driver["source_id"],
                    driver["item_id"],
                    driver["revision"],
                    snapshot_seq,
                    driver["item_state_version"],
                    instant,
                ),
            )
            selected = cursor.rowcount == 1
            if selected:
                created = _utc_now()
                _ = connection.execute(
                    """
                    INSERT INTO content_selections(
                        selection_token, accepted_session_id, accepted_turn_id,
                        snapshot_seq, status, driver_source_id, driver_item_id,
                        driver_revision, decision_format, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'selected', ?, ?, ?, 'items_v1', ?, ?)
                    """,
                    (
                        token,
                        accepted["session_id"],
                        accepted["turn_id"],
                        snapshot_seq,
                        driver["source_id"],
                        driver["item_id"],
                        driver["revision"],
                        created,
                        created,
                    ),
                )
                for position, row in enumerate(candidates, start=1):
                    _ = connection.execute(
                        """
                        INSERT INTO content_selection_members(
                            selection_token, position, source_id, item_id, revision
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            token,
                            position,
                            row["source_id"],
                            row["item_id"],
                            row["revision"],
                        ),
                    )
                    self._append_transition(
                        connection,
                        _mail_id(
                            "content",
                            str(row["source_id"]),
                            str(row["item_id"]),
                            str(row["revision"]),
                        ),
                        "content",
                        "selected",
                        {"selection_token": token, "accepted_turn": accepted},
                    )
                _ = connection.execute(
                    "UPDATE content_state SET state_version = state_version + 1 WHERE singleton = 1"
                )
            self._recompute_wake(connection)
            if selected:
                self._append_content_projections(
                    connection,
                    (
                        _mail_id(
                            "content",
                            str(row["source_id"]),
                            str(row["item_id"]),
                            str(row["revision"]),
                        )
                        for row in candidates
                    ),
                )
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
    def _selection_receipt(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> ContentSelectionReceipt:
        members = connection.execute(
            """
            SELECT i.* FROM content_selection_members AS m
            JOIN items AS i USING(source_id, item_id, revision)
            WHERE m.selection_token = ? ORDER BY m.position
            """,
            (row["selection_token"],),
        ).fetchall()
        if not members:
            raise RuntimeError("Content selection batch 缺少 members")
        driver = next(
            (
                member
                for member in members
                if member["source_id"] == row["driver_source_id"]
                and member["item_id"] == row["driver_item_id"]
                and member["revision"] == row["driver_revision"]
            ),
            None,
        )
        if driver is None:
            raise RuntimeError("Content selection batch 缺少 driver")
        return {
            "selection_token": row["selection_token"],
            "ref": {
                "source_id": driver["source_id"],
                "item_id": driver["item_id"],
                "revision": driver["revision"],
                "state_version": int(driver["item_state_version"]),
            },
            "payload": json.loads(driver["payload_json"]),
            "snapshot_seq": int(row["snapshot_seq"]),
            "status": row["status"],
            "not_before": driver["not_before"],
            "requires_ack": bool(driver["requires_ack"]),
            "accepted_turn": {
                "session_id": row["accepted_session_id"],
                "turn_id": row["accepted_turn_id"],
            },
            "items": tuple(_snapshot_item(member) for member in members),
            "decision_format": row["decision_format"],
        }

    def transition(
        self,
        selection_token: str,
        action: str,
        *,
        not_before: datetime | None = None,
        settlement_ref: str | None = None,
        selected_refs: Sequence[Mapping[str, object]] | None = None,
    ) -> ContentTransitionResult:
        """Commit one explicit domain transition without inferring Turn state."""

        token = _identity("selection_token", selection_token)
        if action not in {
            "ready_for_delivery",
            "release",
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
        if action != "ready_for_delivery" and selected_refs is not None:
            raise ValueError("只有 ready_for_delivery 可以提供 selected_refs")
        deadline = _aware_utc(not_before) if not_before is not None else None
        settlement = (
            _identity("settlement_ref", settlement_ref)
            if settlement_ref is not None
            else None
        )
        if action == "delivered":
            assert settlement is not None
            return cast(
                ContentTransitionResult,
                self.settle_delivery(token, settlement),
            )

        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM content_selections WHERE selection_token = ?",
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
            result_status = status
            updated = _utc_now()
            transition_rows = connection.execute(
                "SELECT source_id, item_id, revision FROM content_selection_members "
                "WHERE selection_token=? ORDER BY position",
                (token,),
            ).fetchall()
            if action == "ready_for_delivery":
                chosen = self._select_delivery_members(
                    connection,
                    token,
                    selected_refs,
                    driver=(
                        row["driver_source_id"],
                        row["driver_item_id"],
                        row["driver_revision"],
                    ),
                )
                driver_chosen = (
                    row["driver_source_id"],
                    row["driver_item_id"],
                    row["driver_revision"],
                ) in chosen
                if driver_chosen:
                    _ = connection.execute(
                        """
                        UPDATE items SET status = 'ready_for_delivery',
                            item_state_version = item_state_version + 1, updated_at = ?
                        WHERE selection_token = ? AND status = 'selected'
                        """,
                        (updated, token),
                    )
                else:
                    self._release_driver(connection, token, updated)
            elif action == "release":
                status = "released"
                result_status = "pending"
                self._release_driver(connection, token, updated)
            else:
                _ = connection.execute(
                    """
                    UPDATE items SET status = ?, not_before = COALESCE(?, not_before),
                        settlement_ref = COALESCE(?, settlement_ref),
                        item_state_version = item_state_version + 1, updated_at = ?
                    WHERE selection_token = ?
                    """,
                    (status, deadline, settlement, updated, token),
                )
            _ = connection.execute(
                """
                UPDATE content_selections
                SET status = ?, settlement_ref = COALESCE(?, settlement_ref),
                    updated_at = ? WHERE selection_token = ?
                """,
                (status, settlement, updated, token),
            )
            for member in transition_rows:
                self._append_transition(
                    connection,
                    _mail_id(
                        "content",
                        str(member["source_id"]),
                        str(member["item_id"]),
                        str(member["revision"]),
                    ),
                    "content",
                    action,
                    {
                        "selection_token": token,
                        **({} if deadline is None else {"not_before": deadline}),
                        **(
                            {} if settlement is None else {"settlement_ref": settlement}
                        ),
                    },
                )
            _ = connection.execute(
                "UPDATE content_state SET state_version = state_version + 1 WHERE singleton = 1"
            )
            self._recompute_wake(connection)
            self._append_content_projections(
                connection,
                (
                    _mail_id(
                        "content",
                        str(member["source_id"]),
                        str(member["item_id"]),
                        str(member["revision"]),
                    )
                    for member in transition_rows
                ),
            )
            state = self._state(connection)
            return {
                "changed": True,
                "status": result_status,
                "state_version": int(state["state_version"]),
                "wake_needed": bool(state["wake_needed"]),
                "earliest_not_before": state["earliest_not_before"],
            }

    @staticmethod
    def _select_delivery_members(
        connection: sqlite3.Connection,
        token: str,
        selected_refs: Sequence[Mapping[str, object]] | None,
        *,
        driver: tuple[str, str, str],
    ) -> set[tuple[str, str, str]]:
        """Validate and persist the one-to-five members represented by one message."""

        raw_refs = (
            ({"source_id": driver[0], "item_id": driver[1], "revision": driver[2]},)
            if selected_refs is None
            else selected_refs
        )
        chosen = {
            (
                _identity("source_id", ref.get("source_id")),
                _identity("item_id", ref.get("item_id")),
                _identity("revision", ref.get("revision")),
            )
            for ref in raw_refs
        }
        if not chosen or len(chosen) > 5 or len(chosen) != len(raw_refs):
            raise ValueError("Content share 必须引用 1..5 个不重复候选")
        available = {
            (str(row["source_id"]), str(row["item_id"]), str(row["revision"]))
            for row in connection.execute(
                """
                SELECT source_id, item_id, revision
                FROM content_selection_members WHERE selection_token = ?
                """,
                (token,),
            )
        }
        if not chosen <= available:
            raise ValueError("Content share 引用了批次外候选")
        for source_id, item_id, revision in chosen:
            _ = connection.execute(
                """
                UPDATE content_selection_members SET selected_for_delivery = 1
                WHERE selection_token = ? AND source_id = ? AND item_id = ? AND revision = ?
                """,
                (token, source_id, item_id, revision),
            )
        return chosen

    @staticmethod
    def _release_driver(
        connection: sqlite3.Connection, token: str, updated_at: str
    ) -> None:
        _ = connection.execute(
            """
            UPDATE items SET status = 'pending', selection_token = NULL,
                selected_session_id = NULL, selected_turn_id = NULL,
                item_state_version = item_state_version + 1, updated_at = ?
            WHERE selection_token = ? AND status IN ('selected', 'ready_for_delivery')
            """,
            (updated_at, token),
        )

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
                SELECT * FROM content_selections
                WHERE status = 'ready_for_delivery'
                ORDER BY snapshot_seq
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            pending: list[dict[str, object]] = []
            for row in rows:
                members = self._delivery_member_rows(connection, row["selection_token"])
                pending.append(
                    {
                        "selection_token": _identity(
                            "selection_token", row["selection_token"]
                        ),
                        "accepted_turn": {
                            "session_id": _identity(
                                "accepted_session_id", row["accepted_session_id"]
                            ),
                            "turn_id": _identity(
                                "accepted_turn_id", row["accepted_turn_id"]
                            ),
                        },
                        "message_metadata": _message_metadata_many(members),
                        "decision_format": row["decision_format"],
                    }
                )
            return tuple(pending)

    def delivery(self, accepted_turn: Mapping[str, object]) -> dict[str, object] | None:
        """Read a body-free delivery receipt by its accepted Turn identity."""

        accepted = _normalize_accepted_turn(accepted_turn)
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """
                SELECT * FROM content_selections
                WHERE accepted_session_id = ? AND accepted_turn_id = ?
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
                member_statuses = {
                    str(member["status"])
                    for member in self._delivery_member_rows(
                        connection, row["selection_token"]
                    )
                }
                if member_statuses == {"settled"}:
                    result["status"] = "settled"
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
                "SELECT * FROM content_selections WHERE selection_token = ?",
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
            members = self._delivery_member_rows(connection, token)
            if not members:
                raise RuntimeError("Content ready batch 缺少 delivery members")
            updated = _utc_now()
            statuses: set[str] = set()
            for member in members:
                member_settlement = (
                    settlement
                    if len(members) == 1
                    else _member_settlement(settlement, member)
                )
                status = "delivered" if bool(member["requires_ack"]) else "settled"
                statuses.add(status)
                cursor = connection.execute(
                    """
                    UPDATE items SET status = ?, settlement_ref = ?,
                        item_state_version = item_state_version + 1, updated_at = ?
                    WHERE source_id = ? AND item_id = ? AND revision = ?
                      AND status IN ('pending', 'deferred', 'selected', 'ready_for_delivery')
                    """,
                    (
                        status,
                        member_settlement,
                        updated,
                        member["source_id"],
                        member["item_id"],
                        member["revision"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "Content delivery member state changed before settlement"
                    )
                self._append_transition(
                    connection,
                    _mail_id(
                        "content",
                        str(member["source_id"]),
                        str(member["item_id"]),
                        str(member["revision"]),
                    ),
                    "content",
                    status,
                    {"settlement_ref": member_settlement},
                )
                _ = connection.execute(
                    """
                    UPDATE content_selection_members SET settlement_ref = ?
                    WHERE selection_token = ? AND source_id = ? AND item_id = ? AND revision = ?
                    """,
                    (
                        member_settlement,
                        token,
                        member["source_id"],
                        member["item_id"],
                        member["revision"],
                    ),
                )
            driver_selected = any(
                member["source_id"] == row["driver_source_id"]
                and member["item_id"] == row["driver_item_id"]
                and member["revision"] == row["driver_revision"]
                for member in members
            )
            if not driver_selected:
                self._release_driver(connection, token, updated)
            status = "delivered" if "delivered" in statuses else "settled"
            _ = connection.execute(
                """
                UPDATE content_selections SET status = ?, settlement_ref = ?, updated_at = ?
                WHERE selection_token = ? AND status = 'ready_for_delivery'
                """,
                (status, settlement, updated, token),
            )
            _ = connection.execute("""
                UPDATE content_state
                SET state_version = state_version + 1 WHERE singleton = 1
                """)
            self._recompute_wake(connection)
            projection_rows = connection.execute(
                "SELECT source_id, item_id, revision FROM content_selection_members "
                "WHERE selection_token=? ORDER BY position",
                (token,),
            ).fetchall()
            self._append_content_projections(
                connection,
                (
                    _mail_id(
                        "content",
                        str(member["source_id"]),
                        str(member["item_id"]),
                        str(member["revision"]),
                    )
                    for member in projection_rows
                ),
            )
            return {
                "settled": True,
                "duplicate": False,
                "status": status,
                "receipt": receipt,
            }

    @staticmethod
    def _delivery_member_rows(
        connection: sqlite3.Connection, selection_token: str
    ) -> tuple[sqlite3.Row, ...]:
        rows = connection.execute(
            """
            SELECT i.* FROM content_selection_members AS m
            JOIN items AS i USING(source_id, item_id, revision)
            WHERE m.selection_token = ? AND m.selected_for_delivery = 1
            ORDER BY m.position
            """,
            (selection_token,),
        ).fetchall()
        return tuple(rows)

    def ack(self, source_id: str, settlement_ref: str) -> dict[str, object]:
        """Settle one delivered row only through its source-bound view."""

        source = _identity("source_id", source_id)
        settlement = _identity("settlement_ref", settlement_ref)
        with self._transaction(write=True) as connection:
            row = connection.execute(
                """
                SELECT status, item_id, revision FROM items
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
            self._append_transition(
                connection,
                _mail_id("content", source, str(row["item_id"]), str(row["revision"])),
                "content",
                "acknowledged",
                {"settlement_ref": settlement},
            )
            _ = connection.execute(
                "UPDATE content_state SET state_version = state_version + 1 WHERE singleton = 1"
            )
            self._append_content_projections(
                connection,
                (
                    _mail_id(
                        "content", source, str(row["item_id"]), str(row["revision"])
                    ),
                ),
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
            raise PermissionError(
                "Content read-only candidate cannot write shared data"
            )

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
        database_uri = self.path.resolve(strict=False).as_uri() + "?mode=ro&immutable=1"
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
        if version not in (0, 1, 2, _SCHEMA_VERSION):
            raise RuntimeError(f"不支持的 EventMail schema version: {version}")
        if version == _SCHEMA_VERSION:
            return
        if version == 1:
            EventMailV3MigrationStore._migrate_v1(connection)
            version = 2
        if version == 2:
            EventMailV3MigrationStore._migrate_v2(connection)
            return
        statements = (
            _TABLE_SQL["mail_envelopes"],
            _TABLE_SQL["mail_transitions"],
            _TABLE_SQL["alert_projection"],
            _TABLE_SQL["context_projection"],
            _TABLE_SQL["content_state"],
            "INSERT INTO content_state VALUES(1, 0, 0, 0, NULL)",
            _TABLE_SQL["items"],
            _TABLE_SQL["submissions"],
            _TABLE_SQL["content_selections"],
            _TABLE_SQL["content_selection_members"],
            *_INDEX_SQL,
            f"PRAGMA user_version = {_SCHEMA_VERSION}",
        )
        _ = connection.executescript(";\n".join(statements) + ";")

    @staticmethod
    def _migrate_v1(connection: sqlite3.Connection) -> None:
        """Add durable batch selections after proving the exact v1 owner schema."""

        # 1. Refuse to reinterpret an unknown database as Content v1.
        tables = {
            str(row["name"]): str(row["sql"])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        legacy_tables = {
            key: _TABLE_SQL[key] for key in ("content_state", "items", "submissions")
        }
        if set(tables) != set(legacy_tables):
            raise RuntimeError("Content v1 schema mismatch: owned tables")
        for table, expected in legacy_tables.items():
            if _normalize_sql(tables[table]) != _normalize_sql(expected):
                raise RuntimeError(f"Content v1 schema mismatch: {table} table SQL")
        legacy_indexes = {
            "content_state": _EXPECTED_INDEXES["content_state"],
            "items": _EXPECTED_INDEXES["items"],
            "submissions": _EXPECTED_INDEXES["submissions"],
        }
        for table, expected in legacy_indexes.items():
            if _schema_indexes(connection, table) != expected:
                raise RuntimeError(f"Content v1 schema mismatch: {table} indexes")

        # 2. Add one selection ledger and preserve every extant single-item selection.
        _ = connection.executescript(
            ";\n".join(
                (
                    _TABLE_SQL["content_selections"],
                    _TABLE_SQL["content_selection_members"],
                    _INDEX_SQL[7],
                    _INDEX_SQL[8],
                )
            )
            + ";"
        )
        rows = connection.execute("""
            SELECT source_id, item_id, revision, snapshot_seq, status,
                   selection_token, selected_session_id, selected_turn_id,
                   settlement_ref, updated_at
            FROM items WHERE selection_token IS NOT NULL
            ORDER BY snapshot_seq
            """).fetchall()
        for row in rows:
            token = _identity("selection_token", row["selection_token"])
            session_id = _identity("selected_session_id", row["selected_session_id"])
            turn_id = _identity("selected_turn_id", row["selected_turn_id"])
            _ = connection.execute(
                """
                INSERT INTO content_selections(
                    selection_token, accepted_session_id, accepted_turn_id,
                    snapshot_seq, status, driver_source_id, driver_item_id,
                    driver_revision, decision_format, settlement_ref,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'legacy_single', ?, ?, ?)
                """,
                (
                    token,
                    session_id,
                    turn_id,
                    int(row["snapshot_seq"]),
                    row["status"],
                    row["source_id"],
                    row["item_id"],
                    row["revision"],
                    row["settlement_ref"],
                    row["updated_at"],
                    row["updated_at"],
                ),
            )
            _ = connection.execute(
                """
                INSERT INTO content_selection_members(
                    selection_token, position, source_id, item_id, revision,
                    selected_for_delivery, settlement_ref
                ) VALUES (?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    row["source_id"],
                    row["item_id"],
                    row["revision"],
                    int(
                        row["status"] in {"ready_for_delivery", "delivered", "settled"}
                    ),
                    row["settlement_ref"],
                ),
            )
        connection.execute("PRAGMA user_version = 2")

    @staticmethod
    def _migrate_v2(connection: sqlite3.Connection) -> None:
        """Add immutable mail ledgers and backfill every existing Content revision."""

        legacy_names = {
            "content_state",
            "items",
            "submissions",
            "content_selections",
            "content_selection_members",
        }
        tables = {
            str(row["name"]): str(row["sql"])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if set(tables) != legacy_names:
            raise RuntimeError("EventMail v2 schema mismatch: owned tables")
        for table in legacy_names:
            if _normalize_sql(tables[table]) != _normalize_sql(_TABLE_SQL[table]):
                raise RuntimeError(f"EventMail v2 schema mismatch: {table} table SQL")
            if _schema_indexes(connection, table) != _EXPECTED_INDEXES[table]:
                raise RuntimeError(f"EventMail v2 schema mismatch: {table} indexes")

        connection.executescript(
            ";\n".join(
                (
                    _TABLE_SQL["mail_envelopes"],
                    _TABLE_SQL["mail_transitions"],
                    _TABLE_SQL["alert_projection"],
                    _TABLE_SQL["context_projection"],
                    *_INDEX_SQL[:4],
                )
            )
            + ";"
        )
        rows = connection.execute("""
            SELECT source_id, item_id, revision, payload_json, status,
                   not_before, created_at, updated_at, settlement_ref
            FROM items ORDER BY snapshot_seq
            """).fetchall()
        for row in rows:
            mail_id, inserted = EventMailV3MigrationStore._append_envelope(
                connection,
                kind="content",
                source_id=str(row["source_id"]),
                item_id=str(row["item_id"]),
                revision=str(row["revision"]),
                payload_json=str(row["payload_json"]),
                observed_at=str(row["created_at"]),
                not_before=str(row["not_before"]),
                expires_at=None,
            )
            if not inserted:
                raise RuntimeError("EventMail v2 backfill envelope 重复")
            EventMailV3MigrationStore._append_transition(
                connection, mail_id, "content", "received", {"migration": "v2"}
            )
            status = str(row["status"])
            if status != "pending":
                detail = {"migration": "v2", "status": status}
                if row["settlement_ref"] is not None:
                    detail["settlement_ref"] = str(row["settlement_ref"])
                EventMailV3MigrationStore._append_transition(
                    connection, mail_id, "content", status, detail
                )
        EventMailV3MigrationStore._append_content_projections(
            connection,
            (
                _mail_id(
                    "content",
                    str(row["source_id"]),
                    str(row["item_id"]),
                    str(row["revision"]),
                )
                for row in rows
            ),
        )
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        """Reject every version-one database that is not this exact schema."""

        # 1. Exact identity includes the schema version, not only matching tables.
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != _SCHEMA_VERSION:
            raise RuntimeError(f"不支持的 EventMail schema version: {version}")

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
                f"EventMail schema mismatch: tables expected={sorted(_TABLE_SQL)} "
                f"actual={sorted(tables)}"
            )
        for table, expected_sql in _TABLE_SQL.items():
            if _normalize_sql(tables[table]) != _normalize_sql(expected_sql):
                raise RuntimeError(f"EventMail schema mismatch: {table} table SQL")

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
                raise RuntimeError(f"EventMail schema mismatch: {table} columns")
            actual_indexes = _schema_indexes(connection, table)
            if actual_indexes != _EXPECTED_INDEXES[table]:
                raise RuntimeError(f"EventMail schema mismatch: {table} indexes")

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


def _projected_snapshot_seq(item: Mapping[str, object]) -> int:
    value = item.get("snapshot_seq")
    if type(value) is not int or value <= 0:
        raise RuntimeError("Content projected snapshot_seq 无效")
    return value


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


def _message_metadata(row: sqlite3.Row) -> dict[str, object]:
    """Project stable source evidence without exposing the candidate body."""

    return _message_metadata_many((row,))


def _message_metadata_many(rows: Sequence[sqlite3.Row]) -> dict[str, object]:
    """Project one ordered evidence list for an aggregated proactive message."""

    evidence: list[str] = []
    source_refs: list[dict[str, object]] = []
    for display_index, row in enumerate(rows, start=1):
        payload = json.loads(row["payload_json"])
        if not isinstance(payload, dict):
            raise ValueError("Content payload_json 必须解码为 object")
        event_id = f"{row['source_id']}:{row['item_id']}:{row['revision']}"
        evidence.append(event_id)
        source_refs.append(
            {
                "display_index": display_index,
                "event_id": event_id,
                **{
                    field: payload[field]
                    for field in ("source_name", "title", "url")
                    if isinstance(payload.get(field), str) and payload[field].strip()
                },
            }
        )
    return {
        "tools_used": ["message_push"],
        "evidence_item_ids": evidence,
        "source_refs": source_refs,
        "state_summary_tag": "none",
    }


def _snapshot_item(row: sqlite3.Row) -> ContentSnapshotItem:
    payload = json.loads(row["payload_json"])
    if not isinstance(payload, dict):
        raise ValueError("Content payload_json 必须解码为 object")
    return {
        "ref": {
            "source_id": row["source_id"],
            "item_id": row["item_id"],
            "revision": row["revision"],
            "state_version": int(row["item_state_version"]),
        },
        "payload": payload,
        "snapshot_seq": int(row["snapshot_seq"]),
        "status": row["status"],
        "not_before": row["not_before"],
        "due": True,
    }


def _member_settlement(settlement_ref: str, row: sqlite3.Row) -> str:
    identity = f"{row['source_id']}\x00{row['item_id']}\x00{row['revision']}".encode(
        "utf-8"
    )
    return settlement_ref + ":" + hashlib.sha256(identity).hexdigest()[:16]


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


def _payload_json(payload: Mapping[str, object]) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError("EventMail payload 必须是 Mapping")
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _mail_id(
    kind: Literal["content", "alert", "context"],
    source_id: str,
    item_id: str,
    revision: str,
) -> str:
    return f"{kind}:{source_id}:{item_id}:{revision}"


def _decode_payload(value: object) -> dict[str, object]:
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise RuntimeError("EventMail payload_json 必须解码为 object")
    return cast(dict[str, object], decoded)


def _alert_view(row: sqlite3.Row, accepted_turn: AcceptedTurn) -> Mapping[str, object]:
    return {
        "source_id": str(row["source_id"]),
        "event_id": str(row["event_id"]),
        "mail_id": str(row["mail_id"]),
        "payload": _decode_payload(row["payload_json"]),
        "observed_at": str(row["observed_at"]),
        "not_before": str(row["not_before"]),
        "expires_at": None if row["expires_at"] is None else str(row["expires_at"]),
        "accepted_turn": dict(accepted_turn),
    }


def _assert_same_revision(
    source_id: str,
    item: _NormalizedItem,
    existing: sqlite3.Row,
) -> None:
    same = (
        existing["payload_json"] == item["payload_json"]
        and (
            int(existing["item_state_version"]) > 1
            or existing["not_before"] == item["not_before"]
        )
        and bool(existing["requires_ack"]) is item["requires_ack"]
    )
    if not same:
        raise EventMailIdentityConflict(
            "Content revision identity conflict: "
            f"{source_id}/{item['item_id']}/{item['revision']}"
        )

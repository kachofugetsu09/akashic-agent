from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator, Mapping, cast

_SCHEMA_VERSION = 1
_TABLE_SQL = """
CREATE TABLE deliveries(
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
)
""".strip()
_INDEX_SQL = """
CREATE INDEX idx_deliveries_recoverable
ON deliveries(state, created_at, logical_delivery_id)
""".strip()


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


class DurableDeliveryStore:
    """Persist immutable delivery envelopes and forward-only settlement state."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        self.read_only = read_only

    def initialize(self) -> None:
        """Create or validate the exact ledger schema."""

        with self._transaction(write=not self.read_only) as connection:
            self._validate_schema(connection)
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                detail = "missing result" if result is None else str(result[0])
                raise RuntimeError(
                    f"Durable delivery SQLite integrity check failed: {detail}"
                )

    def recover_interrupted_provider_calls(self) -> int:
        """Freeze every crash-interrupted provider call as uncertain."""

        now = _utc_now()
        with self._transaction(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE deliveries SET state = 'uncertain', updated_at = ?
                WHERE state = 'provider_started'
                """,
                (now,),
            )
            return cursor.rowcount

    def prepare(self, envelope: Mapping[str, object]) -> dict[str, object]:
        """Insert one immutable envelope or return its exact prior row."""

        normalized = _normalize_envelope(envelope)
        now = _utc_now()
        with self._transaction(write=True) as connection:
            existing = self._lookup_identity(connection, normalized)
            if existing is not None:
                self._assert_same_envelope(existing, normalized)
                return self._view(existing)
            connection.execute(
                """
                INSERT INTO deliveries(
                    logical_delivery_id, accepted_session_id, accepted_turn_id,
                    target_service, channel, recipient, projection_session_id, body,
                    metadata_json, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)
                """,
                (
                    normalized["logical_delivery_id"],
                    normalized["accepted_session_id"],
                    normalized["accepted_turn_id"],
                    normalized["target_service"],
                    normalized["channel"],
                    normalized["recipient"],
                    normalized["projection_session_id"],
                    normalized["body"],
                    normalized["metadata_json"],
                    now,
                    now,
                ),
            )
            return self._required(connection, normalized["logical_delivery_id"])

    def mark_provider_started(
        self,
        logical_delivery_id: str,
        *,
        attempt_id: str,
        snapshot_id: str,
        generation_id: str,
        binding_token: str,
    ) -> dict[str, object]:
        """Commit the exact binding attempt before provider I/O may begin."""

        fields = tuple(
            _identity(name, value)
            for name, value in (
                ("logical_delivery_id", logical_delivery_id),
                ("attempt_id", attempt_id),
                ("snapshot_id", snapshot_id),
                ("generation_id", generation_id),
                ("binding_token", binding_token),
            )
        )
        with self._transaction(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE deliveries
                SET state = 'provider_started', attempt_id = ?, snapshot_id = ?,
                    generation_id = ?, binding_token = ?, updated_at = ?
                WHERE logical_delivery_id = ? AND state = 'prepared'
                """,
                (*fields[1:], _utc_now(), fields[0]),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"delivery provider_started transition invalid: {logical_delivery_id}"
                )
            return self._required(connection, fields[0])

    def mark_provider_result(
        self,
        logical_delivery_id: str,
        *,
        state: str,
        receipt: Mapping[str, object],
    ) -> dict[str, object]:
        """Commit delivered, rejected, or uncertain provider outcome."""

        if state not in {"delivered", "rejected", "uncertain"}:
            raise ValueError(f"provider result state invalid: {state}")
        logical_id = _identity("logical_delivery_id", logical_delivery_id)
        receipt_json = json.dumps(
            dict(receipt), sort_keys=True, separators=(",", ":")
        )
        with self._transaction(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE deliveries
                SET state = ?, provider_receipt_json = ?, updated_at = ?
                WHERE logical_delivery_id = ? AND state = 'provider_started'
                """,
                (state, receipt_json, _utc_now(), logical_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"delivery provider result transition invalid: {logical_id}"
                )
            return self._required(connection, logical_id)

    def mark_projected(
        self, logical_delivery_id: str, message_id: str
    ) -> dict[str, object]:
        """Bind one delivered envelope to its canonical Session message."""

        logical_id = _identity("logical_delivery_id", logical_delivery_id)
        projected = _identity("message_id", message_id)
        with self._transaction(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE deliveries
                SET state = 'projected', projection_message_id = ?, updated_at = ?
                WHERE logical_delivery_id = ? AND state = 'delivered'
                """,
                (projected, _utc_now(), logical_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"delivery projected transition invalid: {logical_id}"
                )
            return self._required(connection, logical_id)

    def confirm_settled(
        self, settlement_ref: str, domain_receipt: str
    ) -> dict[str, object]:
        """Commit the opaque domain receipt after Session projection."""

        logical_id = _identity("settlement_ref", settlement_ref)
        opaque = _identity("domain_receipt", domain_receipt)
        with self._transaction(write=True) as connection:
            row = self._required(connection, logical_id)
            if row["state"] == "settled":
                if row["domain_receipt"] != opaque:
                    raise RuntimeError(
                        f"delivery settlement receipt conflict: {logical_id}"
                    )
                return row
            cursor = connection.execute(
                """
                UPDATE deliveries
                SET state = 'settled', domain_receipt = ?, updated_at = ?
                WHERE logical_delivery_id = ? AND state = 'projected'
                """,
                (opaque, _utc_now(), logical_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"delivery settled transition invalid: {logical_id}"
                )
            return self._required(connection, logical_id)

    def lookup(self, accepted_session_id: str, accepted_turn_id: str) -> dict[str, object] | None:
        """Read the unique delivery associated with one accepted Turn."""

        accepted_session = _identity("accepted_session_id", accepted_session_id)
        accepted_turn = _identity("accepted_turn_id", accepted_turn_id)
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """
                SELECT * FROM deliveries
                WHERE accepted_session_id = ? AND accepted_turn_id = ?
                """,
                (accepted_session, accepted_turn),
            ).fetchone()
            return None if row is None else self._view(row)

    def recoverable(self) -> tuple[dict[str, object], ...]:
        """Read all forward-completable rows in stable creation order."""

        with self._transaction(write=False) as connection:
            rows = connection.execute(
                """
                SELECT * FROM deliveries
                WHERE state IN ('prepared', 'delivered', 'projected')
                ORDER BY created_at, logical_delivery_id
                """
            ).fetchall()
            return tuple(self._view(row) for row in rows)

    def forward_targets(self) -> frozenset[str]:
        """Read target identities still needed for provider or settlement progress."""

        if not self.path.exists():
            return frozenset()
        self.initialize()
        with self._transaction(write=False) as connection:
            rows = connection.execute(
                "SELECT DISTINCT target_service FROM deliveries "
                "WHERE state IN ('prepared', 'delivered', 'projected')"
            ).fetchall()
            return frozenset(str(row[0]) for row in rows)

    def _required(self, connection: sqlite3.Connection, logical_id: str) -> dict[str, object]:
        row = connection.execute(
            "SELECT * FROM deliveries WHERE logical_delivery_id = ?", (logical_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"durable delivery missing: {logical_id}")
        return self._view(row)

    @staticmethod
    def _lookup_identity(
        connection: sqlite3.Connection, envelope: Mapping[str, str]
    ) -> sqlite3.Row | None:
        rows = connection.execute(
            """
            SELECT * FROM deliveries
            WHERE logical_delivery_id = ? OR (
                accepted_session_id = ? AND accepted_turn_id = ?
            )
            """,
            (
                envelope["logical_delivery_id"],
                envelope["accepted_session_id"],
                envelope["accepted_turn_id"],
            ),
        ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("logical delivery 与 accepted Turn identity 分裂")
        return None if not rows else rows[0]

    @staticmethod
    def _assert_same_envelope(row: sqlite3.Row, envelope: Mapping[str, str]) -> None:
        for field in (
            "logical_delivery_id",
            "accepted_session_id",
            "accepted_turn_id",
            "target_service",
            "channel",
            "recipient",
            "projection_session_id",
            "body",
            "metadata_json",
        ):
            if row[field] != envelope[field]:
                raise RuntimeError(
                    f"durable delivery immutable envelope conflict: {field}"
                )

    @staticmethod
    def _view(row: sqlite3.Row) -> dict[str, object]:
        view = {key: row[key] for key in row.keys()}
        view["metadata"] = json.loads(cast(str, view.pop("metadata_json")))
        receipt = view.pop("provider_receipt_json")
        view["provider_receipt"] = None if receipt is None else json.loads(receipt)
        return view

    @contextmanager
    def _transaction(self, *, write: bool) -> Generator[sqlite3.Connection]:
        if write and self.read_only:
            raise PermissionError("durable delivery candidate store is read-only")
        if self.read_only:
            uri = self.path.resolve(strict=False).as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            if self.read_only:
                connection.execute("PRAGMA query_only = ON")
                connection.execute("BEGIN")
            else:
                self._validate_schema_admission(connection)
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("BEGIN IMMEDIATE")
                self._ensure_schema(connection)
            yield connection
            if self.read_only:
                connection.rollback()
            else:
                connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, _SCHEMA_VERSION):
            raise RuntimeError(f"unsupported durable delivery schema: {version}")
        if version == _SCHEMA_VERSION:
            return
        DurableDeliveryStore._validate_schema_admission(connection)
        _ = connection.execute(_TABLE_SQL)
        _ = connection.execute(_INDEX_SQL)
        _ = connection.execute(
            f"PRAGMA user_version = {_SCHEMA_VERSION}"
        )

    @staticmethod
    def _validate_schema_admission(connection: sqlite3.Connection) -> None:
        """Reject unsupported or unowned schemas before journal/schema mutation."""

        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == _SCHEMA_VERSION:
            return
        if version != 0:
            raise RuntimeError(f"unsupported durable delivery schema: {version}")
        objects = connection.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        if objects:
            identities = ", ".join(
                f"{row['type']}:{row['name']}" for row in objects
            )
            raise RuntimeError(
                "durable delivery version 0 schema must be empty: " + identities
            )

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != _SCHEMA_VERSION:
            raise RuntimeError(f"unsupported durable delivery schema: {version}")
        tables = {
            str(row["name"]): str(row["sql"])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if tables.keys() != {"deliveries"} or _normalize_sql(tables["deliveries"]) != _normalize_sql(_TABLE_SQL):
            raise RuntimeError("durable delivery schema table identity mismatch")
        indexes = {
            str(row["name"]): str(row["sql"])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'index' AND sql IS NOT NULL"
            )
        }
        if indexes != {"idx_deliveries_recoverable": _INDEX_SQL}:
            raise RuntimeError("durable delivery schema index identity mismatch")


def _normalize_envelope(envelope: Mapping[str, object]) -> dict[str, str]:
    metadata = envelope.get("metadata")
    if not isinstance(metadata, Mapping) or any(
        not isinstance(key, str) for key in metadata
    ):
        raise ValueError("delivery metadata must be a string-key mapping")
    normalized = {
        field: _identity(field, envelope.get(field))
        for field in (
            "logical_delivery_id",
            "accepted_session_id",
            "accepted_turn_id",
            "target_service",
            "channel",
            "recipient",
            "projection_session_id",
        )
    }
    body = envelope.get("body")
    if not isinstance(body, str) or not body:
        raise ValueError("body must be a non-empty string")
    typed_metadata = cast(Mapping[str, object], metadata)
    return normalized | {
        "body": body,
        "metadata_json": json.dumps(
            dict(typed_metadata), sort_keys=True, separators=(",", ":")
        )
    }


def _identity(field: str, value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be non-empty without surrounding whitespace")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()

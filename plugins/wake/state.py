from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from plugins.wake.hazard import HazardResult, advance_hazard

_SCHEMA_VERSION = 4
_ADMISSION_TABLE_SQL = """
    CREATE TABLE admission_state(
        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
        content_high_watermark INTEGER NOT NULL,
        last_content_attempt_at TEXT
    )
"""
_SEEN_TABLE_SQL = "CREATE TABLE seen_content(item_identity TEXT PRIMARY KEY)"
_ALERT_TABLE_SQL = """
    CREATE TABLE alert_events(
        source_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        not_before TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('pending', 'selected', 'delivered', 'skipped')),
        accepted_session TEXT,
        accepted_turn TEXT,
        PRIMARY KEY(source_id, event_id),
        CHECK((status = 'selected') = (accepted_session IS NOT NULL AND accepted_turn IS NOT NULL))
    )
"""
_ALERT_EXPIRY_TABLE_SQL = """
    CREATE TABLE alert_expiry(
        source_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        expires_at TEXT,
        PRIMARY KEY(source_id, event_id),
        FOREIGN KEY(source_id, event_id) REFERENCES alert_events(source_id, event_id)
    )
"""
_CONTEXT_TABLE_SQL = """
    CREATE TABLE context_events(
        source_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        observed_at TEXT NOT NULL,
        expires_at TEXT,
        PRIMARY KEY(source_id, event_id)
    )
"""
_RUN_TABLE_SQL = """
    CREATE TABLE wake_runs(
        run_id TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        started_at TEXT NOT NULL,
        candidates_seen INTEGER NOT NULL,
        candidates_selected INTEGER NOT NULL,
        screening_json TEXT NOT NULL,
        decision TEXT,
        decision_detail TEXT,
        completed_at TEXT
    )
"""


class WakeState:
    """Persist Content admission history independently from Content inbox state."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        """Create and validate the singleton admission ledger."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                connection.execute(_ADMISSION_TABLE_SQL)
                connection.execute("INSERT INTO admission_state VALUES(1, 0, NULL)")
                connection.execute(_SEEN_TABLE_SQL)
                connection.execute(_ALERT_TABLE_SQL)
                connection.execute(_ALERT_EXPIRY_TABLE_SQL)
                connection.execute(_CONTEXT_TABLE_SQL)
                connection.execute(_RUN_TABLE_SQL)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version == 1:
                self._validate_tables(connection, {"admission_state"})
                connection.execute(_SEEN_TABLE_SQL)
                connection.execute(_ALERT_TABLE_SQL)
                connection.execute(_ALERT_EXPIRY_TABLE_SQL)
                connection.execute(_CONTEXT_TABLE_SQL)
                connection.execute(_RUN_TABLE_SQL)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version == 2:
                self._validate_tables(connection, {"admission_state", "seen_content"})
                connection.execute(_ALERT_TABLE_SQL)
                connection.execute(_ALERT_EXPIRY_TABLE_SQL)
                connection.execute(_CONTEXT_TABLE_SQL)
                connection.execute(_RUN_TABLE_SQL)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version == 3:
                self._validate_tables(
                    connection,
                    {
                        "admission_state",
                        "seen_content",
                        "alert_events",
                        "context_events",
                        "wake_runs",
                    },
                )
                connection.execute(_ALERT_EXPIRY_TABLE_SQL)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version != _SCHEMA_VERSION:
                raise RuntimeError(f"不支持的 Wake state schema version: {version}")
            self._validate_tables(
                connection,
                {
                    "admission_state",
                    "seen_content",
                    "alert_events",
                    "alert_expiry",
                    "context_events",
                    "wake_runs",
                },
            )
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Wake state SQLite integrity check failed")

    @staticmethod
    def _validate_tables(
        connection: sqlite3.Connection, expected_names: set[str]
    ) -> None:
        """Reject same-version databases that do not match Wake-owned DDL."""

        rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        tables = {str(name): str(sql) for name, sql in rows}
        if set(tables) != expected_names:
            raise RuntimeError("Wake state schema mismatch: owned tables")
        expected_sql = {"admission_state": _ADMISSION_TABLE_SQL}
        if "seen_content" in expected_names:
            expected_sql["seen_content"] = _SEEN_TABLE_SQL
        if "alert_events" in expected_names:
            expected_sql["alert_events"] = _ALERT_TABLE_SQL
        if "alert_expiry" in expected_names:
            expected_sql["alert_expiry"] = _ALERT_EXPIRY_TABLE_SQL
        if "context_events" in expected_names:
            expected_sql["context_events"] = _CONTEXT_TABLE_SQL
        if "wake_runs" in expected_names:
            expected_sql["wake_runs"] = _RUN_TABLE_SQL
        for name, sql in expected_sql.items():
            if _normalize_sql(tables[name]) != _normalize_sql(sql):
                raise RuntimeError(f"Wake state schema mismatch: {name} table SQL")
        rows = connection.execute(
            "SELECT singleton FROM admission_state ORDER BY singleton"
        ).fetchall()
        if rows != [(1,)]:
            raise RuntimeError("Wake state schema mismatch: admission singleton")

    def unseen_deadline(self, items: Sequence[Mapping[str, object]]) -> datetime | None:
        """Return the earliest due time owned by unseen pending Content."""

        high_watermark, _ = self._read()
        seen = self._seen()
        deadlines = [
            _datetime(item.get("not_before"))
            for item in items
            if item.get("status") == "pending"
            and not _is_seen(item, high_watermark=high_watermark, seen=seen)
        ]
        return min(deadlines) if deadlines else None

    def has_unseen_due(
        self, items: Sequence[Mapping[str, object]], now: datetime
    ) -> bool:
        deadline = self.unseen_deadline(items)
        return deadline is not None and deadline <= _aware(now)

    def evaluate(
        self,
        items: Sequence[Mapping[str, object]],
        *,
        snapshot_seq: int,
        now: datetime,
        random_draw: float,
    ) -> HazardResult:
        """Evaluate unseen Content without consuming an admitted batch early."""

        if type(snapshot_seq) is not int or snapshot_seq < 0:
            raise ValueError("snapshot_seq 必须是非负整数")
        instant = _aware(now)
        high_watermark, last_attempt = self._read()
        seen = self._seen()
        unseen = [
            item
            for item in items
            if item.get("status") == "pending"
            and item.get("due") is True
            and not _is_seen(item, high_watermark=high_watermark, seen=seen)
        ]
        events = [_event(item) for item in items if item.get("due") is True]
        result = advance_hazard(
            events,
            now=instant,
            new_item_ids={_item_identity(item) for item in unseen},
            random_draw=random_draw,
            last_wake_at=last_attempt,
        )
        if result.should_wake:
            return result
        self._mark_content_seen(unseen, admitted_at=None)
        return result

    def commit_content_admission(
        self,
        items: Sequence[Mapping[str, object]],
        *,
        now: datetime,
    ) -> None:
        """Consume Content only after its durable selection receipt exists."""

        instant = _aware(now)
        high_watermark, _ = self._read()
        seen = self._seen()
        unseen = [
            item
            for item in items
            if item.get("status") == "pending"
            and item.get("due") is True
            and not _is_seen(item, high_watermark=high_watermark, seen=seen)
        ]
        self._mark_content_seen(unseen, admitted_at=instant)

    def _mark_content_seen(
        self,
        items: Sequence[Mapping[str, object]],
        *,
        admitted_at: datetime | None,
    ) -> None:
        """Commit exact Content identities and an optional successful admission."""

        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE admission_state
                SET last_content_attempt_at = COALESCE(?, last_content_attempt_at)
                WHERE singleton = 1
                """,
                (None if admitted_at is None else admitted_at.isoformat(),),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO seen_content(item_identity) VALUES (?)",
                ((_item_identity(item),) for item in items),
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
        """Update one pending Alert; selection freezes its final payload."""

        source, event = _identity(source_id, event_id)
        instant = _aware(observed_at)
        encoded = _payload_json(payload)
        expiry = None if expires_at is None else _aware(expires_at).isoformat()
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "INSERT OR IGNORE INTO alert_events VALUES(?, ?, ?, ?, ?, 'pending', NULL, NULL)",
                (source, event, encoded, instant.isoformat(), instant.isoformat()),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT status FROM alert_events "
                    "WHERE source_id = ? AND event_id = ?",
                    (source, event),
                ).fetchone()
                if row == ("pending",):
                    connection.execute(
                        "UPDATE alert_events SET payload_json = ?, observed_at = ? "
                        "WHERE source_id = ? AND event_id = ? AND status = 'pending'",
                        (encoded, instant.isoformat(), source, event),
                    )
            connection.execute(
                "INSERT INTO alert_expiry VALUES(?, ?, ?) "
                "ON CONFLICT(source_id, event_id) DO UPDATE SET expires_at = excluded.expires_at",
                (source, event, expiry),
            )
        return {
            "accepted": cursor.rowcount == 1,
            "source_id": source,
            "event_id": event,
        }

    def alert_deadline(self, now: datetime | None = None) -> datetime | None:
        self.initialize()
        instant = _aware(now or datetime.now(UTC))
        with closing(sqlite3.connect(self.path)) as connection, connection:
            self._expire_alerts(connection, instant)
            row = connection.execute(
                "SELECT MIN(not_before) FROM alert_events WHERE status = 'pending'"
            ).fetchone()
        return None if row is None or row[0] is None else _datetime(row[0])

    def select_alert(
        self,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object] | None:
        """Claim the oldest due Alert for one accepted Turn."""

        session_id = _identity_part(accepted_turn.get("session_id"), "session_id")
        turn_id = _identity_part(accepted_turn.get("turn_id"), "turn_id")
        instant = _aware(now)
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_alerts(connection, instant)
            row = connection.execute(
                "SELECT source_id, event_id, payload_json, observed_at, not_before "
                "FROM alert_events WHERE status = 'pending' AND not_before <= ? "
                "ORDER BY observed_at, source_id, event_id LIMIT 1",
                (instant.isoformat(),),
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                "UPDATE alert_events SET status = 'selected', accepted_session = ?, "
                "accepted_turn = ? WHERE source_id = ? AND event_id = ? "
                "AND status = 'pending'",
                (session_id, turn_id, row[0], row[1]),
            )
            if cursor.rowcount != 1:
                return None
        return _alert_row(row, session_id=session_id, turn_id=turn_id)

    def selected_alert(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        session_id = _identity_part(accepted_turn.get("session_id"), "session_id")
        turn_id = _identity_part(accepted_turn.get("turn_id"), "turn_id")
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT source_id, event_id, payload_json, observed_at, not_before "
                "FROM alert_events WHERE status = 'selected' "
                "AND accepted_session = ? AND accepted_turn = ?",
                (session_id, turn_id),
            ).fetchone()
        return (
            None
            if row is None
            else _alert_row(row, session_id=session_id, turn_id=turn_id)
        )

    def selected_alerts(self) -> tuple[Mapping[str, object], ...]:
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT source_id, event_id, payload_json, observed_at, not_before, "
                "accepted_session, accepted_turn FROM alert_events "
                "WHERE status = 'selected' ORDER BY observed_at"
            ).fetchall()
        return tuple(
            _alert_row(row[:5], session_id=str(row[5]), turn_id=str(row[6]))
            for row in rows
        )

    def alert_status(self, source_id: str, event_id: str) -> str | None:
        source, event = _identity(source_id, event_id)
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT status FROM alert_events WHERE source_id = ? AND event_id = ?",
                (source, event),
            ).fetchone()
        return None if row is None else str(row[0])

    def expire_alerts(self, now: datetime) -> int:
        """Close every expired Alert before selection or delivery."""

        instant = _aware(now)
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            return self._expire_alerts(connection, instant)

    def expire_alert(self, source_id: str, event_id: str, now: datetime) -> bool:
        """Close one expired Alert before its provider call starts."""

        source, event = _identity(source_id, event_id)
        instant = _aware(now)
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            cursor = connection.execute(
                "UPDATE alert_events SET status = 'skipped', accepted_session = NULL, "
                "accepted_turn = NULL WHERE source_id = ? AND event_id = ? "
                "AND status IN ('pending', 'selected') AND EXISTS ("
                "SELECT 1 FROM alert_expiry WHERE alert_expiry.source_id = ? "
                "AND alert_expiry.event_id = ? AND alert_expiry.expires_at IS NOT NULL "
                "AND alert_expiry.expires_at <= ?)",
                (source, event, source, event, instant.isoformat()),
            )
        return cursor.rowcount == 1

    @staticmethod
    def _expire_alerts(connection: sqlite3.Connection, now: datetime) -> int:
        cursor = connection.execute(
            "UPDATE alert_events SET status = 'skipped', accepted_session = NULL, "
            "accepted_turn = NULL WHERE status IN ('pending', 'selected') AND EXISTS ("
            "SELECT 1 FROM alert_expiry WHERE alert_expiry.source_id = alert_events.source_id "
            "AND alert_expiry.event_id = alert_events.event_id "
            "AND alert_expiry.expires_at IS NOT NULL AND alert_expiry.expires_at <= ?)",
            (now.isoformat(),),
        )
        return cursor.rowcount

    def defer_alert(
        self,
        source_id: str,
        event_id: str,
        not_before: datetime,
    ) -> None:
        self._transition_alert(source_id, event_id, "pending", not_before=not_before)

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
        not_before: datetime | None = None,
    ) -> None:
        source, event = _identity(source_id, event_id)
        deadline = None if not_before is None else _aware(not_before).isoformat()
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            cursor = connection.execute(
                "UPDATE alert_events SET status = ?, "
                "not_before = COALESCE(?, not_before), accepted_session = NULL, "
                "accepted_turn = NULL WHERE source_id = ? AND event_id = ? "
                "AND status = 'selected'",
                (status, deadline, source, event),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("Wake Alert transition 未命中 selected row")

    def report_context(
        self,
        *,
        source_id: str,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None,
    ) -> Mapping[str, object]:
        """Upsert one source-owned current Context fact."""

        source, event = _identity(source_id, event_id)
        instant = _aware(observed_at)
        expiry = None if expires_at is None else _aware(expires_at)
        if expiry is not None and expiry <= instant:
            raise ValueError("Wake Context expires_at 必须晚于 observed_at")
        encoded = _payload_json(payload)
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "INSERT INTO context_events VALUES(?, ?, ?, ?, ?) "
                "ON CONFLICT(source_id, event_id) DO UPDATE SET "
                "payload_json = excluded.payload_json, observed_at = excluded.observed_at, "
                "expires_at = excluded.expires_at",
                (
                    source,
                    event,
                    encoded,
                    instant.isoformat(),
                    None if expiry is None else expiry.isoformat(),
                ),
            )
        return {"accepted": True, "source_id": source, "event_id": event}

    def active_context(self, now: datetime) -> tuple[Mapping[str, object], ...]:
        instant = _aware(now)
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT source_id, event_id, payload_json, observed_at, expires_at "
                "FROM context_events WHERE expires_at IS NULL OR expires_at > ? "
                "ORDER BY observed_at DESC, source_id, event_id",
                (instant.isoformat(),),
            ).fetchall()
        return tuple(
            {
                "source_id": str(row[0]),
                "event_id": str(row[1]),
                "payload": _decode_payload(row[2]),
                "observed_at": str(row[3]),
                "expires_at": None if row[4] is None else str(row[4]),
            }
            for row in rows
        )

    def record_screen(
        self,
        *,
        run_id: str,
        owner: str,
        candidates_seen: int,
        screening: Sequence[Mapping[str, object]],
        started_at: datetime,
    ) -> None:
        """Append one Dashboard run row after the screening result exists."""

        identity = _identity_part(run_id, "run_id")
        if owner not in {"alert", "content", "drift"}:
            raise ValueError("Wake run owner 无效")
        if candidates_seen < 0 or candidates_seen < len(screening):
            raise ValueError("Wake run candidate counts 无效")
        encoded = json.dumps(
            [dict(item) for item in screening],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        instant = _aware(started_at)
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO wake_runs VALUES(?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
                (
                    identity,
                    owner,
                    instant.isoformat(),
                    candidates_seen,
                    len(screening),
                    encoded,
                ),
            )

    def record_decision(
        self,
        *,
        run_id: str,
        decision: str,
        detail: str,
        completed_at: datetime,
    ) -> None:
        if decision not in {"share", "skip", "defer"}:
            raise ValueError("Wake run decision 无效")
        identity = _identity_part(run_id, "run_id")
        instant = _aware(completed_at)
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            cursor = connection.execute(
                "UPDATE wake_runs SET decision = ?, decision_detail = ?, completed_at = ? "
                "WHERE run_id = ? AND decision IS NULL",
                (decision, detail, instant.isoformat(), identity),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT decision, decision_detail FROM wake_runs WHERE run_id = ?",
                    (identity,),
                ).fetchone()
                if row != (decision, detail):
                    raise RuntimeError("Wake run decision 缺少 open screen row")

    def list_runs(
        self, limit: int = 100, *, offset: int = 0
    ) -> tuple[Mapping[str, object], ...]:
        if not 1 <= limit <= 500:
            raise ValueError("Wake run limit 必须是 1..500")
        if offset < 0:
            raise ValueError("Wake run offset 不能为负数")
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT run_id, owner, started_at, candidates_seen, "
                "candidates_selected, decision, decision_detail, completed_at "
                "FROM wake_runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return tuple(_run_summary(row) for row in rows)

    def count_runs(self) -> int:
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute("SELECT count(*) FROM wake_runs").fetchone()
        return int(row[0])

    def get_run(self, run_id: str) -> Mapping[str, object] | None:
        identity = _identity_part(run_id, "run_id")
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT run_id, owner, started_at, candidates_seen, "
                "candidates_selected, decision, decision_detail, completed_at, "
                "screening_json FROM wake_runs WHERE run_id = ?",
                (identity,),
            ).fetchone()
        if row is None:
            return None
        summary = dict(_run_summary(row[:8]))
        screening = json.loads(str(row[8]))
        if not isinstance(screening, list):
            raise RuntimeError("Wake run screening 不是 JSON array")
        summary["screening"] = screening
        return summary

    def _read(self) -> tuple[int, datetime | None]:
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT content_high_watermark, last_content_attempt_at "
                "FROM admission_state WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("Wake admission_state singleton 缺失")
        return int(row[0]), None if row[1] is None else _datetime(row[1])

    def _seen(self) -> frozenset[str]:
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT item_identity FROM seen_content"
            ).fetchall()
        return frozenset(str(row[0]) for row in rows)


def _event(item: Mapping[str, object]) -> dict[str, object]:
    payload = item.get("payload")
    ref = item.get("ref")
    if not isinstance(payload, Mapping) or not isinstance(ref, Mapping):
        raise ValueError("Wake Content item 缺少 payload/ref")
    event = dict(cast(Mapping[str, object], payload))
    event["id"] = _item_id(item)
    event["source_id"] = ref.get("source_id", "")
    event["_wake_admission_identity"] = _item_identity(item)
    if "_wake_interest_score" not in event:
        event["_wake_interest_score"] = payload.get("preprocess_score", 1.0)
    return event


def _item_id(item: Mapping[str, object]) -> str:
    ref = item.get("ref")
    if not isinstance(ref, Mapping):
        raise ValueError("Wake Content item 缺少 ref")
    value = ref.get("item_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Wake Content item_id 必须非空")
    return value


def _item_identity(item: Mapping[str, object]) -> str:
    ref = item.get("ref")
    if not isinstance(ref, Mapping):
        raise ValueError("Wake Content item 缺少 ref")
    fields = tuple(ref.get(field) for field in ("source_id", "item_id", "revision"))
    if any(not isinstance(value, str) or not value for value in fields):
        raise ValueError("Wake Content ref identity 必须是非空字符串")
    return "\x00".join(cast(tuple[str, str, str], fields))


def _is_seen(
    item: Mapping[str, object], *, high_watermark: int, seen: frozenset[str]
) -> bool:
    return (
        _integer(item.get("snapshot_seq"), "snapshot_seq") <= high_watermark
        or _item_identity(item) in seen
    )


def _integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} 必须是非负整数")
    return value


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Wake deadline 必须是 ISO 字符串")
    return _aware(datetime.fromisoformat(value))


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Wake state 时间必须带时区")
    return value.astimezone(UTC)


def _normalize_sql(value: str) -> str:
    return " ".join(value.replace('"', "").split()).lower()


def _identity(source_id: str, event_id: str) -> tuple[str, str]:
    return (
        _identity_part(source_id, "source_id"),
        _identity_part(event_id, "event_id"),
    )


def _identity_part(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"Wake {field} 必须非空且无首尾空白")
    return value


def _payload_json(payload: Mapping[str, object]) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError("Wake payload 必须是 Mapping")
    try:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Wake payload 必须是 JSON object") from error
    if json.loads(encoded) == {}:
        raise ValueError("Wake payload 不能为空")
    return encoded


def _decode_payload(value: object) -> Mapping[str, object]:
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise RuntimeError("Wake stored payload 不是 JSON object")
    return cast(Mapping[str, object], decoded)


def _alert_row(
    row: Sequence[object],
    *,
    session_id: str,
    turn_id: str,
) -> Mapping[str, object]:
    return {
        "source_id": str(row[0]),
        "event_id": str(row[1]),
        "payload": _decode_payload(row[2]),
        "observed_at": str(row[3]),
        "not_before": str(row[4]),
        "accepted_turn": {"session_id": session_id, "turn_id": turn_id},
    }


def _run_summary(row: Sequence[object]) -> Mapping[str, object]:
    return {
        "run_id": str(row[0]),
        "owner": str(row[1]),
        "started_at": str(row[2]),
        "candidates_seen": _stored_int(row[3], "candidates_seen"),
        "candidates_selected": _stored_int(row[4], "candidates_selected"),
        "decision": None if row[5] is None else str(row[5]),
        "decision_detail": None if row[6] is None else str(row[6]),
        "completed_at": None if row[7] is None else str(row[7]),
    }


def _stored_int(value: object, field: str) -> int:
    if not isinstance(value, int):
        raise RuntimeError(f"Wake stored {field} 不是整数")
    return value


class WakeAlertInputs:
    def __init__(self, state: WakeState, changed: Callable[[], None]) -> None:
        self._state = state
        self._changed = changed

    def report(
        self,
        *,
        source_id: str,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]:
        result = self._state.report_alert(
            source_id=source_id,
            event_id=event_id,
            payload=payload,
            observed_at=observed_at,
            expires_at=expires_at,
        )
        self._changed()
        return result

    def status(self, *, source_id: str, event_id: str) -> str | None:
        return self._state.alert_status(source_id, event_id)


class WakeContextInputs:
    def __init__(self, state: WakeState, changed: Callable[[], None]) -> None:
        self._state = state
        self._changed = changed

    def report(
        self,
        *,
        source_id: str,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]:
        result = self._state.report_context(
            source_id=source_id,
            event_id=event_id,
            payload=payload,
            observed_at=observed_at,
            expires_at=expires_at,
        )
        self._changed()
        return result


__all__ = ["WakeAlertInputs", "WakeContextInputs", "WakeState"]

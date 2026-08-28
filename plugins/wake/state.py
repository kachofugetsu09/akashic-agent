from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from .hazard import HazardResult, advance_hazard

_SCHEMA_VERSION = 6
_ADMISSION_TABLE_SQL = """
    CREATE TABLE admission_state(
        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
        content_high_watermark INTEGER NOT NULL,
        last_content_attempt_at TEXT
    )
"""
_SEEN_TABLE_SQL = "CREATE TABLE seen_content(item_identity TEXT PRIMARY KEY)"
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
_ATTEMPT_TABLE_SQL = """
    CREATE TABLE wake_attempts(
        attempt_id TEXT PRIMARY KEY,
        timer_id TEXT NOT NULL,
        scheduled_for TEXT NOT NULL,
        fired_at TEXT NOT NULL,
        mail_watermark INTEGER NOT NULL,
        outcome TEXT NOT NULL CHECK(outcome IN (
            'checking', 'no_due', 'content_insufficient', 'admission_rejected',
            'shared', 'model_skip', 'deferred', 'delivery_unknown', 'failed'
        )),
        owner TEXT CHECK(owner IN ('alert', 'content', 'drift')),
        detail TEXT,
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
                connection.execute(_RUN_TABLE_SQL)
                connection.execute(_ATTEMPT_TABLE_SQL)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version != _SCHEMA_VERSION:
                raise RuntimeError(
                    "旧 Wake state 必须先由 EventMail 安装迁移处理: "
                    f"schema version {version}"
                )
            self._validate_tables(
                connection,
                {
                    "admission_state",
                    "seen_content",
                    "wake_runs",
                    "wake_attempts",
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
        if "wake_runs" in expected_names:
            expected_sql["wake_runs"] = _RUN_TABLE_SQL
        if "wake_attempts" in expected_names:
            expected_sql["wake_attempts"] = _ATTEMPT_TABLE_SQL
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

    def begin_attempt(
        self,
        *,
        attempt_id: str,
        timer_id: str,
        scheduled_for: datetime,
        fired_at: datetime,
        mail_watermark: int,
    ) -> None:
        """Persist one actual Timer fire before reading Wake duties."""

        identity = _identity_part(attempt_id, "attempt_id")
        timer = _identity_part(timer_id, "timer_id")
        if type(mail_watermark) is not int or mail_watermark < 0:
            raise ValueError("EventMail watermark 必须是非负整数")
        scheduled = _aware(scheduled_for).isoformat()
        fired = _aware(fired_at).isoformat()
        self.initialize()
        row = (
            identity,
            timer,
            scheduled,
            fired,
            mail_watermark,
            "checking",
            None,
            None,
            None,
        )
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO wake_attempts VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            stored = connection.execute(
                "SELECT attempt_id, timer_id, scheduled_for, fired_at, "
                "mail_watermark, outcome, owner, detail, completed_at "
                "FROM wake_attempts WHERE attempt_id = ?",
                (identity,),
            ).fetchone()
            if stored != row:
                raise RuntimeError("Wake attempt identity 冲突")

    def finish_attempt(
        self,
        *,
        attempt_id: str,
        outcome: str,
        owner: str | None,
        detail: str,
        completed_at: datetime,
    ) -> None:
        """Close one Timer attempt with its observable outcome."""

        identity = _identity_part(attempt_id, "attempt_id")
        if outcome not in {
            "no_due",
            "content_insufficient",
            "admission_rejected",
            "shared",
            "model_skip",
            "deferred",
            "delivery_unknown",
            "failed",
        }:
            raise ValueError("Wake attempt outcome 无效")
        if owner is not None and owner not in {"alert", "content", "drift"}:
            raise ValueError("Wake attempt owner 无效")
        instant = _aware(completed_at).isoformat()
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            cursor = connection.execute(
                "UPDATE wake_attempts SET outcome = ?, owner = ?, detail = ?, "
                "completed_at = ? WHERE attempt_id = ? AND outcome = 'checking'",
                (outcome, owner, detail, instant, identity),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT outcome, owner, detail FROM wake_attempts "
                    "WHERE attempt_id = ?",
                    (identity,),
                ).fetchone()
                if row != (outcome, owner, detail):
                    raise RuntimeError("Wake attempt 缺少 open row")

    def close_interrupted_attempts(self, recovered_at: datetime) -> int:
        """Close Timer fires interrupted by a previous process generation."""

        instant = _aware(recovered_at).isoformat()
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            cursor = connection.execute(
                "UPDATE wake_attempts SET outcome='delivery_unknown', "
                "detail='进程重启前检查未闭合，外部效果未知', completed_at=? "
                "WHERE outcome='checking'",
                (instant,),
            )
            return cursor.rowcount

    def list_attempts(
        self, limit: int = 100, *, offset: int = 0
    ) -> tuple[Mapping[str, object], ...]:
        if not 1 <= limit <= 500:
            raise ValueError("Wake attempt limit 必须是 1..500")
        if offset < 0:
            raise ValueError("Wake attempt offset 不能为负数")
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT attempt_id, timer_id, scheduled_for, fired_at, "
                "mail_watermark, outcome, owner, detail, completed_at "
                "FROM wake_attempts ORDER BY fired_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return tuple(_attempt_summary(row) for row in rows)

    def count_attempts(self) -> int:
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute("SELECT count(*) FROM wake_attempts").fetchone()
        return int(row[0])

    def get_attempt(self, attempt_id: str) -> Mapping[str, object] | None:
        identity = _identity_part(attempt_id, "attempt_id")
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT attempt_id, timer_id, scheduled_for, fired_at, "
                "mail_watermark, outcome, owner, detail, completed_at "
                "FROM wake_attempts WHERE attempt_id = ?",
                (identity,),
            ).fetchone()
        return None if row is None else _attempt_summary(row)

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


def _identity_part(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"Wake {field} 必须非空且无首尾空白")
    return value


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


def _attempt_summary(row: Sequence[object]) -> Mapping[str, object]:
    return {
        "attempt_id": str(row[0]),
        "timer_id": str(row[1]),
        "scheduled_for": str(row[2]),
        "fired_at": str(row[3]),
        "mail_watermark": _stored_int(row[4], "mail_watermark"),
        "outcome": str(row[5]),
        "owner": None if row[6] is None else str(row[6]),
        "detail": None if row[7] is None else str(row[7]),
        "completed_at": None if row[8] is None else str(row[8]),
    }


def _stored_int(value: object, field: str) -> int:
    if not isinstance(value, int):
        raise RuntimeError(f"Wake stored {field} 不是整数")
    return value


__all__ = ["WakeState"]

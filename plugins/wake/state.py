from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from plugins.wake.hazard import HazardResult, advance_hazard

_SCHEMA_VERSION = 2
_ADMISSION_TABLE_SQL = """
    CREATE TABLE admission_state(
        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
        content_high_watermark INTEGER NOT NULL,
        last_content_attempt_at TEXT
    )
"""
_SEEN_TABLE_SQL = "CREATE TABLE seen_content(item_identity TEXT PRIMARY KEY)"


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
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version == 1:
                self._validate_tables(connection, {"admission_state"})
                connection.execute(_SEEN_TABLE_SQL)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version != _SCHEMA_VERSION:
                raise RuntimeError(f"不支持的 Wake state schema version: {version}")
            self._validate_tables(connection, {"admission_state", "seen_content"})
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
        """Advance the legacy admission draw once for each frozen Content watermark."""

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
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT content_high_watermark FROM admission_state WHERE singleton = 1"
            ).fetchone()
            if current is None or int(current[0]) != high_watermark:
                raise RuntimeError(
                    "Wake Content admission watermark changed concurrently"
                )
            connection.execute(
                """
                UPDATE admission_state
                SET last_content_attempt_at = CASE WHEN ? THEN ? ELSE last_content_attempt_at END
                WHERE singleton = 1
                """,
                (
                    int(result.should_wake),
                    instant.isoformat(),
                ),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO seen_content(item_identity) VALUES (?)",
                ((_item_identity(item),) for item in unseen),
            )
        return result

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


__all__ = ["WakeState"]

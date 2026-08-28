from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from .pool import WAKE_ADMISSION_FLOOR, PoolResult, measure_pool, rank_events

_SCHEMA_VERSION = 8
_ADMISSION_TABLE_SQL = """
    CREATE TABLE admission_state(
        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
        content_high_watermark INTEGER NOT NULL
    )
"""
_SEEN_TABLE_SQL = "CREATE TABLE seen_content(item_identity TEXT PRIMARY KEY)"
_SCORE_TABLE_SQL = """
    CREATE TABLE content_scores(
        source_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        revision TEXT NOT NULL,
        initial_score REAL NOT NULL CHECK(initial_score >= 0 AND initial_score <= 7.0),
        semantic_interest REAL NOT NULL CHECK(semantic_interest >= 0 AND semantic_interest <= 0.999),
        scored_at TEXT NOT NULL,
        PRIMARY KEY(source_id, item_id, revision)
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
_ATTEMPT_TABLE_SQL = """
    CREATE TABLE wake_attempts(
        attempt_id TEXT PRIMARY KEY,
        timer_id TEXT NOT NULL,
        scheduled_for TEXT NOT NULL,
        fired_at TEXT NOT NULL,
        mail_watermark INTEGER,
        outcome TEXT NOT NULL CHECK(outcome IN (
            'checking', 'no_due', 'content_insufficient', 'admission_rejected',
            'shared', 'model_skip', 'deferred', 'cancelled_after_fire',
            'delivery_unknown', 'failed'
        )),
        owner TEXT CHECK(owner IN ('alert', 'content', 'drift')),
        detail TEXT,
        completed_at TEXT
    )
"""


@dataclass(frozen=True, slots=True)
class ContentScore:
    source_id: str
    item_id: str
    revision: str
    initial_score: float
    semantic_interest: float
    scored_at: datetime


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
                connection.execute("INSERT INTO admission_state VALUES(1, 0)")
                connection.execute(_SEEN_TABLE_SQL)
                connection.execute(_SCORE_TABLE_SQL)
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
                    "content_scores",
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
        if "content_scores" in expected_names:
            expected_sql["content_scores"] = _SCORE_TABLE_SQL
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

        high_watermark = self._read()
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

    def unseen_due_count(
        self, items: Sequence[Mapping[str, object]], now: datetime
    ) -> int:
        """Count exact new Content identities that may trigger one pool check."""

        instant = _aware(now)
        high_watermark = self._read()
        seen = self._seen()
        return sum(
            1
            for item in items
            if item.get("status") == "pending"
            and item.get("due") is True
            and _datetime(item.get("not_before")) <= instant
            and not _is_seen(item, high_watermark=high_watermark, seen=seen)
        )

    def unscored_due_items(
        self, items: Sequence[Mapping[str, object]]
    ) -> tuple[Mapping[str, object], ...]:
        """Return due Content whose one-time initial score is still missing."""

        due = tuple(
            item
            for item in items
            if item.get("status") in {"pending", "deferred"}
            and item.get("due") is True
        )
        scores = self._scores({_score_identity(item) for item in due})
        return tuple(item for item in due if _score_identity(item) not in scores)

    def record_content_scores(self, scores: Sequence[ContentScore]) -> None:
        """Append immutable initial scores and reject identity collisions."""

        if not scores:
            return
        rows: list[tuple[object, ...]] = []
        for score in scores:
            identity = tuple(
                _identity_part(value, field)
                for field, value in (
                    ("source_id", score.source_id),
                    ("item_id", score.item_id),
                    ("revision", score.revision),
                )
            )
            initial = _score_value(score.initial_score, "initial_score", maximum=7.0)
            semantic = _score_value(score.semantic_interest, "semantic_interest")
            rows.append((*identity, initial, semantic, _aware(score.scored_at).isoformat()))
        if len({tuple(row[:3]) for row in rows}) != len(rows):
            raise ValueError("Wake Content score batch identity 重复")
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executemany(
                "INSERT OR IGNORE INTO content_scores VALUES(?, ?, ?, ?, ?, ?)",
                rows,
            )
            for row in rows:
                stored = connection.execute(
                    "SELECT source_id, item_id, revision, initial_score, "
                    "semantic_interest, scored_at FROM content_scores "
                    "WHERE source_id=? AND item_id=? AND revision=?",
                    row[:3],
                ).fetchone()
                if stored != row:
                    raise RuntimeError("Wake Content 初始分 identity 冲突")

    def scored_items(
        self, items: Sequence[Mapping[str, object]]
    ) -> tuple[Mapping[str, object], ...]:
        """Attach Wake-owned initial scores to the current EventMail view."""

        due_identities = {
            _score_identity(item)
            for item in items
            if item.get("status") in {"pending", "deferred"}
            and item.get("due") is True
        }
        scores = self._scores(due_identities)
        result: list[Mapping[str, object]] = []
        for item in items:
            if not (
                item.get("status") in {"pending", "deferred"}
                and item.get("due") is True
            ):
                result.append(item)
                continue
            score = scores.get(_score_identity(item))
            if score is None:
                raise RuntimeError("到期 Content 缺少 Wake 初始分")
            payload = item.get("payload")
            if not isinstance(payload, Mapping):
                raise ValueError("Wake Content item 缺少 payload")
            enriched = dict(cast(Mapping[str, object], payload))
            enriched["_wake_initial_score"] = score.initial_score
            enriched["_wake_semantic_interest"] = score.semantic_interest
            result.append({**dict(item), "payload": enriched})
        return tuple(result)

    def next_maintenance_deadline(
        self, now: datetime, *, interval: timedelta
    ) -> datetime:
        """Return the next five-minute pool audit from durable Timer history."""

        instant = _aware(now)
        if interval <= timedelta(0):
            raise ValueError("Wake maintenance interval 必须为正数")
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT MAX(fired_at) FROM wake_attempts"
            ).fetchone()
        if row is None or row[0] is None:
            return instant + interval
        deadline = _datetime(row[0]) + interval
        return max(instant, deadline)

    def expired_content_refs(
        self,
        items: Sequence[Mapping[str, object]],
        *,
        now: datetime,
        minimum_residence: timedelta,
        limit: int = 256,
    ) -> tuple[Mapping[str, object], ...]:
        """Select old low-mass Content without taking EventMail write ownership."""

        instant = _aware(now)
        if minimum_residence < timedelta(0):
            raise ValueError("Content minimum residence 不能为负数")
        if not 1 <= limit <= 256:
            raise ValueError("Content expiry limit 必须是 1..256")
        eligible = sorted(
            (
                item
                for item in items
                if item.get("status") in {"pending", "deferred"}
                and item.get("due") is True
                and _datetime(item.get("observed_at")) <= instant - minimum_residence
            ),
            key=lambda item: _datetime(item.get("observed_at")),
        )[:limit]
        expired: list[Mapping[str, object]] = []
        for item in eligible:
            ranked = rank_events([_event(item)], now=instant)
            features = ranked[0]["_wake_rank_features"]
            if float(features["admission_mass"]) < WAKE_ADMISSION_FLOOR:
                ref = item.get("ref")
                if not isinstance(ref, Mapping):
                    raise ValueError("Wake Content item 缺少 ref")
                expired.append(dict(ref))
        return tuple(expired)

    def audit_pool(
        self,
        items: Sequence[Mapping[str, object]],
        *,
        now: datetime,
    ) -> PoolResult:
        """Measure the decayed due pool without starting a Turn."""

        instant = _aware(now)
        return measure_pool(
            [_event(item) for item in items if item.get("due") is True],
            now=instant,
        )

    def evaluate(
        self,
        items: Sequence[Mapping[str, object]],
        *,
        snapshot_seq: int,
        now: datetime,
    ) -> PoolResult:
        """Compare the fixed-score pool with its deterministic threshold."""

        if type(snapshot_seq) is not int or snapshot_seq < 0:
            raise ValueError("snapshot_seq 必须是非负整数")
        instant = _aware(now)
        high_watermark = self._read()
        seen = self._seen()
        unseen = [
            item
            for item in items
            if item.get("status") == "pending"
            and item.get("due") is True
            and not _is_seen(item, high_watermark=high_watermark, seen=seen)
        ]
        events = [_event(item) for item in items if item.get("due") is True]
        result = measure_pool(
            events,
            now=instant,
            new_item_ids={_item_identity(item) for item in unseen},
        )
        if result.should_wake:
            return result
        self._mark_content_seen(unseen)
        return result

    def commit_content_admission(
        self,
        items: Sequence[Mapping[str, object]],
    ) -> None:
        """Consume Content only after its durable selection receipt exists."""

        high_watermark = self._read()
        seen = self._seen()
        unseen = [
            item
            for item in items
            if item.get("status") == "pending"
            and item.get("due") is True
            and not _is_seen(item, high_watermark=high_watermark, seen=seen)
        ]
        self._mark_content_seen(unseen)

    def _mark_content_seen(
        self,
        items: Sequence[Mapping[str, object]],
    ) -> None:
        """Commit exact Content identities already checked by the pool."""

        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
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
    ) -> None:
        """Persist one actual Timer fire before reading Wake duties."""

        identity = _identity_part(attempt_id, "attempt_id")
        timer = _identity_part(timer_id, "timer_id")
        scheduled = _aware(scheduled_for).isoformat()
        fired = _aware(fired_at).isoformat()
        self.initialize()
        row = (
            identity,
            timer,
            scheduled,
            fired,
            None,
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

    def set_attempt_mail_watermark(
        self, *, attempt_id: str, mail_watermark: int
    ) -> None:
        """Freeze the EventMail watermark after the Timer fire is durable."""

        identity = _identity_part(attempt_id, "attempt_id")
        if type(mail_watermark) is not int or mail_watermark < 0:
            raise ValueError("EventMail watermark 必须是非负整数")
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            cursor = connection.execute(
                "UPDATE wake_attempts SET mail_watermark=? "
                "WHERE attempt_id=? AND outcome='checking' "
                "AND mail_watermark IS NULL",
                (mail_watermark, identity),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT mail_watermark FROM wake_attempts WHERE attempt_id=?",
                    (identity,),
                ).fetchone()
                if row != (mail_watermark,):
                    raise RuntimeError("Wake attempt 缺少 open watermark row")

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
            "cancelled_after_fire",
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

    def _scores(
        self, identities: set[tuple[str, str, str]]
    ) -> dict[tuple[str, str, str], ContentScore]:
        """Load only score rows referenced by the current EventMail view."""

        if not identities:
            return {}
        self.initialize()
        rows: list[tuple[object, ...]] = []
        with closing(sqlite3.connect(self.path)) as connection:
            ordered = sorted(identities)
            for offset in range(0, len(ordered), 256):
                chunk = ordered[offset : offset + 256]
                placeholders = ",".join("(?, ?, ?)" for _ in chunk)
                rows.extend(
                    connection.execute(
                        "SELECT source_id, item_id, revision, initial_score, "
                        "semantic_interest, scored_at FROM content_scores "
                        f"WHERE (source_id, item_id, revision) IN ({placeholders})",
                        tuple(value for identity in chunk for value in identity),
                    ).fetchall()
                )
        return {
            (str(source), str(item), str(revision)): ContentScore(
                source_id=str(source),
                item_id=str(item),
                revision=str(revision),
                initial_score=_score_value(
                    cast(float, initial), "stored initial_score", maximum=7.0
                ),
                semantic_interest=_score_value(
                    cast(float, semantic), "stored semantic_interest"
                ),
                scored_at=_datetime(scored_at),
            )
            for source, item, revision, initial, semantic, scored_at in rows
        }

    def _read(self) -> int:
        self.initialize()
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute(
                "SELECT content_high_watermark FROM admission_state WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("Wake admission_state singleton 缺失")
        return int(row[0])

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
    event.setdefault("first_seen_at", item.get("observed_at"))
    event["_wake_admission_identity"] = _item_identity(item)
    if "_wake_initial_score" not in event:
        raise RuntimeError("Wake Content item 缺少持久化初始质量")
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


def _score_identity(item: Mapping[str, object]) -> tuple[str, str, str]:
    ref = item.get("ref")
    if not isinstance(ref, Mapping):
        raise ValueError("Wake Content item 缺少 ref")
    values = tuple(ref.get(field) for field in ("source_id", "item_id", "revision"))
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Wake Content ref identity 必须是非空字符串")
    return cast(tuple[str, str, str], values)


def _score_value(value: float, field: str, *, maximum: float = 0.999) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Wake Content {field} 必须是数字")
    result = float(value)
    if not 0.0 <= result <= maximum:
        raise ValueError(f"Wake Content {field} 必须在 0..{maximum}")
    return result


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
        "mail_watermark": (
            None if row[4] is None else _stored_int(row[4], "mail_watermark")
        ),
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

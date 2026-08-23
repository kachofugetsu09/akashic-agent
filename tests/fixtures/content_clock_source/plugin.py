from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Generator, Protocol

from agent.control.timer import TimerHandle, TimerStatus
from agent.plugin_composition import (
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    TIMERS,
    Context,
    PluginTimers,
    ServiceKey,
)

api_version = 3
name = "content_clock_source"
version = "3.0.0"
desc = "Deterministic clock and feed boundary for Content composition tests"
author = "Akashic Core"
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


class BoundContentSource(Protocol):
    def submit(
        self, batch_id: str, items: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object]: ...

    def unsettled(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def ack(self, settlement_ref: str) -> Mapping[str, object]: ...


class ContentSourceServices(Protocol):
    def bind(self, source_id: str) -> BoundContentSource: ...


CONTENT_SOURCE = ServiceKey[ContentSourceServices]("content.source.v1")
inject = (TIMERS, CONTENT_SOURCE)


class FixtureSourceStore:
    """Persist the fake external feed and the source-owned cursor/deadline."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def seed(self, payloads: Sequence[Mapping[str, object]], now: datetime) -> None:
        with self._transaction() as connection:
            for payload in payloads:
                connection.execute(
                    "INSERT INTO feed(payload_json, observed_at) VALUES (?, ?)",
                    (
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                        _aware_utc(now),
                    ),
                )
            connection.execute(
                "UPDATE source_state SET next_due = ? WHERE singleton = 1",
                (_aware_utc(now),),
            )

    def poll(self) -> tuple[int, tuple[dict[str, object], ...]]:
        with self._transaction() as connection:
            cursor = int(
                connection.execute(
                    "SELECT cursor FROM source_state WHERE singleton = 1"
                ).fetchone()[0]
            )
            rows = connection.execute(
                """
                SELECT seq, payload_json, observed_at
                FROM feed WHERE seq > ? ORDER BY seq
                """,
                (cursor,),
            ).fetchall()
            return cursor, tuple(
                {
                    "item_id": f"event-{int(row['seq'])}",
                    "revision": "1",
                    "payload": json.loads(row["payload_json"]),
                    "not_before": row["observed_at"],
                    "requires_ack": True,
                }
                for row in rows
            )

    def commit_poll(self, cursor: int, count: int, next_due: datetime) -> None:
        with self._transaction() as connection:
            changed = connection.execute(
                """
                UPDATE source_state
                SET cursor = ?, next_due = ?, poll_count = poll_count + 1
                WHERE singleton = 1 AND cursor = ?
                """,
                (cursor + count, _aware_utc(next_due), cursor),
            )
            if changed.rowcount != 1:
                raise RuntimeError("fixture source cursor CAS failed")

    def state(self, now: datetime) -> dict[str, object]:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT cursor, next_due, poll_count FROM source_state WHERE singleton = 1"
            ).fetchone()
            next_due = row["next_due"] or _aware_utc(now)
            return {
                "cursor": int(row["cursor"]),
                "next_due": datetime.fromisoformat(next_due),
                "poll_count": int(row["poll_count"]),
            }

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS feed(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload_json TEXT NOT NULL,
                    observed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_state(
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    cursor INTEGER NOT NULL,
                    next_due TEXT,
                    poll_count INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO source_state VALUES(1, 0, NULL, 0);
                """)
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


class SourceRuntime:
    """Compose one-shot Timer waits with a submit-before-cursor source protocol."""

    def __init__(
        self,
        store: FixtureSourceStore,
        timers: PluginTimers,
        content: BoundContentSource,
        *,
        now: Callable[[], datetime] | None = None,
        after_submit: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self._timers = timers
        self._content = content
        self._now = now or (lambda: datetime.now(UTC))
        self._after_submit = after_submit
        self._handle: TimerHandle | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        """Recover the source-owned deadline and arm exactly one wait."""

        if self._closed:
            raise RuntimeError("content fixture source 已关闭")
        if self._handle is not None:
            return
        state = self.store.state(self._aware_now())
        self._arm(state["next_due"])

    async def close(self) -> None:
        """Cancel the owned wait/task without changing cursor or feed."""

        self._closed = True
        handle = self._handle
        task = self._task
        self._handle = None
        self._task = None
        if handle is not None:
            _ = await handle.cancel()
        if task is not None and task is not asyncio.current_task():
            _ = await asyncio.gather(task, return_exceptions=True)
        if handle is not None:
            await handle.cleanup()

    def _arm(self, deadline: object) -> None:
        if self._closed or self._handle is not None:
            return
        if not isinstance(deadline, datetime):
            raise RuntimeError("fixture source next_due 不是 datetime")
        handle = self._timers.schedule(deadline)
        self._handle = handle
        self._task = asyncio.create_task(
            self._wait_poll_rearm(handle), name="content-clock-source:poll"
        )

    async def _wait_poll_rearm(self, handle: TimerHandle) -> None:
        """Poll, commit Content, advance the cursor, then re-arm."""

        try:
            receipt = await handle.result()
            if receipt.status is TimerStatus.CANCELLED or self._closed:
                return

            # 1. Read the fake external feed without changing its cursor.
            cursor, items = self.store.poll()
            batch_id = f"poll:{cursor}:{cursor + len(items)}"

            # 2. Commit Content before any source-owned progress becomes visible.
            _ = self._content.submit(batch_id, items)
            if self._after_submit is not None:
                self._after_submit()

            # 3. Advance the source cursor/deadline only after the submit receipt.
            next_due = self._aware_now() + timedelta(minutes=5)
            self.store.commit_poll(cursor, len(items), next_due)
        finally:
            self._handle = None
            self._task = None
            await handle.cleanup()
        if not self._closed:
            state = self.store.state(self._aware_now())
            self._arm(state["next_due"])

    def _aware_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            raise ValueError("fixture source clock 必须带时区")
        return value.astimezone(UTC)


async def apply(ctx: Context, config: object) -> None:
    """Bind the fake source to formal runtime lifecycle only."""

    _ = config
    runtime = SourceRuntime(
        FixtureSourceStore(ctx.data_root / "source.sqlite3"),
        ctx.require(TIMERS),
        ctx.require(CONTENT_SOURCE).bind("clock-feed"),
    )

    def setup() -> object:
        return runtime.close

    _ = await ctx.effect(setup, label="content-clock-source-runtime")

    async def start(_event: object) -> None:
        await runtime.start()

    async def stop(_event: object) -> None:
        await runtime.close()

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)


def _aware_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("fixture source 时间必须带时区")
    return value.astimezone(UTC).isoformat()

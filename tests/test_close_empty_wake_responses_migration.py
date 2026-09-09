from __future__ import annotations

import json
import importlib.util
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest
import yoyo

from plugins.eventmail.plugin import EVENTMAIL_CONTENT_SOURCE
from plugins.wake.api import EVENTMAIL_WAKE
from plugins.wake.request import Phase, check_phase, retryable
from plugins.wake.state import WakeState
from session.message import ContentPart, Control, Input
from tests.test_wake_messages import application, request


def _migration(monkeypatch: pytest.MonkeyPatch, name: str = "20260909_01_close_empty_wake_responses") -> ModuleType:
    path = Path(__file__).parents[1] / "migrations/yoyo" / (name + ".py")
    spec = importlib.util.spec_from_file_location("close_empty_wake_responses_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(yoyo, "step", lambda callback: callback)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _legacy_attempts(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """用历史表约束构造旧 attempt，不让当前运行时再生产旧状态。"""
    migration = _migration(monkeypatch, "20260909_02_execution_failures")
    path = workspace / "plugin-data/wake-builtin/wake.sqlite3"
    with closing(sqlite3.connect(path)) as connection, connection:
        rows = connection.execute("SELECT * FROM wake_attempts").fetchall()
        connection.execute("DROP TABLE wake_attempts")
        connection.execute(migration._WAKE_OLD)
        connection.executemany("INSERT INTO wake_attempts VALUES(?,?,?,?,?,?,?,?,?)", rows)
        connection.execute("UPDATE wake_attempts SET outcome='delivery_unknown'")
        connection.execute("PRAGMA user_version=8")


def _raw_messages(path: Path) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute("SELECT * FROM messages ORDER BY session_key, seq").fetchall()


def test_migration_rejects_incomplete_sessions_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    state = WakeState(workspace / "plugin-data/wake-builtin/wake.sqlite3")
    state.initialize()
    now = datetime.now(timezone.utc)
    state.begin_attempt(
        attempt_id="b" * 32,
        timer_id="timer:legacy-empty",
        scheduled_for=now,
        fired_at=now,
    )
    state.finish_attempt(
        attempt_id="b" * 32,
        outcome="failed",
        owner="content",
        detail="ValueError: 模型没有产生内容或工具调用；空响应不是 quiet",
        completed_at=now,
    )
    _legacy_attempts(workspace, monkeypatch)
    sessions = workspace / "sessions.db"
    sessions.touch()

    with pytest.raises(RuntimeError, match="sessions.db schema 不完整"):
        _migration(monkeypatch).migrate(workspace)

    assert sessions.read_bytes() == b""
    assert not (workspace / "backups/close-empty-wake-responses").exists()


@pytest.mark.asyncio
async def test_migration_closes_only_exact_legacy_empty_response_and_source_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧空响应只追加失败终态，随后由现有 Source 结算原职责。"""

    workspace = tmp_path / "workspace"
    async with application(tmp_path) as (_, log, ctx, source, control):
        now = datetime.now(timezone.utc)
        producer = ctx.require(EVENTMAIL_CONTENT_SOURCE).bind("feed")
        producer.submit(
            "batch",
            [{
                "item_id": "one",
                "revision": "1",
                "not_before": now,
                "requires_ack": True,
                "payload": {"title": "keep this content"},
            }],
        )
        snapshot = ctx.require(EVENTMAIL_WAKE).snapshot(now)
        original = request(ctx, "content", now).model_copy(
            update={
                "snapshot_seq": snapshot["snapshot_seq"],
                "items": tuple(dict(item) for item in snapshot["items"]),
            }
        )
        source.accept(original)
        writer = log.writer(
            original.session_id,
            author="wake",
            source="wake",
            body_types=(Input,),
            content={"wake.phase": check_phase},
        )
        try:
            writer.append(
                original.phase_id("screen"),
                Input((ContentPart(
                    "wake.phase",
                    Phase(input_id=original.input_id, stage="screen").model_dump(mode="json"),
                ),)),
            )
        finally:
            writer.expire()

        source.state.begin_attempt(
            attempt_id=original.flow_id,
            timer_id="timer:legacy-empty",
            scheduled_for=now,
            fired_at=now,
        )
        detail = "ValueError: 模型没有产生内容或工具调用；空响应不是 quiet"
        source.state.finish_attempt(
            attempt_id=original.flow_id,
            outcome="failed",
            owner="content",
            detail=detail,
            completed_at=now,
        )
        _legacy_attempts(workspace, monkeypatch)
        migrate = _migration(monkeypatch).migrate
        pointer = log.owner("plugin:wake").read("flow:" + original.flow_id)
        assert pointer is not None
        with closing(sqlite3.connect(workspace / "sessions.db")) as connection, connection:
            connection.execute(
                "DELETE FROM owner_records WHERE owner=? AND key=?",
                ("plugin:wake", "flow:" + original.flow_id),
            )
        with pytest.raises(RuntimeError, match="缺少恢复指针"):
            migrate(workspace)
        assert not (workspace / "backups/close-empty-wake-responses").exists()
        log.owner("plugin:wake").transact(
            lambda tx: tx.save(
                "flow:" + original.flow_id,
                pointer.value,
                expected_version=None,
            )
        )
        before = _raw_messages(workspace / "sessions.db")

        assert migrate(workspace) == 1

        rows = log.reader(original.session_id).snapshot()
        assert _raw_messages(workspace / "sessions.db")[:-1] == before
        assert isinstance(rows[-1].body, Control)
        assert rows[-1].body.action == "failure"
        assert retryable(rows[-1]) is True
        with closing(sqlite3.connect(workspace / "plugin-data/wake-builtin/wake.sqlite3")) as database:
            assert database.execute("SELECT detail FROM wake_attempts WHERE attempt_id=?", (original.flow_id,)).fetchone() == (detail,)

        backups = list((workspace / "backups/close-empty-wake-responses").glob("*"))
        assert len(backups) == 1
        manifest = json.loads((backups[0] / "manifest.json").read_text())
        assert manifest["migration"] == "close-empty-wake-responses"
        with closing(sqlite3.connect(backups[0] / "sessions.db")) as saved:
            assert saved.execute(
                "SELECT * FROM messages ORDER BY session_key, seq"
            ).fetchall() == before

        assert migrate(workspace) == 0
        assert list((workspace / "backups/close-empty-wake-responses").glob("*")) == backups
        from agent.migrations.context import bind_migration_context
        with bind_migration_context(config_path=tmp_path / "config.toml", workspace=workspace):
            _migration(monkeypatch, "20260909_02_execution_failures").migrate_execution_failures(None)
        assert source.state.get_attempt(original.flow_id)["outcome"] == "failed"
        task = await source.start(original.flow_id)
        assert task is not None
        assert await task.join() == "deferred"
        assert control["calls"] == []
        assert source.pending() == ()

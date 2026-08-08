from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest
import yoyo

from agent.migrations.context import bind_migration_context


_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = (
    _PROJECT_ROOT
    / "migrations"
    / "yoyo"
    / "20260808_04_session_compaction_source_plan_digest.py"
)


def _load_migration():
    """Load the real migration callback without wrapping it in a Yoyo step."""

    spec = importlib.util.spec_from_file_location(
        "session_compaction_source_plan_digest_migration_under_test",
        _MIGRATION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载迁移: {_MIGRATION_PATH}")
    original_step = yoyo.step
    yoyo.step = lambda callback: callback  # type: ignore[assignment]
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        yoyo.step = original_step
    return module


def _create_legacy_db(path: Path, *, with_row: bool) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE sessions ("
            "key TEXT PRIMARY KEY, last_consolidated INTEGER NOT NULL)"
        )
        connection.execute(
            """
            CREATE TABLE session_compactions (
                session_key TEXT NOT NULL,
                generation INTEGER NOT NULL,
                parent_generation INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                trigger TEXT NOT NULL,
                summary_format_version INTEGER NOT NULL,
                summary TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                source_from_seq INTEGER NOT NULL,
                consolidated_through_seq INTEGER NOT NULL,
                source_message_ids_json TEXT NOT NULL,
                retained_tail_json TEXT NOT NULL,
                model_runtime_id TEXT NOT NULL,
                model TEXT NOT NULL,
                context_window INTEGER NOT NULL,
                threshold_tokens INTEGER NOT NULL,
                hard_input_tokens INTEGER NOT NULL,
                keep_recent_tokens INTEGER NOT NULL,
                tokens_before INTEGER NOT NULL,
                tokens_after INTEGER NOT NULL,
                summary_usage_json TEXT NOT NULL,
                invalidated_at TEXT,
                invalidated_reason TEXT,
                PRIMARY KEY (session_key, generation),
                UNIQUE (session_key, source_ref)
            )
            """
        )
        connection.execute(
            "CREATE INDEX idx_session_compactions_active ON "
            "session_compactions(session_key, invalidated_at, generation)"
        )
        if with_row:
            connection.execute(
                """
                INSERT INTO session_compactions (
                    session_key, generation, parent_generation, created_at,
                    trigger, summary_format_version, summary, source_ref,
                    source_from_seq, consolidated_through_seq,
                    source_message_ids_json, retained_tail_json,
                    model_runtime_id, model, context_window, threshold_tokens,
                    hard_input_tokens, keep_recent_tokens, tokens_before,
                    tokens_after, summary_usage_json
                ) VALUES (
                    'session', 1, 0, '2026-08-08T00:00:00+00:00',
                    'soft_limit', 1, 'summary', 'source:1',
                    0, 0, '["m0"]', '[]', 'runtime', 'model', 100,
                    74, 90, 20, 80, 40, '{}'
                )
                """
            )
        connection.commit()
    finally:
        connection.close()


def _run(module, workspace: Path) -> None:
    config = workspace.parent / "config.toml"
    config.write_text("", encoding="utf-8")
    with bind_migration_context(config_path=config, workspace=workspace):
        module.add_source_plan_digest(None)


def _backup_root(workspace: Path) -> Path:
    roots = sorted((workspace / "backups/session-compaction-source-plan-digest").iterdir())
    assert len(roots) == 1
    return roots[0]


def test_empty_legacy_ledger_is_rebuilt_to_final_digest_schema(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = workspace / "sessions.db"
    _create_legacy_db(sessions, with_row=False)

    _run(module, workspace)

    connection = sqlite3.connect(sessions)
    try:
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(session_compactions)")
        ]
        assert "source_plan_digest" in columns
        assert connection.execute(
            "SELECT COUNT(1) FROM session_compactions"
        ).fetchone() == (0,)
        table_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='session_compactions'"
        ).fetchone()[0]
        assert "source_plan_digest TEXT NOT NULL" in table_sql
        assert "length(source_plan_digest) = 64" in table_sql
        assert {
            row[1] for row in connection.execute("PRAGMA index_list(session_compactions)")
        } >= {"idx_session_compactions_active"}
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        connection.close()

    manifest = json.loads((_backup_root(workspace) / "manifest.json").read_text())
    assert manifest["sqlite_integrity"] == "ok"


def test_legacy_rows_without_digest_fail_loud_and_preserve_backup(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = workspace / "sessions.db"
    _create_legacy_db(sessions, with_row=True)
    before = sqlite3.connect(sessions)
    try:
        before_row = before.execute(
            "SELECT source_ref, summary FROM session_compactions"
        ).fetchall()
    finally:
        before.close()

    with pytest.raises(RuntimeError, match="缺少 source_plan_digest 且已有数据"):
        _run(module, workspace)

    connection = sqlite3.connect(sessions)
    try:
        assert connection.execute(
            "SELECT source_ref, summary FROM session_compactions"
        ).fetchall() == before_row
        assert "source_plan_digest" not in {
            row[1] for row in connection.execute("PRAGMA table_info(session_compactions)")
        }
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        connection.close()

    backup_root = _backup_root(workspace)
    manifest = json.loads((backup_root / "manifest.json").read_text())
    assert manifest["sqlite_integrity"] == "ok"
    archived = sqlite3.connect(backup_root / manifest["backup"])
    try:
        assert archived.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert archived.execute(
            "SELECT source_ref, summary FROM session_compactions"
        ).fetchall() == before_row
    finally:
        archived.close()


def test_existing_final_schema_rejects_invalid_digest_rows(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = workspace / "sessions.db"
    _create_legacy_db(sessions, with_row=False)
    _run(module, workspace)

    connection = sqlite3.connect(sessions)
    try:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            INSERT INTO session_compactions (
                session_key, generation, parent_generation, created_at,
                trigger, summary_format_version, summary, source_ref,
                source_plan_digest, source_from_seq, consolidated_through_seq,
                source_message_ids_json, retained_tail_json,
                model_runtime_id, model, context_window, threshold_tokens,
                hard_input_tokens, keep_recent_tokens, tokens_before,
                tokens_after, summary_usage_json
            ) VALUES (
                'session', 1, 0, '2026-08-08T00:00:00+00:00',
                'soft_limit', 1, 'summary', 'source:1', '', 0, 0,
                '["m0"]', '[]', 'runtime', 'model', 100, 74, 90, 20,
                80, 40, '{}'
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    connection = sqlite3.connect(sessions)
    try:
        with pytest.raises(RuntimeError, match="source_plan_digest 非法"):
            module._validate_digest_values(connection)
    finally:
        connection.close()

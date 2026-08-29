from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest
import yoyo

from agent.migrations.context import bind_migration_context

_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = (
    _PROJECT_ROOT
    / "migrations/yoyo/20260829_01_backfill_plugin_programmatic_effects.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location(
        "backfill_plugin_programmatic_effects_under_test",
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


def _create_database(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript("""
            CREATE TABLE sessions(key TEXT PRIMARY KEY, metadata TEXT NOT NULL);
            CREATE TABLE messages(
                id TEXT PRIMARY KEY, session_key TEXT NOT NULL, seq INTEGER NOT NULL,
                extra TEXT NOT NULL
            );
            CREATE TABLE turns(
                id TEXT PRIMARY KEY, session_key TEXT NOT NULL, input_json TEXT NOT NULL
            );
        """)


def _insert_turn(
    connection: sqlite3.Connection,
    session_key: str,
    metadata: dict[str, object],
    *,
    extra: dict[str, object] | None = None,
    inbound: dict[str, object] | None = None,
) -> None:
    connection.execute(
        "INSERT INTO sessions VALUES (?, ?)",
        (session_key, json.dumps(metadata)),
    )
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, 0, ?)",
        (f"message:{session_key}", session_key, json.dumps(extra or {})),
    )
    connection.execute(
        "INSERT INTO turns VALUES (?, ?, ?)",
        (
            f"turn:{session_key}",
            session_key,
            json.dumps(
                {
                    "input": "work",
                    "metadata": {"inboundMetadata": inbound or {}},
                }
            ),
        ),
    )


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.backfill_plugin_programmatic_effects(None)


def test_repairs_only_core_owned_plugin_programmatic_turns(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "sessions.db"
    _create_database(database)
    provenance = {
        "programmatic": True,
        "plugin_id": "github-watch@github",
        "job_name": "poll",
        "generation_id": "generation-1",
        "snapshot_id": "snapshot-1",
    }
    with closing(sqlite3.connect(database)) as connection, connection:
        _insert_turn(connection, "programmatic:plugin", provenance)
        _insert_turn(connection, "programmatic:direct", {"programmatic": True})

    _run(module, tmp_path / "config.toml", workspace)
    _run(module, tmp_path / "config.toml", workspace)

    with closing(sqlite3.connect(database)) as connection:
        plugin_extra = json.loads(
            connection.execute(
                "SELECT extra FROM messages WHERE id='message:programmatic:plugin'"
            ).fetchone()[0]
        )
        plugin_input = json.loads(
            connection.execute(
                "SELECT input_json FROM turns WHERE id='turn:programmatic:plugin'"
            ).fetchone()[0]
        )
        direct_extra = json.loads(
            connection.execute(
                "SELECT extra FROM messages WHERE id='message:programmatic:direct'"
            ).fetchone()[0]
        )
    effect = {"effects": {"post_commit": "suppress"}}
    assert plugin_extra == effect
    assert plugin_input["metadata"]["inboundMetadata"] == effect
    assert direct_extra == {}
    backups = list(
        (workspace / "backups" / "backfill-plugin-programmatic-effects").glob("*")
    )
    assert len(backups) == 1


def test_rejects_contradictory_allow_before_backup(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "sessions.db"
    _create_database(database)
    provenance = {
        "programmatic": True,
        "plugin_id": "plugin",
        "job_name": "job",
        "generation_id": "generation",
        "snapshot_id": "snapshot",
    }
    allow: dict[str, object] = {"effects": {"post_commit": "allow"}}
    with closing(sqlite3.connect(database)) as connection, connection:
        _insert_turn(connection, "programmatic:plugin", provenance, extra=allow)

    with pytest.raises(RuntimeError, match="suppress 合同冲突"):
        _run(module, tmp_path / "config.toml", workspace)
    assert not (workspace / "backups").exists()

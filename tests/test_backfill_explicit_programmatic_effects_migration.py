from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from agent.migrations.context import bind_migration_context
from tests.legacy_migration_loader import load_migration_namespace

_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = (
    _PROJECT_ROOT
    / "plugins/legacy_upgrade/legacy_upgrade_migrations/20260829_02_backfill_explicit_programmatic_effects.py"
)


def _load_migration():
    return load_migration_namespace("20260829_02_backfill_explicit_programmatic_effects")


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


def _insert(
    connection: sqlite3.Connection,
    session_key: str,
    session_metadata: dict[str, object],
    turn_metadata: dict[str, object],
) -> None:
    turn_id = f"turn:{session_key}"
    connection.execute(
        "INSERT INTO sessions VALUES (?, ?)",
        (session_key, json.dumps(session_metadata)),
    )
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, 0, ?)",
        (f"message:{session_key}", session_key, json.dumps({"control_turn_id": turn_id})),
    )
    connection.execute(
        "INSERT INTO turns VALUES (?, ?, ?)",
        (turn_id, session_key, json.dumps({"input": "work", "metadata": turn_metadata})),
    )


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.backfill_explicit_programmatic_effects(None)


def test_repairs_only_explicit_suppress_and_preserves_ambiguous_legacy_allow(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "sessions.db"
    _create_database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        _insert(
            connection,
            "programmatic:explicit",
            {},
            {"effects": {"post_commit": "suppress"}},
        )
        _insert(connection, "programmatic:legacy-persist", {}, {})
        _insert(
            connection,
            "programmatic:new-persist",
            {"effects": {"post_commit": "allow"}},
            {},
        )

    _run(module, tmp_path / "config.toml", workspace)
    _run(module, tmp_path / "config.toml", workspace)

    with closing(sqlite3.connect(database)) as connection:
        extras = {
            row[0]: json.loads(row[1])
            for row in connection.execute(
                "SELECT session_key, extra FROM messages ORDER BY session_key"
            )
        }
        explicit_input = json.loads(
            connection.execute(
                "SELECT input_json FROM turns "
                "WHERE id='turn:programmatic:explicit'"
            ).fetchone()[0]
        )
    assert extras["programmatic:explicit"]["effects"] == {
        "post_commit": "suppress"
    }
    assert "effects" not in extras["programmatic:legacy-persist"]
    assert "effects" not in extras["programmatic:new-persist"]
    assert explicit_input["metadata"]["inboundMetadata"] == {
        "effects": {"post_commit": "suppress"}
    }
    backups = list(
        (workspace / "backups" / "backfill-explicit-programmatic-effects").glob("*")
    )
    assert len(backups) == 1

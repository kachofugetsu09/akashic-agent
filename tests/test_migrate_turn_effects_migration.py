import hashlib
import importlib.util
import json
import sqlite3
import stat
import sys
from contextlib import closing
from pathlib import Path

import pytest
import yoyo

from agent.migrations.context import bind_migration_context

_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = _PROJECT_ROOT / "migrations/yoyo/20260826_01_migrate_turn_effects.py"


def _load_migration():
    """Load the migration callback without wrapping it in Yoyo."""

    spec = importlib.util.spec_from_file_location(
        "migrate_turn_effects_under_test",
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
            CREATE TABLE sessions (
                key TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_consolidated INTEGER NOT NULL DEFAULT 0,
                metadata TEXT
            );
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_chain TEXT,
                extra TEXT,
                ts TEXT NOT NULL,
                UNIQUE(session_key, seq)
            );
            CREATE TABLE turns (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL,
                items_json TEXT NOT NULL,
                usage_json TEXT,
                error_json TEXT,
                final_response TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE TABLE interaction_memory_reconciliations (
                reconciliation_id TEXT PRIMARY KEY,
                control_turn_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                message_ids_json TEXT NOT NULL,
                owner TEXT NOT NULL,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                last_error TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
        """)


def _insert_session(
    connection: sqlite3.Connection,
    key: str,
    metadata: dict[str, object],
) -> None:
    connection.execute(
        "INSERT INTO sessions VALUES (?, 'created', 'updated', 0, ?)",
        (key, json.dumps(metadata)),
    )


def _insert_turn(
    connection: sqlite3.Connection,
    *,
    session_key: str,
    sequence: int,
    extra: dict[str, object] | None = None,
    turn_metadata: dict[str, object] | None = None,
) -> None:
    message_id = f"message:{session_key}:{sequence}"
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, ?, 'user', 'content', NULL, ?, 'ts')",
        (message_id, session_key, sequence, json.dumps(extra or {})),
    )
    turn_id = f"turn:{session_key}:{sequence}"
    connection.execute(
        "INSERT INTO turns VALUES (?, ?, 'completed', ?, '[]', NULL, NULL, 'ok', "
        "'created', 'started', 'completed')",
        (
            turn_id,
            session_key,
            json.dumps(
                {"input": "content", "metadata": turn_metadata or {}},
                ensure_ascii=False,
            ),
        ),
    )


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.migrate_turn_effects(None)


def _json_cell(
    database: Path,
    table: str,
    column: str,
    identity: str,
) -> dict[str, object]:
    key = "key" if table == "sessions" else "id"
    with closing(sqlite3.connect(database)) as connection, connection:
        row = connection.execute(
            f"SELECT {column} FROM {table} WHERE {key} = ?",
            (identity,),
        ).fetchone()
    assert row is not None
    payload = json.loads(row[0])
    assert isinstance(payload, dict)
    return payload


def _effect() -> dict[str, object]:
    return {"effects": {"post_commit": "suppress"}}


def test_migrates_session_message_scheduler_and_turn_replay_semantics(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "sessions.db"
    _create_database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        _insert_session(connection, "normal", {})
        _insert_session(connection, "marked", {"skip_post_memory": True, "keep": 1})
        _insert_session(connection, "scheduler:job", {})
        _insert_session(
            connection,
            "plugin-owned",
            {"skip_post_memory": "plugin-value"},
        )
        _insert_turn(
            connection,
            session_key="normal",
            sequence=0,
            extra={"skip_post_memory": True, "keep": "message"},
            turn_metadata={"skip_post_memory": True, "keep": "turn"},
        )
        _insert_turn(connection, session_key="marked", sequence=0)
        _insert_turn(connection, session_key="scheduler:job", sequence=0)
        _insert_turn(connection, session_key="plugin-owned", sequence=0)

    config = tmp_path / "config.toml"
    config.write_text("current = true\n", encoding="utf-8")
    _run(module, config, workspace)
    _run(module, config, workspace)

    assert _json_cell(database, "sessions", "metadata", "marked") == {"keep": 1}
    assert _json_cell(database, "sessions", "metadata", "plugin-owned") == {
        "skip_post_memory": "plugin-value"
    }
    normal_message = _json_cell(
        database,
        "messages",
        "extra",
        "message:normal:0",
    )
    assert normal_message == {"keep": "message", **_effect()}
    normal_turn = _json_cell(
        database,
        "turns",
        "input_json",
        "turn:normal:0",
    )
    assert normal_turn["metadata"] == {"keep": "turn", **_effect()}
    for session_key in ("marked", "scheduler:job"):
        assert (
            _json_cell(
                database,
                "messages",
                "extra",
                f"message:{session_key}:0",
            )
            == _effect()
        )
        turn = _json_cell(
            database,
            "turns",
            "input_json",
            f"turn:{session_key}:0",
        )
        assert turn["metadata"] == _effect()

    roots = sorted((workspace / "backups/migrate-turn-effects").iterdir())
    assert len(roots) == 1
    backup = roots[0] / "sessions.db"
    manifest = json.loads((roots[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == hashlib.sha256(backup.read_bytes()).hexdigest()
    assert _json_cell(backup, "sessions", "metadata", "marked") == {
        "skip_post_memory": True,
        "keep": 1,
    }
    assert stat.S_IMODE(roots[0].stat().st_mode) == 0o700
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    with closing(sqlite3.connect(database)) as connection, connection:
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='interaction_memory_reconciliations'"
            ).fetchone()
            is None
        )
    with closing(sqlite3.connect(backup)) as connection, connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='interaction_memory_reconciliations'"
        ).fetchone() == (1,)


@pytest.mark.parametrize(
    "extra",
    (
        {"skip_post_memory": "true"},
        {
            "skip_post_memory": True,
            "effects": {"post_commit": "allow"},
        },
        {"skip_post_memory": True, "effects": "suppress"},
    ),
)
def test_invalid_or_conflicting_message_semantics_fail_before_backup(
    tmp_path: Path,
    extra: dict[str, object],
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "sessions.db"
    _create_database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        _insert_session(connection, "normal", {})
        _insert_turn(connection, session_key="normal", sequence=0, extra=extra)
    original = database.read_bytes()
    config = tmp_path / "config.toml"

    with pytest.raises((ValueError, RuntimeError)):
        _run(module, config, workspace)

    assert database.read_bytes() == original
    assert not (workspace / "backups").exists()


def test_failed_transaction_rolls_back_to_original_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "sessions.db"
    _create_database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        _insert_session(connection, "marked", {"skip_post_memory": True})
        _insert_turn(connection, session_key="marked", sequence=0)
    original = database.read_bytes()
    config = tmp_path / "config.toml"
    real_apply = module._apply_rewrites

    def fail_after_updates(connection, rewrites):
        real_apply(connection, rewrites)
        raise RuntimeError("forced transaction failure")

    monkeypatch.setattr(module, "_apply_rewrites", fail_after_updates)
    with pytest.raises(RuntimeError, match="forced transaction failure"):
        _run(module, config, workspace)

    assert database.read_bytes() == original
    assert len(list((workspace / "backups/migrate-turn-effects").iterdir())) == 1

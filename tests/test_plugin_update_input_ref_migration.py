"""通过真实 MigrationRunner/Yoyo 边界验证 input_ref 的单列迁移源。"""
from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from agent.migrations.runner import MigrationRunner
from agent.plugins.artifacts import ArtifactPointer
from agent.plugins.reload_journal import ReloadJournal


MIGRATION_ID = "20260921_01_plugin_update_input_ref"
OLD_TABLE = """CREATE TABLE plugin_updates (
        update_id TEXT PRIMARY KEY,
        plugin_id TEXT NOT NULL,
        plugin_base TEXT NOT NULL,
        previous_pointers_json TEXT,
        candidate_pointer TEXT NOT NULL,
        previous_enabled INTEGER CHECK (previous_enabled IN (0, 1)),
        phase TEXT NOT NULL CHECK (phase IN ('armed', 'committed', 'rolled_back')),
        reload_tx_id TEXT UNIQUE REFERENCES reload_transactions(tx_id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error TEXT NOT NULL
    )"""
INDEX = """CREATE UNIQUE INDEX plugin_update_active
        ON plugin_updates(plugin_id) WHERE phase='armed'"""


def _repo(tmp_path: Path, *, fail_before_alter: bool = False) -> Path:
    repo = tmp_path / "repo"
    (repo / "migrations/core").mkdir(parents=True)
    migration = repo / f"migrations/core/{MIGRATION_ID}.py"
    source = Path(__file__).parents[1] / "migrations/core/20260921_01_plugin_update_input_ref.py"
    shutil.copyfile(source, migration)
    if fail_before_alter:
        text = migration.read_text(encoding="utf-8")
        needle = '        conn.execute("ALTER TABLE plugin_updates ADD COLUMN input_ref TEXT")'
        assert text.count(needle) == 1
        migration.write_text(
            text.replace(needle, '        raise RuntimeError("controlled migration failure")\n' + needle),
            encoding="utf-8",
        )
    (repo / "migrations/catalog.toml").write_text(
        "schema_version = 1\nmigrations = []\n", encoding="utf-8",
    )
    return repo


def _runner(tmp_path: Path, repo: Path) -> MigrationRunner:
    return MigrationRunner(
        repo_root=repo,
        config_path=tmp_path / "config.toml",
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "plugin-cache",
        migration_catalog=repo / "migrations/catalog.toml",
    )


def _old_database(workspace: Path) -> Path:
    database = workspace / "runtime/plugin-reloads.sqlite3"
    database.parent.mkdir(parents=True)
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.executescript("""
            CREATE TABLE reload_transactions (
                tx_id TEXT PRIMARY KEY,
                plugin_id TEXT NOT NULL,
                base_snapshot_id TEXT,
                candidate_snapshot_id TEXT,
                base_generation_id TEXT,
                generation_id TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                config_revision TEXT NOT NULL,
                phase TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT NOT NULL,
                formal_effects_json TEXT NOT NULL DEFAULT '[]',
                failure_resource TEXT,
                recovery_action TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                runtime_owner_boot_id TEXT,
                base_artifact_pointer TEXT,
                candidate_artifact_pointer TEXT,
                recovery_target TEXT
            );
            CREATE TABLE reload_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_id TEXT NOT NULL REFERENCES reload_transactions(tx_id),
                phase TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX idx_reload_transactions_phase ON reload_transactions(phase);
            CREATE INDEX idx_reload_events_tx ON reload_events(tx_id, sequence);
        """)
        conn.execute(OLD_TABLE)
        conn.execute(INDEX)
        conn.execute(
            "INSERT INTO reload_transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("tx1", "plug@market", None, None, None, "g1", "src", "cfg", "prepared",
             "created", "updated", "", "[]", None, None, 0, None, None, "candidate", "base"),
        )
        conn.execute(
            "INSERT INTO reload_events(tx_id,phase,details_json,created_at) VALUES (?, ?, ?, ?)",
            ("tx1", "prepared", "{}", "event"),
        )
        conn.execute(
            "INSERT INTO plugin_updates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("u1", "plug@market", "/plugins/cache/plug", None, "a1", 1,
             "committed", "tx1", "created", "updated", ""),
        )
    return database


def _shape(path: Path) -> tuple[tuple[object, ...], ...]:
    with closing(sqlite3.connect(path)) as conn:
        return tuple(tuple(row) for row in conn.execute("PRAGMA table_info(plugin_updates)"))


def test_runner_uses_connection_callback_and_preserves_old_rows_fk_and_indexes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    workspace = tmp_path / "workspace"
    database = _old_database(workspace)
    outcome = _runner(tmp_path, repo).run()
    assert outcome.migrations == (MIGRATION_ID,)
    with closing(sqlite3.connect(database)) as conn, conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(plugin_updates)")]
        assert columns[-1] == "input_ref"
        assert conn.execute(
            "SELECT update_id,input_ref,plugin_id,plugin_base,candidate_pointer,error "
            "FROM plugin_updates"
        ).fetchone() == ("u1", None, "plug@market", "/plugins/cache/plug", "a1", "")
        assert conn.execute("SELECT COUNT(*) FROM reload_transactions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM reload_events").fetchone() == (1,)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='plugin_update_active'"
        ).fetchone() == ("plugin_update_active",)
    backups = list(workspace.glob("runtime/plugin-reloads.sqlite3.before-input-ref.*.bak"))
    assert len(backups) == 1
    journal = ReloadJournal(workspace)
    assert journal.update("u1").input_ref is None

    second = _runner(tmp_path, repo).run()
    assert second.state == "current"
    assert len(list(workspace.glob("runtime/plugin-reloads.sqlite3.before-input-ref.*.bak"))) == 1


def test_failed_upgrade_keeps_backup_and_can_retry_with_a_new_backup(tmp_path: Path) -> None:
    repo = _repo(tmp_path, fail_before_alter=True)
    workspace = tmp_path / "workspace"
    database = _old_database(workspace)
    before = database.read_bytes()
    with pytest.raises(RuntimeError, match="Yoyo 迁移失败") as failure:
        _runner(tmp_path, repo).run()
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert database.read_bytes() == before
    assert len(list(workspace.glob("runtime/plugin-reloads.sqlite3.before-input-ref.*.bak"))) == 1

    shutil.copyfile(
        Path(__file__).parents[1] / "migrations/core/20260921_01_plugin_update_input_ref.py",
        repo / f"migrations/core/{MIGRATION_ID}.py",
    )
    outcome = _runner(tmp_path, repo).run()
    assert outcome.migrations == (MIGRATION_ID,)
    assert len(list(workspace.glob("runtime/plugin-reloads.sqlite3.before-input-ref.*.bak"))) == 2


def test_fresh_runner_does_not_create_plugin_database(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    outcome = _runner(tmp_path, repo).run()
    assert outcome.migrations == (MIGRATION_ID,)
    assert not (tmp_path / "workspace/runtime/plugin-reloads.sqlite3").exists()


def test_old_and_unknown_runtime_schema_are_rejected_without_startup_writes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    database = _old_database(workspace)
    before = database.read_bytes()
    with pytest.raises(RuntimeError, match="旧 plugin_updates schema"):
        ReloadJournal(workspace)
    assert database.read_bytes() == before

    unknown = tmp_path / "unknown"
    path = unknown / "runtime/plugin-reloads.sqlite3"
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE plugin_updates (update_id TEXT PRIMARY KEY)")
        conn.execute("CREATE UNIQUE INDEX plugin_update_active ON plugin_updates(update_id)")
    before_unknown = path.read_bytes()
    with pytest.raises(RuntimeError, match="schema"):
        ReloadJournal(unknown)
    assert path.read_bytes() == before_unknown


def test_fresh_and_upgraded_plugin_update_shapes_have_same_known_columns(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    workspace = tmp_path / "workspace"
    upgraded = _old_database(workspace)
    _runner(tmp_path, repo).run()
    fresh_workspace = tmp_path / "fresh"
    ReloadJournal(fresh_workspace)
    assert [row[1] for row in _shape(upgraded)] == [row[1] for row in _shape(
        fresh_workspace / "runtime/plugin-reloads.sqlite3"
    )]


def test_journal_input_ref_has_one_writer_and_duplicate_install_is_not_a_journal_update(tmp_path: Path) -> None:
    journal = ReloadJournal(tmp_path / "workspace")
    journal.arm_update(
        update_id="u1", plugin_id="plug@market",
        plugin_base=tmp_path / "home/cache/market/plug",
        previous=None, candidate=ArtifactPointer(".artifacts/a"), previous_enabled=True,
    )
    journal.set_input_ref("u1", "archive-a")
    journal.set_input_ref("u1", "archive-a")
    with pytest.raises(RuntimeError, match="不能改变"):
        journal.set_input_ref("u1", "archive-b")
    assert journal.update("u1").input_ref == "archive-a"

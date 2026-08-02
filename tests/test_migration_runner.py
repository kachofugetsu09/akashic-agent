from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from agent.migrations.runner import MigrationRunner
from bootstrap.workspace_lock import WorkspaceInstanceLock


_PROJECT_ROOT = Path(__file__).parents[1]
_ORIGIN_ID = "20260802_01_yoyo_origin"


def _runner(root: Path, *, repo_root: Path = _PROJECT_ROOT) -> MigrationRunner:
    return MigrationRunner(
        repo_root=repo_root,
        config_path=root / "config.toml",
        workspace=root / "workspace",
    )


def _applied_ids(ledger: Path) -> list[str]:
    connection = sqlite3.connect(ledger)
    try:
        rows = connection.execute(
            "SELECT migration_id FROM _yoyo_migration ORDER BY migration_id"
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def test_origin_removes_legacy_state_without_touching_business_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    config = root / "config.toml"
    config.write_bytes(b"current = true\n")
    sessions = workspace / "sessions.db"
    sessions.write_bytes(b"session-bytes")
    memory = workspace / "memory/MEMORY.md"
    memory.parent.mkdir()
    memory.write_bytes(b"memory-bytes")

    cursor = root / "config.toml.migration-cursor"
    lock = root / "config.toml.migration-lock"
    backups = root / "config.toml.migration-backups"
    cursor.write_text("retired\n", encoding="utf-8")
    lock.write_text("123\n", encoding="utf-8")
    backups.mkdir()
    (backups / "old.bak").write_bytes(b"backup")

    outcome = _runner(root).run()

    assert outcome.state == "migrated"
    assert outcome.migrations == (_ORIGIN_ID,)
    assert not cursor.exists()
    assert not lock.exists()
    assert not backups.exists()
    assert config.read_bytes() == b"current = true\n"
    assert sessions.read_bytes() == b"session-bytes"
    assert memory.read_bytes() == b"memory-bytes"
    assert _applied_ids(workspace / "migrations.sqlite3") == [_ORIGIN_ID]


def test_origin_is_a_noop_when_legacy_state_is_absent(tmp_path: Path) -> None:
    runner = _runner(tmp_path / "state")

    first = runner.run()
    second = runner.run()

    assert first.state == "migrated"
    assert first.migrations == (_ORIGIN_ID,)
    assert second.state == "current"
    assert second.migrations == ()


def test_runner_supplies_yoyo_identity_without_os_username(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.delenv(key, raising=False)

    def get_user() -> str:
        username = os.environ.get("USER")
        if not username:
            raise OSError("No username set in the environment")
        return username

    monkeypatch.setattr("yoyo.backends.base.getpass.getuser", get_user)

    outcome = _runner(tmp_path / "state").run()

    assert outcome.migrations == (_ORIGIN_ID,)
    assert "USER" not in os.environ


def test_origin_failure_is_not_marked_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    backups = root / "config.toml.migration-backups"
    backups.mkdir(parents=True)
    runner = _runner(root)

    def fail(_path: str | Path) -> None:
        raise PermissionError("forced cleanup failure")

    monkeypatch.setattr("shutil.rmtree", fail)
    with pytest.raises(RuntimeError, match="forced cleanup failure"):
        runner.run()

    assert backups.exists()
    assert _applied_ids(runner.ledger_path) == []

    monkeypatch.undo()
    assert runner.run().migrations == (_ORIGIN_ID,)
    assert not backups.exists()


def test_workspace_lock_prevents_concurrent_migration(tmp_path: Path) -> None:
    root = tmp_path / "state"
    runner = _runner(root)
    lock = WorkspaceInstanceLock(runner.workspace)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="workspace 已由其他 runtime 占用"):
            runner.run()
    finally:
        lock.release()

    assert not runner.ledger_path.exists()


def test_catalog_ignores_archived_git_cursor_migrations(tmp_path: Path) -> None:
    runner = _runner(tmp_path / "state")

    outcome = runner.run()

    assert outcome.migrations == (_ORIGIN_ID,)
    assert _applied_ids(runner.ledger_path) == [_ORIGIN_ID]


def test_new_branch_migration_is_applied_even_after_sibling_ran(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    catalog = repo / "migrations/yoyo"
    catalog.mkdir(parents=True)
    root = tmp_path / "state"
    workspace_literal = repr(str(root / "workspace"))

    def write_migration(name: str, depends: str) -> None:
        (catalog / f"{name}.py").write_text(
            "from pathlib import Path\n"
            "from yoyo import step\n"
            f"__depends__ = {depends}\n"
            f"def apply(_connection):\n"
            f"    marker = Path({workspace_literal}) / 'order.log'\n"
            "    marker.parent.mkdir(parents=True, exist_ok=True)\n"
            f"    with marker.open('a', encoding='utf-8') as stream:\n"
            f"        stream.write({name!r} + '\\n')\n"
            "steps = [step(apply)]\n",
            encoding="utf-8",
        )

    write_migration("base", "set()")
    runner = _runner(root, repo_root=repo)
    assert runner.run().migrations == ("base",)

    write_migration("bob", "{'base'}")
    assert runner.run().migrations == ("bob",)

    write_migration("alice", "{'base'}")
    assert runner.run().migrations == ("alice",)
    assert (root / "workspace/order.log").read_text() == "base\nbob\nalice\n"


def test_ledger_supports_workspace_path_with_uri_characters(tmp_path: Path) -> None:
    root = tmp_path / "state with # and ?"

    outcome = _runner(root).run()

    assert outcome.migrations == (_ORIGIN_ID,)
    assert (root / "workspace/migrations.sqlite3").is_file()

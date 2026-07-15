import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from bootstrap.workspace_lock import WorkspaceInstanceLock
from scripts import migrate_plugin_data as migration_module
from scripts.migrate_plugin_data import migrate_plugin_data


def test_migrate_plugin_data_copies_files_and_live_sqlite(tmp_path: Path) -> None:
    plugins_home = tmp_path / "plugins-home"
    source = plugins_home / "data" / "feed-github"
    source.mkdir(parents=True)
    (source / "config.local.toml").write_text("enabled = true\n", encoding="utf-8")
    database = source / "feed.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE items (value TEXT NOT NULL)")
    connection.execute("INSERT INTO items VALUES ('kept')")
    connection.commit()

    workspace = tmp_path / "workspace"
    try:
        migrated = migrate_plugin_data(
            workspace=workspace,
            plugins_home=plugins_home,
        )
    finally:
        connection.close()

    target = workspace / "plugin-data" / "feed-github"
    assert migrated == [target]
    assert (target / "config.local.toml").read_text(encoding="utf-8") == "enabled = true\n"
    with closing(sqlite3.connect(target / "feed.sqlite3")) as copied:
        assert copied.execute("SELECT value FROM items").fetchone() == ("kept",)
        assert copied.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_migrate_plugin_data_refuses_existing_target(tmp_path: Path) -> None:
    plugins_home = tmp_path / "plugins-home"
    source = plugins_home / "data" / "feed-github"
    source.mkdir(parents=True)
    (source / "state.json").write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    target = workspace / "plugin-data" / "feed-github"
    target.mkdir(parents=True)
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        migrate_plugin_data(workspace=workspace, plugins_home=plugins_home)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_migrate_plugin_data_rolls_back_published_targets_on_failure(
    tmp_path: Path,
) -> None:
    plugins_home = tmp_path / "plugins-home"
    first = plugins_home / "data" / "a-github"
    second = plugins_home / "data" / "b-github"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "state.json").write_text("{}\n", encoding="utf-8")
    (second / "escape").symlink_to(tmp_path / "outside")
    workspace = tmp_path / "workspace"

    with pytest.raises(ValueError, match="符号链接"):
        migrate_plugin_data(workspace=workspace, plugins_home=plugins_home)

    assert not (workspace / "plugin-data").exists()


def test_migrate_plugin_data_rejects_symlink_target_root(tmp_path: Path) -> None:
    plugins_home = tmp_path / "plugins-home"
    source = plugins_home / "data" / "feed-github"
    source.mkdir(parents=True)
    (source / "state.json").write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "plugin-data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="符号链接"):
        migrate_plugin_data(workspace=workspace, plugins_home=plugins_home)

    assert list(outside.iterdir()) == []


def test_migrate_plugin_data_requires_idle_workspace(tmp_path: Path) -> None:
    plugins_home = tmp_path / "plugins-home"
    source = plugins_home / "data" / "feed-github"
    source.mkdir(parents=True)
    (source / "state.json").write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    lock = WorkspaceInstanceLock(workspace)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="其他 runtime 占用"):
            migrate_plugin_data(workspace=workspace, plugins_home=plugins_home)
    finally:
        lock.release()

    assert not (workspace / "plugin-data").exists()


def test_migrate_plugin_data_publish_failure_never_exposes_partial_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugins_home = tmp_path / "plugins-home"
    for name in ("calendar-github", "feed-github"):
        source = plugins_home / "data" / name
        source.mkdir(parents=True)
        (source / "state.json").write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"

    def fail_publish(_source: Path, _destination: Path) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(migration_module.os, "replace", fail_publish)

    with pytest.raises(OSError, match="publish failed"):
        migrate_plugin_data(workspace=workspace, plugins_home=plugins_home)

    assert not (workspace / "plugin-data").exists()
    assert list(workspace.glob(".plugin-data-migrate-*")) == []

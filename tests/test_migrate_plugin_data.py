import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from bootstrap.workspace_lock import WorkspaceInstanceLock
from scripts import migrate_plugin_data as migration_module
from scripts.migrate_plugin_data import migrate_plugin_data, replace_plugin_data


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
    assert database.with_name(f"{database.name}-wal").exists()
    assert database.with_name(f"{database.name}-shm").exists()

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
    assert not (target / "feed.sqlite3-wal").exists()
    assert not (target / "feed.sqlite3-shm").exists()
    assert not (target / "feed.sqlite3-journal").exists()


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


def test_replace_plugin_data_preserves_unselected_and_backs_up_target(
    tmp_path: Path,
) -> None:
    plugins_home = tmp_path / "plugins-home"
    source = plugins_home / "data" / "feed-github"
    source.mkdir(parents=True)
    (source / "state.json").write_text("old-global\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    target = workspace / "plugin-data" / "feed-github"
    target.mkdir(parents=True)
    (target / "state.json").write_text("new-empty\n", encoding="utf-8")
    untouched = workspace / "plugin-data" / "calendar-github"
    untouched.mkdir()
    (untouched / "state.json").write_text("calendar-current\n", encoding="utf-8")

    replaced, backup_root = replace_plugin_data(
        workspace=workspace,
        plugins_home=plugins_home,
        plugin_names=("feed-github",),
    )

    assert replaced == [target]
    assert (target / "state.json").read_text(encoding="utf-8") == "old-global\n"
    assert (backup_root / "plugin-data" / "feed-github" / "state.json").read_text(
        encoding="utf-8"
    ) == "new-empty\n"
    assert (backup_root / "selection.txt").read_text(
        encoding="utf-8"
    ) == "feed-github\n"
    assert (
        untouched / "state.json"
    ).read_text(encoding="utf-8") == "calendar-current\n"


def test_replace_plugin_data_requires_idle_workspace(tmp_path: Path) -> None:
    plugins_home = tmp_path / "plugins-home"
    source = plugins_home / "data" / "feed-github"
    source.mkdir(parents=True)
    (source / "state.json").write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    lock = WorkspaceInstanceLock(workspace)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="其他 runtime 占用"):
            replace_plugin_data(
                workspace=workspace,
                plugins_home=plugins_home,
                plugin_names=("feed-github",),
            )
    finally:
        lock.release()


def test_replace_plugin_data_rolls_back_all_targets_on_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugins_home = tmp_path / "plugins-home"
    workspace = tmp_path / "workspace"
    for name in ("feed-github", "fitbit-github"):
        source = plugins_home / "data" / name
        source.mkdir(parents=True)
        (source / "state.json").write_text(f"source:{name}\n", encoding="utf-8")
        target = workspace / "plugin-data" / name
        target.mkdir(parents=True)
        (target / "state.json").write_text(f"target:{name}\n", encoding="utf-8")

    real_replace = migration_module.os.replace
    call_count = 0

    def fail_fourth_publish(source: Path, destination: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 4:
            raise OSError("publish failed")
        real_replace(source, destination)

    monkeypatch.setattr(migration_module.os, "replace", fail_fourth_publish)

    with pytest.raises(OSError, match="publish failed"):
        replace_plugin_data(
            workspace=workspace,
            plugins_home=plugins_home,
            plugin_names=("feed-github", "fitbit-github"),
        )

    for name in ("feed-github", "fitbit-github"):
        restored = workspace / "plugin-data" / name / "state.json"
        assert restored.read_text(encoding="utf-8") == f"target:{name}\n"
    assert list((workspace / "backups").iterdir()) == []


@pytest.mark.parametrize("interrupt_after", (1, 2))
def test_replace_plugin_data_rolls_back_interrupt_after_successful_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_after: int,
) -> None:
    plugins_home = tmp_path / "plugins-home"
    source = plugins_home / "data" / "feed-github"
    source.mkdir(parents=True)
    (source / "state.json").write_text("source\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    target = workspace / "plugin-data" / "feed-github"
    target.mkdir(parents=True)
    (target / "state.json").write_text("target\n", encoding="utf-8")

    real_replace = migration_module.os.replace
    call_count = 0

    def interrupt_after_replace(source_path: Path, destination: Path) -> None:
        nonlocal call_count
        real_replace(source_path, destination)
        call_count += 1
        if call_count == interrupt_after:
            raise KeyboardInterrupt

    monkeypatch.setattr(migration_module.os, "replace", interrupt_after_replace)

    with pytest.raises(KeyboardInterrupt):
        replace_plugin_data(
            workspace=workspace,
            plugins_home=plugins_home,
            plugin_names=("feed-github",),
        )

    assert (target / "state.json").read_text(encoding="utf-8") == "target\n"
    assert list((workspace / "backups").iterdir()) == []


@pytest.mark.parametrize("name", ("../feed-github", "feed/github", ""))
def test_replace_plugin_data_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="名称无效"):
        replace_plugin_data(
            workspace=tmp_path / "workspace",
            plugins_home=tmp_path / "plugins-home",
            plugin_names=(name,),
        )


def test_replace_plugin_data_rejects_non_directory_target(tmp_path: Path) -> None:
    plugins_home = tmp_path / "plugins-home"
    source = plugins_home / "data" / "feed-github"
    source.mkdir(parents=True)
    (source / "state.json").write_text("source\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    target = workspace / "plugin-data" / "feed-github"
    target.parent.mkdir(parents=True)
    target.write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="目标不是目录"):
        replace_plugin_data(
            workspace=workspace,
            plugins_home=plugins_home,
            plugin_names=("feed-github",),
        )

    assert target.read_text(encoding="utf-8") == "corrupt\n"


def test_replace_plugin_data_rejects_symlink_source_root(tmp_path: Path) -> None:
    real_data = tmp_path / "real-data"
    source = real_data / "feed-github"
    source.mkdir(parents=True)
    (source / "state.json").write_text("source\n", encoding="utf-8")
    plugins_home = tmp_path / "plugins-home"
    plugins_home.mkdir()
    (plugins_home / "data").symlink_to(real_data, target_is_directory=True)

    with pytest.raises(FileNotFoundError, match="数据根不存在或不安全"):
        replace_plugin_data(
            workspace=tmp_path / "workspace",
            plugins_home=plugins_home,
            plugin_names=("feed-github",),
        )


@pytest.mark.parametrize("plugin_names", ((), ("feed-github", "feed-github")))
def test_replace_plugin_data_rejects_invalid_selection(
    tmp_path: Path,
    plugin_names: tuple[str, ...],
) -> None:
    message = "至少指定" if not plugin_names else "不能重复"
    with pytest.raises(ValueError, match=message):
        replace_plugin_data(
            workspace=tmp_path / "workspace",
            plugins_home=tmp_path / "plugins-home",
            plugin_names=plugin_names,
        )


def test_migrate_plugin_data_module_cli_runs_from_repo_root(tmp_path: Path) -> None:
    plugins_home = tmp_path / "plugins-home"
    source = plugins_home / "data" / "feed-github"
    source.mkdir(parents=True)
    (source / "state.json").write_text("kept\n", encoding="utf-8")
    workspace = tmp_path / "workspace"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.migrate_plugin_data",
            "--workspace",
            str(workspace),
            "--plugins-home",
            str(plugins_home),
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    target = workspace / "plugin-data" / "feed-github"
    assert str(target) in result.stdout
    assert (target / "state.json").read_text(encoding="utf-8") == "kept\n"


def test_replace_plugin_data_module_cli_runs_from_repo_root(tmp_path: Path) -> None:
    plugins_home = tmp_path / "plugins-home"
    source = plugins_home / "data" / "feed-github"
    source.mkdir(parents=True)
    (source / "state.json").write_text("legacy\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    target = workspace / "plugin-data" / "feed-github"
    target.mkdir(parents=True)
    (target / "state.json").write_text("current\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.migrate_plugin_data",
            "--workspace",
            str(workspace),
            "--plugins-home",
            str(plugins_home),
            "--replace-plugin",
            "feed-github",
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "原 workspace 数据备份:" in result.stdout
    assert (target / "state.json").read_text(encoding="utf-8") == "legacy\n"

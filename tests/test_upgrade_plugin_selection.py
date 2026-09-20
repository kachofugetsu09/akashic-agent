import json
from pathlib import Path
import sqlite3

import pytest

from agent.plugins.selection import PluginSelection, SelectionFormatError, SelectionWriteError
from bootstrap.workspace_lock import PluginPublicationLock, WorkspaceInstanceLock
from scripts.upgrade_plugin_selection import initialize_selection


def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace, home = tmp_path / "workspace", tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    (home / "manifest.toml").write_text('[plugins."probe@lab"]\nenabled = true\n')
    plugin = home / "cache/lab/probe"
    plugin.mkdir(parents=True)
    (plugin / ".pointers.json").write_text('{"stable":".artifacts/old","latest":".artifacts/new"}')
    # 无制品也不猜测、安装或导入；这个命令只建立空选择。
    (workspace / "sessions.db").write_bytes(b"message body must stay opaque")
    data = workspace / "plugin-data/probe-lab"
    data.mkdir(parents=True)
    (data / "private").write_bytes(b"must not enter metadata backup")
    return workspace, home, tmp_path / "recovery"


def test_explicit_upgrade_backs_up_metadata_and_only_initializes_null(tmp_path: Path) -> None:
    workspace, home, backup = inputs(tmp_path)
    before = (home / "cache/lab/probe/.pointers.json").read_bytes()
    result = initialize_selection(workspace=workspace, plugins_home=home, backup_dir=backup)
    assert result == backup
    assert PluginSelection(workspace).read() is None
    assert (backup / "plugins/cache/lab/probe/.pointers.json").read_bytes() == before
    assert (home / "cache/lab/probe/.pointers.json").read_bytes() == before
    assert (workspace / "sessions.db").read_bytes() == b"message body must stay opaque"
    assert (workspace / "plugin-data/probe-lab/private").read_bytes() == b"must not enter metadata backup"
    files = {path.relative_to(backup).as_posix() for path in backup.rglob("*") if path.is_file()}
    assert files == {"recovery.json", "plugins/manifest.toml", "plugins/cache/lab/probe/.pointers.json"}
    recovery = json.loads((backup / "recovery.json").read_text())
    assert recovery["previous_selection"] == "absent"


def test_reload_backup_includes_committed_wal_without_touching_messages(tmp_path: Path) -> None:
    workspace, home, backup = inputs(tmp_path)
    runtime = workspace / "runtime"
    runtime.mkdir()
    source = sqlite3.connect(runtime / "plugin-reloads.sqlite3")
    try:
        source.execute("PRAGMA journal_mode=WAL")
        source.execute("CREATE TABLE evidence (value TEXT)")
        source.execute("INSERT INTO evidence VALUES ('pending update')")
        source.commit()
        initialize_selection(workspace=workspace, plugins_home=home, backup_dir=backup)
        copied = sqlite3.connect(backup / "workspace/runtime/plugin-reloads.sqlite3")
        try:
            assert copied.execute("SELECT value FROM evidence").fetchall() == [("pending update",)]
        finally:
            copied.close()
        assert source.execute("SELECT value FROM evidence").fetchall() == [("pending update",)]
    finally:
        source.close()


@pytest.mark.parametrize("content", ["broken", '{}', '{"version":1,"root_ref":null}'])
def test_any_existing_selection_fails_without_backup_or_overwrite(tmp_path: Path, content: str) -> None:
    workspace, home, backup = inputs(tmp_path)
    path = workspace / "runtime/plugin-stable.json"
    path.parent.mkdir()
    path.write_text(content)
    with pytest.raises(SelectionFormatError, match="已存在"):
        initialize_selection(workspace=workspace, plugins_home=home, backup_dir=backup)
    assert path.read_text() == content
    assert not backup.exists()


@pytest.mark.parametrize("lock_type", [WorkspaceInstanceLock, PluginPublicationLock])
def test_active_owner_blocks_upgrade(tmp_path: Path, lock_type) -> None:
    workspace, home, backup = inputs(tmp_path)
    lock = lock_type(workspace if lock_type is WorkspaceInstanceLock else home)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError):
            initialize_selection(workspace=workspace, plugins_home=home, backup_dir=backup)
    finally:
        lock.release()
    assert not PluginSelection(workspace).path.exists()
    assert not backup.exists()


def test_backup_failure_prevents_initialization(tmp_path: Path) -> None:
    workspace, home, backup = inputs(tmp_path)
    backup.mkdir()
    (backup / "keep").write_text("original")
    with pytest.raises(FileExistsError):
        initialize_selection(workspace=workspace, plugins_home=home, backup_dir=backup)
    assert not PluginSelection(workspace).path.exists()
    assert (backup / "keep").read_text() == "original"


def test_initialization_failure_preserves_completed_recovery_point(tmp_path: Path, monkeypatch) -> None:
    workspace, home, backup = inputs(tmp_path)

    def fail(self):
        assert (backup / "recovery.json").is_file()
        assert (backup / "plugins/manifest.toml").read_bytes() == (home / "manifest.toml").read_bytes()
        raise SelectionWriteError(operation="initialize", target_ref=None, outcome="uncertain",
                                  observed_ref=None, observation_error=None)

    monkeypatch.setattr(PluginSelection, "initialize", fail)
    with pytest.raises(SelectionWriteError):
        initialize_selection(workspace=workspace, plugins_home=home, backup_dir=backup)
    assert (backup / "recovery.json").is_file()


def test_metadata_symlink_or_bad_pointer_is_not_silently_accepted(tmp_path: Path) -> None:
    workspace, home, backup = inputs(tmp_path)
    pointer = home / "cache/lab/probe/.pointers.json"
    pointer.write_text("broken")
    with pytest.raises(ValueError):
        initialize_selection(workspace=workspace, plugins_home=home, backup_dir=backup)
    pointer.unlink()
    pointer.symlink_to(workspace / "sessions.db")
    with pytest.raises(ValueError, match="普通文件"):
        initialize_selection(workspace=workspace, plugins_home=home, backup_dir=backup)
    assert not backup.exists()
    assert not PluginSelection(workspace).path.exists()

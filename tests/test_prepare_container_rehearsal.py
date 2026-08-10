from __future__ import annotations

import json
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

import pytest

from scripts.container_rehearsal import workspace_snapshot
from scripts.container_rehearsal.model import SnapshotDriftError
from scripts.container_rehearsal.prepare import prepare_rehearsal
from scripts.container_rehearsal.sqlite_snapshot import verify_session_media_references


def _write_config(path: Path) -> None:
    path.write_text(
        """
[runtime]
workspace = "/formal/workspace"

[llm]
registry = "workspace"

[channels.chat]
enabled = false
channel_name = "web"

[channels.telegram]
enabled = true
token = "telegram-secret"
allow_from = ["owner"]

[channels.qq]
enabled = true
bot_uin = "123456"
allow_from = ["owner"]

[mobile_realtime]
enabled = true
public_url = "wss://mobile.example/ws"

[proactive]
enabled = true

[proactive.target]
channel = "telegram"
chat_id = "secret-chat-id"

[proactive.drift]
enabled = true
""".lstrip(),
        encoding="utf-8",
    )


def _create_live_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("CREATE TABLE events (value TEXT NOT NULL)")
    connection.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, extra TEXT)")
    connection.execute("INSERT INTO events VALUES ('live-row')")
    connection.commit()
    return connection


def test_prepare_rehearsal_copies_business_state_and_live_sqlite(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "formal-workspace"
    workspace.mkdir()
    (workspace / "memory").mkdir()
    (workspace / "memory" / "MEMORY.md").write_text("kept\n", encoding="utf-8")
    schedules = [{"id": "formal-job", "enabled": True, "channel": "mobile"}]
    (workspace / "schedules.json").write_text(
        json.dumps(schedules, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (workspace / "plugin-data" / "feed-github").mkdir(parents=True)
    (workspace / "plugin-data" / "feed-github" / "state.json").write_text(
        '{"kept": true}\n', encoding="utf-8"
    )
    (workspace / "uploads").mkdir()
    media = workspace / "uploads" / "photo.png"
    media.write_bytes(b"stable-photo")
    database = _create_live_database(workspace / "sessions.db")
    database.execute(
        "INSERT INTO messages VALUES (?, ?)",
        ("message-1", json.dumps({"media": [str(media)]})),
    )
    database.commit()

    for excluded in ("backups", "cache", "downloads", "runtime", "rebuilds"):
        directory = workspace / excluded
        directory.mkdir()
        (directory / "must-not-copy.txt").write_text("excluded", encoding="utf-8")
    (workspace / ".runtime-ready.json").write_text("{}", encoding="utf-8")
    (workspace / "observe.db.corrupt.20260412-165929").write_bytes(
        b"SQLite format 3\x00broken"
    )
    (workspace / "skills").mkdir()
    (workspace / "skills" / "cached-skill").symlink_to(
        tmp_path / "plugin-cache" / "skill", target_is_directory=True
    )
    (workspace / "skills" / "local-skill").mkdir()
    (workspace / "skills" / "local-skill" / "SKILL.md").write_text(
        "local\n", encoding="utf-8"
    )

    config = tmp_path / "config.toml"
    _write_config(config)
    plugin_home = tmp_path / "plugin-home"
    plugin_home.mkdir()
    (plugin_home / "manifest.toml").write_text(
        "[plugins.feed]\nenabled = true\n\n"
        '[plugins."feed@github"]\nenabled = true\n',
        encoding="utf-8",
    )
    (plugin_home / "cache").mkdir()
    (plugin_home / "cache" / "code.py").write_text("not copied\n", encoding="utf-8")
    target = tmp_path / "rehearsal"

    try:
        manifest_path = prepare_rehearsal(
            source_workspace=workspace,
            source_config=config,
            plugin_home=plugin_home,
            target=target,
        )
    finally:
        database.close()

    assert manifest_path == target / "rehearsal-manifest.json"
    assert (target / "workspace" / "memory" / "MEMORY.md").read_text() == "kept\n"
    assert (
        target / "workspace" / "plugin-data" / "feed-github" / "state.json"
    ).is_file()
    assert (target / "workspace" / "skills" / "local-skill" / "SKILL.md").is_file()
    assert not (target / "workspace" / "skills" / "cached-skill").exists()
    assert not (target / "workspace" / "backups").exists()
    assert not (target / "workspace" / "observe.db.corrupt.20260412-165929").exists()
    assert not (target / "workspace" / "sessions.db-wal").exists()
    with closing(sqlite3.connect(target / "workspace" / "sessions.db")) as copied:
        assert copied.execute("SELECT value FROM events").fetchall() == [("live-row",)]
        assert copied.execute("PRAGMA integrity_check").fetchall() == [("ok",)]

    candidate = tomllib.loads((target / "config.toml").read_text(encoding="utf-8"))
    assert candidate["runtime"]["workspace"] == str(target / "workspace")
    assert candidate["llm"] == {"registry": "workspace"}
    assert candidate["channels"]["chat"]["enabled"] is True
    assert candidate["channels"]["telegram"]["enabled"] is False
    assert candidate["channels"]["telegram"]["token"] == ""
    assert candidate["channels"]["qq"]["enabled"] is False
    assert candidate["channels"]["qq"]["bot_uin"] == ""
    assert candidate["mobile_realtime"]["enabled"] is False
    assert candidate["proactive"]["enabled"] is False
    assert candidate["proactive"]["drift"]["enabled"] is False
    assert (target / "workspace" / "schedules.json").read_text() == "[]\n"
    assert (
        json.loads((target / "workspace" / "schedules.source.json").read_text())
        == schedules
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "telegram-secret" not in serialized
    assert "secret-chat-id" not in serialized
    assert manifest["candidate"]["plugin_cache_copied"] is False
    assert manifest["candidate"]["plugin_manifest_copied_unmodified"] is False
    assert manifest["candidate"]["schedules_disabled"] == 1
    assert manifest["candidate"]["source_schedules"] == (
        "workspace/schedules.source.json"
    )
    assert manifest["candidate"]["plugins_disabled_until_rebuilt"] == ["feed@github"]
    plugin_manifest = tomllib.loads(
        (target / "plugin-home" / "manifest.toml").read_text(encoding="utf-8")
    )
    assert plugin_manifest["plugins"]["feed"]["enabled"] is True
    assert plugin_manifest["plugins"]["feed@github"]["enabled"] is False
    assert any(
        item["path"] == "observe.db.corrupt.20260412-165929"
        and item["reason"] == "forensic_corrupt_artifact"
        for item in manifest["excluded"]
    )
    assert manifest["cleanup"]["exact_paths"] == [str(target)]
    assert len(manifest["databases"]) == 1
    database_evidence = manifest["databases"][0]
    assert database_evidence["path"] == "sessions.db"
    assert database_evidence["source_integrity_check"] == "ok"
    assert database_evidence["target_integrity_check"] == "ok"
    assert database_evidence["workspace_media_references"] == {
        "checked": 1,
        "preexisting_missing": [],
        "status": "ok",
    }
    assert manifest["consistency"] == {
        "attempts": 1,
        "drift_retries": [],
        "max_attempts": 3,
    }


def test_prepare_rehearsal_refuses_existing_or_overlapping_target(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "keep.txt"
    marker.write_text("formal", encoding="utf-8")
    config = tmp_path / "config.toml"
    _write_config(config)
    plugin_home = tmp_path / "plugin-home"
    plugin_home.mkdir()
    (plugin_home / "manifest.toml").write_text("[plugins]\n", encoding="utf-8")

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="尚不存在"):
        prepare_rehearsal(
            source_workspace=workspace,
            source_config=config,
            plugin_home=plugin_home,
            target=existing,
        )
    with pytest.raises(ValueError, match="Workspace.*内部"):
        prepare_rehearsal(
            source_workspace=workspace,
            source_config=config,
            plugin_home=plugin_home,
            target=workspace / "candidate",
        )
    assert marker.read_text(encoding="utf-8") == "formal"
    assert not (workspace / "candidate").exists()


def test_prepare_rehearsal_rejects_included_external_symlink_atomically(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "plugin-data").mkdir()
    (workspace / "plugin-data" / "escape").symlink_to(tmp_path / "outside")
    config = tmp_path / "config.toml"
    _write_config(config)
    plugin_home = tmp_path / "plugin-home"
    plugin_home.mkdir()
    (plugin_home / "manifest.toml").write_text("[plugins]\n", encoding="utf-8")
    target = tmp_path / "candidate"

    with pytest.raises(ValueError, match="符号链接"):
        prepare_rehearsal(
            source_workspace=workspace,
            source_config=config,
            plugin_home=plugin_home,
            target=target,
        )

    assert not target.exists()
    assert list(tmp_path.glob(".candidate.preparing-*")) == []


def test_file_created_during_database_backup_retries_whole_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "uploads").mkdir()
    database = _create_live_database(workspace / "sessions.db")
    config = tmp_path / "config.toml"
    _write_config(config)
    plugin_home = tmp_path / "plugin-home"
    plugin_home.mkdir()
    (plugin_home / "manifest.toml").write_text("[plugins]\n", encoding="utf-8")
    target = tmp_path / "candidate"
    original_copy_sqlite = workspace_snapshot.copy_sqlite
    calls = 0

    def create_file_then_backup(source: Path, destination: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            (workspace / "uploads" / "arrived-during-db.png").write_bytes(b"new")
        return original_copy_sqlite(source, destination)

    monkeypatch.setattr(workspace_snapshot, "copy_sqlite", create_file_then_backup)
    try:
        manifest_path = prepare_rehearsal(
            source_workspace=workspace,
            source_config=config,
            plugin_home=plugin_home,
            target=target,
        )
    finally:
        database.close()

    assert calls == 2
    assert (
        target / "workspace" / "uploads" / "arrived-during-db.png"
    ).read_bytes() == b"new"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["consistency"]["attempts"] == 2
    assert (
        "added=['uploads/arrived-during-db.png']"
        in manifest["consistency"]["drift_retries"][0]
    )


def test_missing_workspace_media_reference_is_reported_as_preexisting(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = _create_live_database(workspace / "sessions.db")
    missing = workspace / "uploads" / "missing.png"
    database.execute(
        "INSERT INTO messages VALUES (?, ?)",
        ("message-missing", json.dumps({"media": [str(missing)]})),
    )
    database.commit()
    config = tmp_path / "config.toml"
    _write_config(config)
    plugin_home = tmp_path / "plugin-home"
    plugin_home.mkdir()
    (plugin_home / "manifest.toml").write_text("[plugins]\n", encoding="utf-8")
    target = tmp_path / "candidate"

    try:
        manifest_path = prepare_rehearsal(
            source_workspace=workspace,
            source_config=config,
            plugin_home=plugin_home,
            target=target,
        )
    finally:
        database.close()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    database_evidence = manifest["databases"][0]
    assert database_evidence["workspace_media_references"] == {
        "checked": 0,
        "preexisting_missing": ["uploads/missing.png"],
        "status": "ok",
    }


def test_existing_workspace_media_missing_from_copy_fails_loud(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "uploads").mkdir(parents=True)
    destination.mkdir()
    media = source / "uploads" / "photo.png"
    media.write_bytes(b"photo")
    database = _create_live_database(destination / "sessions.db")
    database.execute(
        "INSERT INTO messages VALUES (?, ?)",
        ("message-omitted", json.dumps({"media": [str(media)]})),
    )
    database.commit()
    database.close()
    records: list[dict[str, object]] = [{"path": "sessions.db"}]

    with pytest.raises(SnapshotDriftError, match="媒体未进入副本"):
        verify_session_media_references(source, destination, records)

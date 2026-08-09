from __future__ import annotations

import json
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

import pytest

from scripts.prepare_container_rehearsal import prepare_rehearsal


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
    (workspace / "plugin-data" / "feed-github").mkdir(parents=True)
    (workspace / "plugin-data" / "feed-github" / "state.json").write_text(
        '{"kept": true}\n', encoding="utf-8"
    )
    database = _create_live_database(workspace / "sessions.db")

    for excluded in ("backups", "cache", "downloads", "runtime", "rebuilds"):
        directory = workspace / excluded
        directory.mkdir()
        (directory / "must-not-copy.txt").write_text("excluded", encoding="utf-8")
    (workspace / ".runtime-ready.json").write_text("{}", encoding="utf-8")
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
        "[plugins.feed]\nenabled = true\n", encoding="utf-8"
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

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "telegram-secret" not in serialized
    assert "secret-chat-id" not in serialized
    assert manifest["candidate"]["plugin_cache_copied"] is False
    assert manifest["cleanup"]["exact_paths"] == [str(target)]
    assert manifest["databases"] == [
        {
            "page_count": manifest["databases"][0]["page_count"],
            "path": "sessions.db",
            "source_integrity_check": "ok",
            "target_integrity_check": "ok",
        }
    ]


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

from contextlib import closing
import sqlite3

import pytest

from session.identities import ChannelIdentities
from session.log import MessageLog, SessionAttributes


def _history(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)):
        pass
    config = tmp_path / "config.toml"
    config.write_text('[channels.telegram]\nchannel_name="telegram_work"\n')
    with closing(sqlite3.connect(path)) as db, db:
        for key, metadata, updated in (
            ("telegram_work:a", '{"username":" Alice "}', "2026-09-01"),
            ("telegram_work:b", '{"username":"ALICE"}', "2026-09-02"),
            ("telegram_work:z", '{"username":"alice", "unrelated": [1,2]}', "2026-09-02"),
            ("telegramXwork:wrong", '{"username":"ALICE"}', "2026-09-03"),
            ("feishu:room", '{"feishu_open_id":" ou_123 "}', "2026-09-01"),
            ("qq:room", '{"user_id":"00123"}', "2026-09-01"),
            ("unknown:room", '{"provider_identity":"keep"}', "2026-09-01"),
        ):
            db.execute(
                "INSERT INTO sessions(key,created_at,updated_at,metadata) VALUES (?,?,?,?)",
                (key, updated, updated, metadata),
            )
    return path, config


def test_legacy_channel_config_has_a_recoverable_plugin_migration(tmp_path):
    from scripts.migrate_legacy_channels import migrate_legacy_channels

    config = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    config.write_text(
        """
[channels.telegram]
token = "123:token"
allow_from = ["alice"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    assert migrate_legacy_channels(config, workspace, marketplace="installed") == ("telegram_channel",)
    assert "channels.telegram" not in config.read_text(encoding="utf-8")
    plugin = workspace / "plugin-data/telegram_channel-installed/config.local.toml"
    assert 'token = "123:token"' in plugin.read_text(encoding="utf-8")
    assert (tmp_path / "config.toml.before-channel-plugin-migration.bak").exists()


def test_channel_migration_rejects_custom_telegram_identity_without_silent_rekey(tmp_path):
    from scripts.migrate_legacy_channels import migrate_legacy_channels

    config = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    original = '[channels.telegram]\nchannel_name = "telegram_work"\ntoken = "secret"\n'
    config.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="自定义 channels.telegram.channel_name"):
        migrate_legacy_channels(config, workspace, marketplace="installed")

    assert config.read_text(encoding="utf-8") == original
    assert not workspace.exists()


def test_channel_migration_preserves_legacy_empty_channel_as_disabled_plugin(tmp_path):
    from scripts.migrate_legacy_channels import migrate_legacy_channels

    config = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    config.write_text("[channels.telegram]\n", encoding="utf-8")

    assert migrate_legacy_channels(config, workspace, marketplace="installed") == ("telegram_channel",)
    plugin = workspace / "plugin-data/telegram_channel-installed/config.local.toml"
    assert plugin.read_text(encoding="utf-8") == "enabled = false\nallow_from = []\n"


def test_explicit_session_delete_keeps_identity_in_same_audit_backup(tmp_path):
    from session.manager import SessionManager

    with closing(SessionManager(tmp_path)) as manager:
        manager.get_or_create("probe:room")
        manager.get_or_create("probe:other")
        manager.identities.remember("probe", "one", "room")
        manager.identities.remember("probe", "two", "other")
        audit = manager.delete_session_with_audit("probe:room")
        assert audit.result == "committed"
        assert manager.identities.load("probe") == {"two": "other"}
        assert manager.identities.migration_completed("probe")
        assert audit.backup_path is not None
        with closing(sqlite3.connect(audit.backup_path)) as backup:
            assert backup.execute("SELECT identity,chat_id FROM channel_identities ORDER BY identity").fetchall() == [("one", "room"), ("two", "other")]


def test_route_rollback_reopen_preserves_sessions_and_marker(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        log.ensure_session("probe:old", SessionAttributes())
    with closing(sqlite3.connect(path)) as db:
        before = db.execute("SELECT * FROM sessions").fetchall()
    with closing(ChannelIdentities(path)) as identities:
        first = identities.remember("probe", "Alice", "old")
        moved = identities.remember("probe", "Alice", "new")
        assert identities.rollback(first) is False
        assert identities.resolve("probe", "Alice") == "new"
        assert identities.rollback(moved) is True
        assert identities.resolve("probe", "Alice") == "old"
        assert identities.resolve("probe", "alice") is None
        assert identities.rollback(first) is True
    with closing(ChannelIdentities(path)) as identities:
        assert identities.load("probe") == {}
        assert identities.migration_completed("probe")
        identities.seed("probe", {"Alice": ("stale", "old timestamp")})
        assert identities.load("probe") == {}
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT * FROM sessions").fetchall() == before
        assert db.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


def test_independent_connections_do_not_rollback_later_acceptance(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(ChannelIdentities(path)) as first, closing(ChannelIdentities(path)) as second:
        old = first.remember("probe", "same", "old")
        _ = second.remember("probe", "same", "new")
        assert first.rollback(old) is False
        assert first.resolve("probe", "same") == "new"
        assert first.resolve("another", "same") is None


@pytest.mark.parametrize("damage", ["partial", "column"])
def test_unknown_identity_schema_is_not_repaired(tmp_path, damage):
    path = tmp_path / "sessions.db"
    with closing(ChannelIdentities(path)):
        pass
    with closing(sqlite3.connect(path)) as db:
        if damage == "partial":
            db.execute("DROP TABLE channel_identity_migrations")
        else:
            db.execute("ALTER TABLE channel_identities ADD COLUMN extra TEXT")
        db.commit()
        before = tuple(db.iterdump())
    with pytest.raises(RuntimeError, match="schema"):
        ChannelIdentities(path)
    with closing(sqlite3.connect(path)) as db:
        assert tuple(db.iterdump()) == before


def test_seed_commits_routes_and_marker_together(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(ChannelIdentities(path)) as identities:
        # 迁移标记缺失而已有路由是未完成状态，不能覆盖它或部分发布。
        with identities._conn:
            identities._conn.execute(
                "INSERT INTO channel_identities VALUES ('probe', 'occupied', 'old', 'time')"
            )
        with pytest.raises(ValueError, match="缺少迁移标记"):
            identities.seed("probe", {"new": ("new", "time"), "occupied": ("other", "time")})
        assert identities.load("probe") == {"occupied": "old"}
        assert identities.migration_completed("probe") is False

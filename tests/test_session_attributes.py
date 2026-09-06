from contextlib import closing
from dataclasses import FrozenInstanceError
import json
import sqlite3

import pytest

from agent.migrations.session_attributes import migrate
from session.log import MessageConflict, MessageLog, SessionAttributes, _OLD_SESSION_SCHEMA, _LEGACY_SESSION_SCHEMA
from session.message import ContentPart, ContentReferences, Input


def test_session_admission_preserves_independent_attributes_across_restart(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        for visibility in ("listed", "internal"):
            for learning in ("eligible", "excluded"):
                key = visibility + learning
                attributes = SessionAttributes(visibility, learning)
                assert log.ensure_session(key, attributes) == attributes
                writer = log.writer(key, author="app", source="work", body_types=(Input,),
                                    content={"text": lambda part: ContentReferences()})
                writer.append(key, Input((ContentPart("text", key),)))
                assert log.ensure_session(key, attributes) == attributes
                with pytest.raises(FrozenInstanceError):
                    attributes.learning = "eligible"
        before = dict(log.catalog().snapshot_attributes())
        with pytest.raises(MessageConflict):
            log.ensure_session("internalexcluded", SessionAttributes())
        assert dict(log.catalog().snapshot_attributes()) == before
    with closing(MessageLog(path)) as log:
        assert dict(log.catalog().snapshot_attributes()) == before
        assert log.catalog().snapshot_heads() == {key: 0 for key in before}


@pytest.mark.parametrize("schema", [_OLD_SESSION_SCHEMA, _LEGACY_SESSION_SCHEMA])
def test_migration_preserves_legacy_metadata_and_never_guesses_from_source(tmp_path, schema):
    path = tmp_path / "sessions.db"
    raw_metadata = '{ "programmatic":true, "ephemeral": true, "effects":{"post_commit":"suppress"} }'
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(schema)
        connection.execute("CREATE TABLE messages(id TEXT, session_key TEXT, seq INTEGER, body TEXT)")
        connection.execute("INSERT INTO sessions(key,created_at,updated_at,metadata) VALUES(?,?,?,?)",
                           ("scheduler:old-job", "old", "old", raw_metadata))
        connection.execute("INSERT INTO messages VALUES(?,?,?,?)", ("old-message", "scheduler:old-job", 0, "exact original"))
        connection.commit()
    recovery = tmp_path / "backup"
    migrate(path, recovery)
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute("SELECT metadata,attributes FROM sessions").fetchone()
        assert row == (raw_metadata, '{"learning": "eligible", "visibility": "listed"}')
        assert connection.execute("SELECT body FROM messages").fetchone() == ("exact original",)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        # 已成功迁移后新增的排除属性不能被重复迁移改回默认值。
        connection.execute("INSERT INTO sessions(key,created_at,updated_at,attributes) VALUES(?,?,?,?)",
                           ("new-internal", "new", "new", '{"learning":"excluded","visibility":"internal"}'))
        connection.commit()
    migrate(path, recovery)
    with closing(sqlite3.connect(path)) as connection:
        assert 'excluded' in connection.execute("SELECT attributes FROM sessions WHERE key='new-internal'").fetchone()[0]
    manifest = json.loads((recovery / "manifest.json").read_text())
    with closing(sqlite3.connect(recovery / manifest["backup"])) as connection:
        assert connection.execute("SELECT metadata FROM sessions").fetchone()[0] == raw_metadata
        assert "attributes" not in {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}


def test_unknown_session_schema_blocks_migration_without_backup(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE sessions(key TEXT PRIMARY KEY, metadata TEXT)")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="未知 schema"):
        migrate(path, tmp_path / "backup")
    assert path.read_bytes() == before
    assert not (tmp_path / "backup").exists()

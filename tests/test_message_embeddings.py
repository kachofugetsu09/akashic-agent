from session.message import ContentReferences
from dataclasses import replace
from contextlib import closing
import sqlite3

import pytest

from session.embedding_store import MessageEmbeddingStore, MessageEmbeddings
from session.log import MessageConflict, MessageLog
from session.message import ContentPart, Input


def test_message_vectors_reuse_legacy_rows_and_never_overwrite_fixed_facts(tmp_path):
    path = tmp_path / "sessions.db"
    log = MessageLog(path)
    store = MessageEmbeddingStore(path)
    try:
        writer = log.writer("s", author="user", source="chat", body_types=(Input,), content={"text": lambda part: ContentReferences()})
        message = writer.append("u", Input((ContentPart("text", "actual"),)))
        store.upsert(message_id="u", content="actual", model="frozen-model", embedding=[0.25, 0.5])
        records = MessageEmbeddings(log).bind(lambda m: m.body.parts[0].value)
        assert records.read(message, model="frozen-model", dimension=2) == (0.25, 0.5)
        records.save(message, model="frozen-model", embedding=[0.25, 0.5])
        with pytest.raises(MessageConflict):
            records.save(message, model="frozen-model", embedding=[0.5, 0.5])
        with pytest.raises(MessageConflict):
            records.read(replace(message, body=Input((ContentPart("text", "forged"),))), model="frozen-model", dimension=2)
        with pytest.raises(ValueError, match="不匹配"):
            records.read(message, model="frozen-model", dimension=3)
        other_projection = MessageEmbeddings(log).bind(lambda m: "different")
        with pytest.raises(ValueError, match="不匹配"):
            other_projection.read(message, model="frozen-model", dimension=2)
        assert records.read(message, model="new-model", dimension=2) is None
        records.save(message, model="new-model", embedding=[1.0, 0.0])
        assert records.read(message, model="frozen-model", dimension=2) == (0.25, 0.5)
        assert log.reader("s").snapshot() == (message,)
    finally:
        store.close()
        log.close()


def test_missing_schema_and_corrupt_vectors_fail_without_reembedding(tmp_path):
    log = MessageLog(tmp_path / "sessions.db")
    message = log.writer("s", author="user", source="chat", body_types=(Input,), content={}).append("u", Input(()))
    with closing(sqlite3.connect(tmp_path / "sessions.db")) as db, db:
        db.execute("DROP TABLE message_embeddings")
    records = MessageEmbeddings(log).bind(lambda m: "")
    try:
        with pytest.raises(sqlite3.OperationalError, match="message_embeddings"):
            records.read(message, model="model", dimension=2)
        with pytest.raises(ValueError):
            records.save(message, model="model", embedding=[float("nan"), 0.0])
        with pytest.raises(RuntimeError, match="candidate"):
            MessageEmbeddings(None).bind(lambda m: "")
    finally:
        log.close()


@pytest.mark.parametrize('legacy', [False, True])
def test_yoyo_embedding_owner_preserves_existing_vectors_and_backs_up_missing_schema(tmp_path, legacy):
    from pathlib import Path
    from yoyo import read_migrations
    from agent.migrations.context import bind_migration_context
    path = tmp_path / 'sessions.db'
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE unrelated_owner (value TEXT)')
        db.execute("INSERT INTO unrelated_owner VALUES ('preserved')")
    if legacy:
        store = MessageEmbeddingStore(path)
        try:
            store.upsert(message_id='historical', content='original text', model='fixed', embedding=[0.6, 0.8])
        finally:
            store.close()
    before = path.read_bytes()
    migration = next(m for m in read_migrations(str(Path(__file__).parents[1] / 'migrations/yoyo'))
                     if m.id == '20260905_05_message_embeddings')
    migration.load()
    with bind_migration_context(config_path=tmp_path / 'config.toml', workspace=tmp_path):
        migration.module.migrate_message_embeddings(None)
        once = path.read_bytes()
        migration.module.migrate_message_embeddings(None)
    assert path.read_bytes() == once
    with closing(sqlite3.connect(path)) as db:
        assert db.execute('SELECT value FROM unrelated_owner').fetchall() == [('preserved',)]
        assert db.execute('SELECT count(*) FROM message_embeddings').fetchone()[0] == int(legacy)
    if legacy:
        assert path.read_bytes() == before
        assert not (tmp_path / 'backups').exists()
    else:
        backups = list((tmp_path / 'backups').glob('message-embeddings-owner-v1/*/sessions.db'))
        assert len(backups) == 1
        with closing(sqlite3.connect(backups[0])) as db:
            assert db.execute("SELECT name FROM sqlite_master WHERE name='message_embeddings'").fetchone() is None


def test_yoyo_embedding_owner_rejects_unknown_schema_before_backup(tmp_path):
    from pathlib import Path
    from yoyo import read_migrations
    from agent.migrations.context import bind_migration_context
    path = tmp_path / 'sessions.db'
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE message_embeddings (message_id TEXT)')
    before = path.read_bytes()
    migration = next(m for m in read_migrations(str(Path(__file__).parents[1] / 'migrations/yoyo'))
                     if m.id == '20260905_05_message_embeddings')
    migration.load()
    with bind_migration_context(config_path=tmp_path / 'config.toml', workspace=tmp_path):
        with pytest.raises(RuntimeError, match='schema lineage'):
            migration.module.migrate_message_embeddings(None)
    assert path.read_bytes() == before
    assert not (tmp_path / 'backups').exists()

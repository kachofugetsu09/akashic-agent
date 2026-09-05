from session.message import ContentReferences
from contextlib import closing
from pathlib import Path
import sqlite3

import pytest
from yoyo import get_backend, read_migrations

from agent.migrations.akasha_consumption import cutover_akasha
from agent.migrations.context import bind_migration_context
from plugins.akasha.application.rebuild import rebuild_memory
from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.infrastructure.persistence import load_consumption, logical_state_sha256, sha256_file
from plugins.akasha.infrastructure.sparse_index import BuildConfig, build_sparse_index
from session.embedding_store import MessageEmbeddingStore
from session.log import MessageLog
from session.message import Input, Output, ContentPart
from tests.test_akasha_embedding_backfill import _create_sessions


@pytest.fixture
def workspace(tmp_path):
    old = tmp_path / 'old-sessions.db'
    _create_sessions(old)
    vectors = MessageEmbeddingStore(old)
    try:
        for identity, text in [('u1', 'remember me'), ('a1', 'I will')]:
            vectors.upsert(message_id=identity, content=text, model='fixed', embedding=[0.6, 0.8])
    finally:
        vectors.close()
    workspace = tmp_path / 'workspace'
    memory = workspace / 'memory'
    memory.mkdir(parents=True)
    index = memory / 'akasha-v2-index.db'
    build_sparse_index(old, index, BuildConfig(embedding_model='fixed', embedding_dimension=2))
    rebuild_memory(index, memory / 'akasha.db', target_sequences=())
    log = MessageLog(workspace / 'sessions.db')
    try:
        log.writer('chat:one', author='user', source='chat', body_types=(Input,), content={'text': lambda part: ContentReferences()}).append('u1', Input((ContentPart('text', 'remember me'),)))
        log.writer('chat:one', author='agent', source='chat', body_types=(Output,), content={'text': lambda part: ContentReferences()}).append('a1', Output((ContentPart('text', 'I will'),), 'complete'))
    finally:
        log.close()
    return workspace


def test_yoyo_cutover_is_durable_idempotent_and_preserves_learned_state(workspace, tmp_path):
    source = Path(__file__).resolve().parents[1] / 'migrations/yoyo/20260905_04_akasha_consumption.py'
    # 用真正 yoyo Python step；隔离目录将已存在的父迁移声明为已完成前提。
    directory = tmp_path / 'migrations'
    directory.mkdir()
    parent = directory / '20260905_03_model_calls.py'
    parent.write_text('from yoyo import step\nsteps = [step("SELECT 1")]\n')
    (directory / source.name).write_bytes(source.read_bytes())
    memory = workspace / 'memory/akasha.db'
    index = workspace / 'memory/akasha-v2-index.db'
    before_graph = logical_state_sha256(memory)
    before_index = sha256_file(index)
    before_messages = sha256_file(workspace / 'sessions.db')
    backend = get_backend(f'sqlite:///{tmp_path / "ledger.db"}')
    migrations = read_migrations(str(directory))
    with backend, bind_migration_context(config_path=tmp_path / 'config.toml', workspace=workspace):
        backend.apply_migrations(backend.to_apply(migrations))
        assert not backend.to_apply(migrations)
        # 模拟 sidecar 已发布但 yoyo 尚未落账，再调用同一个 step。
        migrations[-1].module.migrate_akasha_consumption(None)
    state = load_consumption(memory)
    assert state.legacy_prefix.count == 1 and state.applied == ()
    assert state.cutover_heads == (('chat:one', 1),)
    assert logical_state_sha256(memory, include_consumption=False) == before_graph
    assert sha256_file(index) == before_index
    assert sha256_file(workspace / 'sessions.db') == before_messages
    backups = list((workspace / 'backups').glob('akasha-message-consumption-v1/*/manifest.json'))
    assert len(backups) == 1
    assert load_consumption(backups[0].parent / 'akasha.db') is None


def test_unknown_same_version_schema_fails_before_backup_or_mutation(workspace):
    memory = workspace / 'memory/akasha.db'
    with closing(sqlite3.connect(memory)) as db, db:
        db.execute('CREATE TABLE unknown_owner (data TEXT)')
    before = memory.read_bytes()
    with pytest.raises(ValueError, match='schema lineage'):
        cutover_akasha(memory=memory, index=workspace / 'memory/akasha-v2-index.db',
                      heads={'chat:one': 1}, config=MemoryConfig(), backup_root=workspace / 'backup')
    assert memory.read_bytes() == before
    assert not (workspace / 'backup').exists()

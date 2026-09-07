import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from agent.migrations.turn_messages import migrate_turn_messages
from plugins.content.api import legacy_post_commit_effect
from plugins.content.plugin import check_text
from plugins.conversation.source import needs_reply
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Control, Input, Output
from session.store import SessionStore

STAMP = "2026-09-05T10:00:00+08:00"


def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    with closing(MessageLog(root / "sessions.db")):
        pass
    # 从真实旧 owner 取 schema，避免测试用一个简化 turns 表替代生产谱系。
    old = tmp_path / "old.db"
    with closing(SessionStore(old)):
        pass
    with closing(sqlite3.connect(old)) as old_db, closing(sqlite3.connect(root / 'sessions.db')) as db, db:
        db.execute('INSERT INTO sessions (key,created_at,updated_at,metadata,next_seq) VALUES (?,?,?,?,?)',
                   ('probe:room', STAMP, STAMP, '{}', 0))
        for table in ('turns', 'inbound_handoffs'):
            db.execute(old_db.execute('SELECT sql FROM sqlite_master WHERE name=?', (table,)).fetchone()[0])
    return root


def user(ordinal, *, metadata=None):
    return {"id": "item" + str(ordinal), "type": "userMessage", "data": {
        "ordinal": ordinal, "content": "input " + str(ordinal), "timestamp": STAMP,
        "media": [], "metadata": {"client_message_id": "client" + str(ordinal), **(metadata or {})},
    }}


def turn(root, identity, items, *, continued=None, status='interrupted'):
    metadata = {"interactionId": "chain", "channel": "probe", "chatId": "room", "sender": "user",
                "channelMessageId": "client0", "channelSnapshotId": "snapshot", "channelGenerationId": "generation",
                "channelBindingToken": "token", "inboundMetadata": {"client_message_id": "client0"}}
    if continued:
        metadata['continuedFromTurnId'] = continued
    with closing(sqlite3.connect(root / 'sessions.db')) as db, db:
        db.execute('INSERT INTO turns (id,session_key,status,input_json,items_json,usage_json,error_json,final_response,created_at) VALUES (?,?,?,?,?,?,?,?,?)',
                   (identity, 'probe:room', status, json.dumps({'input': 'original input', 'metadata': metadata}, ensure_ascii=False),
                    json.dumps(items, ensure_ascii=False, indent=1), '{ "requests": 1 }', None, None, STAMP))
        return db.execute('SELECT * FROM turns WHERE id=?', (identity,)).fetchone()


def handoff(root, item):
    data = item['data']
    with closing(sqlite3.connect(root / 'sessions.db')) as db, db:
        db.execute('INSERT INTO inbound_handoffs VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   ('handoff' + item['id'], 'dedupe' + item['id'], 'probe', 'user', 'room', 'probe:room', data['content'],
                    STAMP, '[]', json.dumps(data['metadata']), STAMP))


def persisted(root, ordinal, *, metadata=None):
    import hashlib
    raw = json.dumps({'control_turn_id': 'chain', 'turn_input_ordinal': ordinal, 'client_message_id': 'client' + str(ordinal), **(metadata or {})})
    with closing(MessageLog(root / 'sessions.db')) as log:
        return log.writer('probe:room', author='legacy-attribution-unknown', source='legacy-unattributed',
                          body_types=(Input,), content={'text': check_text, 'history.provenance': lambda part: ContentReferences()}).append(
            'old' + str(ordinal), Input((ContentPart('text', 'input ' + str(ordinal)), ContentPart('history.provenance', {
                'schema': 'sessions.messages.v0', 'role': 'user', 'content_was_null': False,
                'extra': raw, 'extra_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            }))))


@pytest.mark.parametrize('all_mapped', [False, True])
def test_open_chain_preserves_originals_maps_every_input_and_pauses_atomically(tmp_path, all_mapped):
    root = workspace(tmp_path)
    original = persisted(root, 0)
    if all_mapped:
        persisted(root, 1, metadata={'effects': {'post_commit': 'suppress'}})
    first, second = user(0), user(1, metadata={'effects': {'post_commit': 'suppress'}})
    before = [turn(root, 't1', [first]), turn(root, 't2', [second], continued='t1')]
    handoff(root, first)
    handoff(root, second)
    receipt = migrate_turn_messages(root)
    with closing(MessageLog(root / 'sessions.db')) as log:
        reader = log.reader('probe:room')
        rows = reader.snapshot()
        assert reader.get(original.message_id) == original
        assert len(receipt['input_mapping']) == 2
        assert receipt['unmapped'] == []
        assert len({item['message_id'] for item in receipt['input_mapping']}) == 2
        assert rows[-1].body.action == 'pause' and not needs_reply(rows, 'conversation')
        assert rows[-1].body.through_seq == max(reader.get(item['message_id']).seq for item in receipt['input_mapping'])
        assert len([row for row in rows if isinstance(row.body, Input)]) == 2
        archives = [row for row in rows if isinstance(row.body, Output)]
        assert all(row.source == 'history' and row.body.finish == 'quiet' for row in archives)
        for index, archive in enumerate(archives):
            value = archive.body.parts[0].value['row']
            assert tuple(value[key] for key in ('id','session_key','status','input_json','items_json','usage_json','error_json','final_response','created_at','started_at','completed_at')) == before[index]
        if not all_mapped:
            from agent.turn_effects import PostCommitEffect
            imported = reader.get(receipt['input_mapping'][1]['message_id'])
            assert legacy_post_commit_effect(imported) == PostCommitEffect.SUPPRESS
        first_snapshot = rows
    assert migrate_turn_messages(root) == receipt
    with closing(MessageLog(root / 'sessions.db')) as log:
        assert log.reader('probe:room').snapshot() == first_snapshot
        log.writer('probe:room', author='user', source='conversation', body_types=(Input,), content={'text': check_text}).append(
            'new', Input((ContentPart('text', 'new input'),)))
        assert needs_reply(log.reader('probe:room').snapshot(), 'conversation')
    with closing(sqlite3.connect(root / 'sessions.db')) as db:
        assert db.execute('SELECT * FROM turns ORDER BY id').fetchall() == before
    assert Path(receipt['backup']).exists()


def test_forged_control_channel_metadata_is_only_archived_without_independent_receipt(tmp_path):
    root = workspace(tmp_path)
    turn(root, 't1', [user(0)])
    receipt = migrate_turn_messages(root)
    assert receipt['input_mapping'] == [] and receipt['pauses'] == []
    assert receipt['unmapped'] == [{'tail_id': 't1', 'reason': 'missing_independent_channel_receipt'}]
    with closing(MessageLog(root / 'sessions.db')) as log:
        rows = log.reader('probe:room').snapshot()
        assert len(rows) == 1 and rows[0].source == 'history'
        assert not needs_reply(rows, 'conversation')


@pytest.mark.parametrize('status', ['in_progress', 'interrupted', 'cancelled', 'completed', 'invented'])
def test_open_tool_without_domain_receipt_blocks_entire_migration_and_preserves_state(tmp_path, status):
    root = workspace(tmp_path)
    original = persisted(root, 0)
    old = turn(root, 't1', [user(0), {'id': 'tool', 'type': 'toolCall', 'data': {
        'status': status, 'callId': 'call', 'name': 'shell', 'args': {}, 'resultPreview': 'complete text',
    }}])
    with pytest.raises(RuntimeError, match='terminal receipt'):
        migrate_turn_messages(root)
    with closing(MessageLog(root / 'sessions.db')) as log:
        assert log.reader('probe:room').snapshot() == (original,)
        assert log.owner('migration:turn-messages-v1').list() == ()
    with closing(sqlite3.connect(root / 'sessions.db')) as db:
        assert db.execute('SELECT * FROM turns').fetchall() == [old]
    assert list((root / 'backups/turn-messages-v1').iterdir())


def test_conflicting_exact_input_reference_aborts_without_partial_archive(tmp_path):
    root = workspace(tmp_path)
    first, second = persisted(root, 0), persisted(root, 1)
    item = user(0, metadata={'persisted_user_message_id': 'old1'})
    turn(root, 't1', [item])
    handoff(root, item)
    with pytest.raises(ValueError, match='精确引用不一致'):
        migrate_turn_messages(root)
    with closing(MessageLog(root / 'sessions.db')) as log:
        assert log.reader('probe:room').snapshot() == (first, second)


def test_archive_never_enters_default_model_content(tmp_path):
    from plugins.models.content import render_content
    root = workspace(tmp_path)
    turn(root, 't1', [user(0)])
    migrate_turn_messages(root)
    with closing(MessageLog(root / 'sessions.db')) as log:
        archive = log.reader('probe:room').snapshot()[0]
        assert render_content(archive.body.parts[0], artifacts={}) == ()


@pytest.mark.parametrize('stage', ['before_receipt', 'after_commit'])
def test_crash_keeps_transaction_atomic_and_retry_uses_same_message_identities(tmp_path, monkeypatch, stage):
    from session.log import OwnerStore, OwnerTransaction
    root = workspace(tmp_path)
    item = user(0)
    old = turn(root, 't1', [item])
    handoff(root, item)
    if stage == 'before_receipt':
        save = OwnerTransaction.save
        def interrupted(tx, key, value, **kwargs):
            if key == 'manifest':
                raise OSError('crash before receipt')
            return save(tx, key, value, **kwargs)
        target, method = OwnerTransaction, 'save'
    else:
        transact = OwnerStore.transact
        def interrupted(store, callback):
            transact(store, callback)
            raise OSError('crash after commit before yoyo ledger')
        target, method = OwnerStore, 'transact'
    with monkeypatch.context() as patch:
        patch.setattr(target, method, interrupted)
        with pytest.raises(OSError, match='crash'):
            migrate_turn_messages(root)
    with closing(MessageLog(root / 'sessions.db')) as log:
        rows = log.reader('probe:room').snapshot()
        assert len(rows) == (0 if stage == 'before_receipt' else 3)
    receipt = migrate_turn_messages(root)
    with closing(MessageLog(root / 'sessions.db')) as log:
        after = log.reader('probe:room').snapshot()
        assert len(after) == 3 and after[-1].body.action == 'pause'
        if stage == 'after_commit':
            assert rows == after
    backup = Path(receipt['backup']) / 'sessions.db'
    with closing(sqlite3.connect(backup)) as db:
        assert db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
        assert db.execute('SELECT * FROM turns').fetchall() == [old]
        assert db.execute('SELECT COUNT(*) FROM messages').fetchone() == (0,)


@pytest.mark.parametrize('damage', ['schema', 'original_row', 'imported_message'])
def test_unknown_schema_or_damaged_migration_facts_fail_loud(tmp_path, damage):
    root = workspace(tmp_path)
    turn(root, 't1', [user(0)])
    if damage != 'schema':
        migrate_turn_messages(root)
    with closing(sqlite3.connect(root / 'sessions.db')) as db, db:
        if damage == 'schema':
            db.execute('ALTER TABLE turns ADD COLUMN unexpected TEXT')
        elif damage == 'original_row':
            db.execute("UPDATE turns SET final_response='changed'")
        else:
            db.execute("DELETE FROM messages WHERE id='history:turn:t1'")
    with pytest.raises(RuntimeError, match='schema lineage|源记录不一致|缺失或改变'):
        migrate_turn_messages(root)


def test_turn_metadata_cannot_override_independent_handoff_effects(tmp_path):
    root = workspace(tmp_path)
    accepted = user(0)
    handoff(root, accepted)
    forged = user(0, metadata={'effects': {'post_commit': 'suppress'}})
    old = turn(root, 't1', [forged])
    with pytest.raises(ValueError, match='metadata 与入站 receipt 冲突'):
        migrate_turn_messages(root)
    with closing(MessageLog(root / 'sessions.db')) as log:
        assert log.reader('probe:room').snapshot() == ()
        assert log.owner('migration:turn-messages-v1').list() == ()
    with closing(sqlite3.connect(root / 'sessions.db')) as db:
        assert db.execute('SELECT * FROM turns').fetchall() == [old]


@pytest.mark.parametrize('media', [['lost-photo.jpg'], None, False, ''])
def test_migration_refuses_missing_or_malformed_handoff_media(tmp_path, media):
    root = workspace(tmp_path)
    item = user(0)
    turn(root, 't1', [item])
    handoff(root, item)
    with closing(sqlite3.connect(root / 'sessions.db')) as db, db:
        db.execute('UPDATE inbound_handoffs SET media_json=?', (json.dumps(media),))
    with pytest.raises(ValueError, match='media'):
        migrate_turn_messages(root)
    with closing(MessageLog(root / 'sessions.db')) as log:
        assert log.reader('probe:room').snapshot() == ()
        assert log.owner('migration:turn-messages-v1').list() == ()


def test_orphan_turn_does_not_recreate_deleted_session(tmp_path):
    root = workspace(tmp_path)
    turn(root, 't1', [user(0)])
    with closing(sqlite3.connect(root / 'sessions.db')) as db, db:
        db.execute('DELETE FROM sessions')
    with pytest.raises(RuntimeError, match='缺少原 Session'):
        migrate_turn_messages(root)
    with closing(sqlite3.connect(root / 'sessions.db')) as db:
        assert db.execute('SELECT * FROM sessions').fetchall() == []
        assert db.execute('SELECT * FROM messages').fetchall() == []


@pytest.mark.parametrize('conflict', ['text', 'attachment'])
def test_mapped_message_must_match_proven_content_and_attachment_ids(tmp_path, conflict):
    root = workspace(tmp_path)
    original = persisted(root, 0)
    item = user(0)
    if conflict == 'attachment':
        item['data']['metadata']['attachment_ids'] = ['0' * 64]
    else:
        item['data']['content'] = 'different accepted text'
    turn(root, 't1', [item])
    handoff(root, item)
    with pytest.raises(ValueError, match='正文或附件冲突'):
        migrate_turn_messages(root)
    with closing(MessageLog(root / 'sessions.db')) as log:
        assert log.reader('probe:room').snapshot() == (original,)
        assert log.owner('migration:turn-messages-v1').list() == ()

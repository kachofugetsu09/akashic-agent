import json
import re
import runpy
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yoyo

from agent.migrations.context import bind_migration_context
from agent.migrations.mobile_input import migrate
from infra.mobile_realtime.storage import MobileRealtimeStorage
from tests.mobile_realtime.test_storage import _device, _ready_upload


def legacy_database(path, *, rejected_import=False):
    """构造已含 outcome_unknown、尚无失败交接终态的旧 Mobile 数据库。"""
    with closing(MobileRealtimeStorage(path)) as storage:
        storage.register_device(_device())
        storage.create_attachment(_ready_upload(attachment_id='upload-1'))
        storage.prepare_attachment_imports(device_id='device-1', session_id='akashic:chat-1',
                                          client_message_id='client-1', attachment_ids=('upload-1',))
        if rejected_import:
            storage.create_attachment(_ready_upload(attachment_id='upload-2'))
            storage.prepare_attachment_imports(device_id='device-1', session_id='akashic:chat-1',
                                              client_message_id='rejected', attachment_ids=('upload-2',))
        for identity in ('client-1', 'unknown', 'rejected'):
            storage.reserve_command(device_id='device-1', command_id=identity,
                                    command_type='message.send', request_hash=identity, created_at=datetime.now(UTC))
        storage.mark_command_outcome_unknown(device_id='device-1', command_id='unknown')
        storage.complete_command(device_id='device-1', command_id='rejected', reply_type='message.send.error',
                                 reply_payload_json='{"code":"old_rejection"}', session_id='akashic:chat-1',
                                 turn_id=None, completed_at=datetime.now(UTC))
    # 独立复制旧字段，保留旧 CHECK；不调用待测迁移生成夹具。
    with closing(sqlite3.connect(path)) as connection, connection:
        if rejected_import:
            connection.execute("UPDATE mobile_attachment_imports SET phase='artifact_committed', error=NULL WHERE client_message_id='rejected'")
        for table in ('mobile_command_receipts', 'mobile_attachment_imports'):
            sql = connection.execute('SELECT sql FROM sqlite_master WHERE name=?', (table,)).fetchone()[0]
            sql = re.sub(r'\s*handoff_pending INTEGER NOT NULL DEFAULT 0 CHECK\(handoff_pending IN \(0, 1\)\),', '', sql)
            sql = sql.replace(", 'rejected'", '')
            names = ','.join(row[1] for row in connection.execute(f'PRAGMA table_info({table})') if row[1] != 'handoff_pending')
            connection.execute(sql.replace(table, table + '_old', 1))
            connection.execute(f'INSERT INTO {table}_old ({names}) SELECT {names} FROM {table}')
            connection.execute(f'DROP TABLE {table}')
            connection.execute(f'ALTER TABLE {table}_old RENAME TO {table}')
    return snapshot(path)


def snapshot(path):
    with closing(sqlite3.connect(path)) as connection:
        return {name: connection.execute(f'SELECT * FROM {name} ORDER BY rowid').fetchall()
                for name, in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()}


def test_old_mobile_state_requires_yoyo_then_preserves_data_and_rejects_durably(tmp_path):
    path = tmp_path / 'mobile.db'
    before = legacy_database(path)
    with pytest.raises(RuntimeError, match='20260906_05_mobile_input_rejections'):
        MobileRealtimeStorage(path)
    assert snapshot(path) == before
    backup = migrate(path, tmp_path / 'backup')
    assert snapshot(backup) == before
    after = snapshot(path)
    for table, rows in before.items():
        if table != 'mobile_command_receipts':
            assert after[table] == rows
    with closing(MobileRealtimeStorage(path)) as storage:
        assert storage.read_command(device_id='device-1', command_id='unknown').status == 'outcome_unknown'
        assert [r.command_id for r in storage.pending_message_rejections()] == ['rejected']
        storage.complete_command(device_id='device-1', command_id='client-1', reply_type='message.send.error',
            reply_payload_json=json.dumps({'code': 'message_conflict'}), session_id='akashic:chat-1', turn_id=None,
            completed_at=datetime.now(UTC))
        mapping, = storage.list_attachment_imports(session_id='akashic:chat-1', client_message_id='client-1')
        assert mapping.phase == 'rejected' and 'message_conflict' in mapping.error
        assert not storage.list_incomplete_attachment_imports()
        assert storage.cleanup_command_receipts(device_id='device-1', now=datetime.now(UTC) + timedelta(days=20)) == 0
        storage.complete_rejected_message_handoff(device_id='device-1', command_id='client-1')
        assert storage.cleanup_command_receipts(device_id='device-1', now=datetime.now(UTC) + timedelta(days=20)) == 1
    assert migrate(path, tmp_path / 'unused-backup') is None
    assert not (tmp_path / 'unused-backup').exists()


@pytest.mark.parametrize('damage', ['column', 'trigger'])
def test_unknown_mobile_schema_is_rejected_without_data_changes(tmp_path, damage):
    path = tmp_path / 'mobile.db'
    legacy_database(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        if damage == 'column':
            connection.execute('ALTER TABLE mobile_command_receipts ADD COLUMN private_state TEXT')
        else:
            connection.execute('CREATE TRIGGER custom_import AFTER INSERT ON mobile_attachment_imports BEGIN SELECT 1; END')
    before = snapshot(path)
    with pytest.raises(RuntimeError, match='schema lineage|未声明'):
        migrate(path, tmp_path / 'backup')
    assert snapshot(path) == before
    assert not (tmp_path / 'backup').exists()


def test_yoyo_uses_the_configured_mobile_database(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    path = tmp_path / 'custom-mobile.db'
    legacy_database(path)
    config = tmp_path / 'config.toml'
    config.write_text(f'[mobile_realtime]\ndatabase = "{path}"\n')
    monkeypatch.setattr(yoyo, 'step', lambda callback: callback)
    module = runpy.run_path(str(Path(__file__).parents[1] / 'migrations/yoyo/20260906_05_mobile_input_rejections.py'))
    with bind_migration_context(config_path=config, workspace=workspace):
        module['steps'][0](None)
    with closing(MobileRealtimeStorage(path)) as storage:
        assert storage.pending_message_rejections()
    assert list((workspace / 'backups/mobile-input-rejections').glob('*/custom-mobile.db'))


def test_old_rejection_marks_only_its_unbound_import_terminal(tmp_path):
    path = tmp_path / 'mobile.db'
    before = legacy_database(path, rejected_import=True)
    backup = migrate(path, tmp_path / 'backup')
    assert snapshot(backup) == before
    with closing(MobileRealtimeStorage(path)) as storage:
        rejected, = storage.list_attachment_imports(session_id='akashic:chat-1', client_message_id='rejected')
        assert rejected.phase == 'rejected'
        assert rejected.error == '{"code":"old_rejection"}'
        other, = storage.list_attachment_imports(session_id='akashic:chat-1', client_message_id='client-1')
        assert other.phase == 'prepared' and other.error is None
        assert storage.list_incomplete_attachment_imports() == (other,)
        assert len(storage.pending_message_rejections()) == 1

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path

from agent.migrations.session_db_backup import backup_sqlite_database
from infra.mobile_realtime.storage import ATTACHMENT_IMPORT_SCHEMA, COMMAND_RECEIPT_SCHEMA

_PENDING_COLUMN = 'handoff_pending INTEGER NOT NULL DEFAULT 0 CHECK(handoff_pending IN (0, 1)),'
_OLD_RECEIPT_SCHEMA = COMMAND_RECEIPT_SCHEMA.replace(_PENDING_COLUMN, '')
_OLD_IMPORT_SCHEMA = ATTACHMENT_IMPORT_SCHEMA.replace(", 'rejected'", '')


def _sql(value: str) -> str:
    return re.sub(r'\s+', '', value.lower().replace('if not exists', '').replace('"', '')).rstrip(';')


def migrate(path: Path, backup_root: Path) -> Path | None:
    """扩展 Mobile 失败输入状态，保留全部旧行并在 DDL 前生成可恢复快照。"""
    if not path.exists():
        return None
    with closing(sqlite3.connect(f'{path.resolve().as_uri()}?mode=rw', uri=True)) as connection:
        _ = connection.execute('PRAGMA foreign_keys=ON')
        schemas = {
            'mobile_command_receipts': (_OLD_RECEIPT_SCHEMA, COMMAND_RECEIPT_SCHEMA),
            'mobile_attachment_imports': (_OLD_IMPORT_SCHEMA, ATTACHMENT_IMPORT_SCHEMA),
        }
        before: dict[str, tuple[tuple[object, ...], ...]] = {}
        columns: dict[str, tuple[str, ...]] = {}
        old: list[str] = []
        # 1. 核对两个表的已知 lineage；未知 schema、额外索引或 trigger 不得被重建吞掉。
        for table, (previous, current) in schemas.items():
            row = connection.execute('SELECT sql FROM sqlite_master WHERE type = ? AND name = ?', ('table', table)).fetchone()
            if row is None or _sql(row[0]) not in {_sql(previous), _sql(current)}:
                raise RuntimeError(f'{table} schema lineage 不匹配')
            extras = connection.execute(
                'SELECT name FROM sqlite_master WHERE tbl_name = ? AND type IN (?, ?) AND sql IS NOT NULL',
                (table, 'index', 'trigger'),
            ).fetchall()
            if extras:
                raise RuntimeError(f'{table} 存在未声明的索引或 trigger')
            if _sql(row[0]) == _sql(previous):
                old.append(table)
            columns[table] = tuple(row[1] for row in connection.execute(f'PRAGMA table_info({table})'))
            before[table] = tuple(connection.execute(f'SELECT * FROM {table} ORDER BY rowid'))
        if not old:
            return None
        if len(old) != 2:
            raise RuntimeError('Mobile 输入状态 schema 迁移不完整')
        if connection.execute('PRAGMA quick_check').fetchall() != [('ok',)] or connection.execute('PRAGMA foreign_key_check').fetchall():
            raise RuntimeError('Mobile 输入状态迁移前完整性失败')
        backup = backup_sqlite_database(path, backup_root, migration='mobile-input-rejections')
        # 2. 在同一事务重建 CHECK 与保留位；任何失败回滚全部 DDL 和数据。
        _ = connection.execute('BEGIN IMMEDIATE')
        for table in old:
            current = schemas[table][1]
            temporary = f'{table}_input_migration'
            _ = connection.execute(current.replace(table, temporary, 1))
            names = ', '.join(columns[table])
            _ = connection.execute(f'INSERT INTO {temporary} ({names}) SELECT {names} FROM {table} ORDER BY rowid')
            _ = connection.execute(f'DROP TABLE {table}')
            _ = connection.execute(f'ALTER TABLE {temporary} RENAME TO {table}')
            restored = tuple(connection.execute(f'SELECT {names} FROM {table} ORDER BY rowid'))
            if restored != before[table]:
                raise RuntimeError(f'{table} 迁移改变了既有字段或行')
        # 3. 已明确拒绝的旧输入保留回执待结算；附件仅进入逻辑终态，文件和行均保留。
        _ = connection.execute("UPDATE mobile_command_receipts SET handoff_pending = 1 WHERE status = 'completed' AND reply_type = 'message.send.error'")
        _ = connection.execute("""
            UPDATE mobile_attachment_imports AS imports SET phase = 'rejected',
                error = (SELECT reply_payload_json FROM mobile_command_receipts AS receipts
                         WHERE receipts.device_id = imports.device_id AND receipts.command_id = imports.client_message_id)
            WHERE phase IN ('prepared', 'artifact_committed') AND EXISTS (
                SELECT 1 FROM mobile_command_receipts AS receipts WHERE receipts.device_id = imports.device_id
                AND receipts.command_id = imports.client_message_id AND receipts.session_id = imports.session_id
                AND receipts.handoff_pending = 1)
        """)
        if connection.execute('PRAGMA quick_check').fetchall() != [('ok',)] or connection.execute('PRAGMA foreign_key_check').fetchall():
            raise RuntimeError('Mobile 输入状态迁移后完整性失败')
        connection.commit()
    return backup

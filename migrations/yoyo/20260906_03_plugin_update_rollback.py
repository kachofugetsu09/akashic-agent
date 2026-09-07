"""只增插件更新恢复点；旧资源 journal 及事件原样保留。"""
from contextlib import closing
import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database
from agent.plugins.update_rollback import SCHEMA, check_schema

__depends__ = {"20260906_02_session_attributes"}
__transactional__ = False


def migrate_updates(_ledger: object) -> None:
    """备份旧库后一次增加全部表；重复运行不改任何历史记录。"""
    workspace = current_migration_context().workspace
    path = workspace / "runtime/plugin-reloads.sqlite3"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as conn:
        if check_schema(conn):
            return
        # 1. 新表只引用既有 transaction 主键，不解释或改写旧 phase。
        keys = [row for row in conn.execute("PRAGMA table_info(reload_transactions)") if row[5]]
        if len(keys) != 1 or keys[0][1:3] != ("tx_id", "TEXT") or keys[0][5] != 1:
            raise ValueError("更新恢复点迁移缺少已知 reload transaction 主键")
        if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("更新恢复点迁移前完整性检查失败")
    _ = backup_sqlite_database(
        path, workspace / "backups/plugin-update-rollback" / uuid4().hex,
        migration="20260906_03_plugin_update_rollback",
    )
    with closing(sqlite3.connect(path)) as conn, conn:
        _ = conn.execute("PRAGMA foreign_keys=ON")
        _ = conn.execute("BEGIN IMMEDIATE")
        if check_schema(conn):
            return
        transactions = conn.execute("SELECT * FROM reload_transactions ORDER BY tx_id").fetchall()
        events = conn.execute("SELECT * FROM reload_events ORDER BY sequence").fetchall()
        for statement in SCHEMA.values():
            _ = conn.execute(statement)
        # 2. 全部 DDL 与保全检查只有一个提交点；失败整笔回滚。
        if not check_schema(conn):
            raise ValueError("更新恢复点迁移未建立完整 schema")
        if transactions != conn.execute("SELECT * FROM reload_transactions ORDER BY tx_id").fetchall():
            raise ValueError("更新恢复点迁移改变了旧 reload transactions")
        if events != conn.execute("SELECT * FROM reload_events ORDER BY sequence").fetchall():
            raise ValueError("更新恢复点迁移改变了旧 reload events")
        if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)] or conn.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("更新恢复点迁移后完整性检查失败")


steps = [step(migrate_updates)]

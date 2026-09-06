"""Session 固定展示与学习属性，不推断旧来源或改写旧 metadata。"""
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_attributes import migrate

__depends__ = {"20260906_01_scheduler_messages"}
__transactional__ = False


def migrate_session_attributes(_ledger):
    workspace = current_migration_context().workspace
    migrate(workspace / "sessions.db", workspace / "backups/session-attributes" / uuid4().hex)


steps = [step(migrate_session_attributes)]

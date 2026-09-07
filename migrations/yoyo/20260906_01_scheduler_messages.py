"""调度任务与操作、触发回执在同一个候选文件中提交。"""
from yoyo import step

from agent.migrations.context import current_migration_context
from plugins.scheduler.migration import migrate

__depends__ = {"20260906_01_turn_messages"}
__transactional__ = False


def migrate_scheduler_messages(_ledger):
    workspace = current_migration_context().workspace
    migrate(workspace / "schedules.json", workspace / "backups/scheduler-message-state-v2")


steps = [step(migrate_scheduler_messages)]

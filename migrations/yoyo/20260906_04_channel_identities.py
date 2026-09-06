"""一次迁移已知旧渠道的身份路由，不再在运行时扫描 metadata。"""
from uuid import uuid4

from yoyo import step

from agent.migrations.channel_identities import migrate
from agent.migrations.context import current_migration_context

__depends__ = {"20260906_03_plugin_update_rollback"}
__transactional__ = False


def migrate_channel_identities(_ledger: object) -> None:
    context = current_migration_context()
    migrate(
        context.workspace / "sessions.db", context.config_path,
        context.workspace / "backups/channel-identities" / uuid4().hex,
    )


steps = [step(migrate_channel_identities)]

"""只增转换当前有效的旧摘要链，保留全部历史与外部效果证据。"""

from yoyo import step
from agent.migrations.context import current_migration_context
from agent.migrations.legacy_summaries import migrate_legacy_summaries

__depends__ = {"20260907_03_message_metadata", "20260907_03_skill_prompt_grant"}
__transactional__ = False


def migrate(_ledger: object):
    return migrate_legacy_summaries(current_migration_context().workspace)


steps = [step(migrate)]

"""保全旧执行事实；有独立入站证据的 open 输入先暂停再交接。"""
from yoyo import step  # pyright: ignore[reportUnknownVariableType]
from agent.migrations.context import current_migration_context
from agent.migrations.turn_messages import migrate_turn_messages

__depends__ = {"20260905_06_message_artifacts"}
__transactional__ = False


def migrate(_ledger: object):
    return migrate_turn_messages(current_migration_context().workspace)


steps = [step(migrate)]

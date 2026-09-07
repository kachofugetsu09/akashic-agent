"""保留旧学习图，从固定日志上界开始消费新投影。"""
from uuid import uuid4

from yoyo import step

from agent.migrations.akasha_consumption import cutover_akasha
from agent.migrations.context import current_migration_context
from agent.plugins.manifest import builtin_plugin_data_dir
from plugins.akasha.config import load_akasha_config, resolve_memory_path
from plugins.akasha.infrastructure.loader import load_turns
from session.log import MessageLog

__depends__ = {"20260905_03_model_calls"}
__transactional__ = False


def migrate_akasha_consumption(_ledger):
    """运行前由 MigrationRunner 持有 workspace 锁；没有旧图时无旧前缀要迁移。"""
    workspace = current_migration_context().workspace
    config = load_akasha_config(builtin_plugin_data_dir("akasha", workspace) / "config.local.toml")
    memory = resolve_memory_path(workspace / "memory", config.db_path)
    index = resolve_memory_path(workspace / "memory", config.index_path)
    if not memory.exists():
        if index.exists() and load_turns(index):
            raise RuntimeError("非空旧索引缺少学习图，不能推断为新安装")
        return
    if not index.is_file() or not (workspace / "sessions.db").is_file():
        raise RuntimeError("旧学习图缺少固定索引或消息日志")
    log = MessageLog(workspace / "sessions.db")
    try:
        _ = cutover_akasha(
            memory=memory, index=index, heads=log.catalog().snapshot_heads(),
            config=config.memory_config(),
            backup_root=workspace / "backups/akasha-message-consumption-v1" / uuid4().hex,
        )
    finally:
        log.close()


steps = [step(migrate_akasha_consumption)]

import tomllib
from pathlib import Path

from yoyo import step

from agent.migrations.context import current_migration_context

__depends__ = {"20260825_02_select_akasha_embedding_plugin"}
__transactional__ = False


def _backfill_enabled_history(
    *,
    config_path: Path,
    migrated_config: bytes,
    workspace: Path,
) -> None:
    """Load the Akasha implementation only after the enabled boundary."""

    from agent.migrations.akasha_embedding_backfill import (
        backfill_akasha_message_embeddings,
    )

    _ = backfill_akasha_message_embeddings(
        config_path=config_path,
        migrated_config=migrated_config,
        workspace=workspace,
    )


def backfill_akasha_history(_connection: object) -> None:
    """Prepare complete dense SessionDB history before Akasha runtime replay."""

    _ = _connection
    current = current_migration_context()
    if not current.config_path.is_file():
        return
    raw = current.config_path.read_bytes()
    document = tomllib.loads(raw.decode("utf-8"))
    memory = document.get("memory")
    if memory is None:
        return
    if not isinstance(memory, dict):
        raise ValueError("memory 必须是 table")
    if "engine" in memory:
        raise ValueError("memory.engine 自定义选择器已移除；请先迁移为普通插件声明")
    enabled = memory.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("memory.enabled 必须是 boolean")
    if not enabled:
        return
    _backfill_enabled_history(
        config_path=current.config_path,
        migrated_config=raw,
        workspace=current.workspace,
    )


steps = [step(backfill_akasha_history)]

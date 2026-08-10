from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context


__depends__ = {"20260802_01_yoyo_origin"}
__transactional__ = False

_LEGACY_INDEX_VERSION = "8"


def _uses_akasha(config_path: Path) -> bool:
    if not config_path.is_file():
        return False
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    raw_memory = payload.get("memory", {})
    if not isinstance(raw_memory, dict):
        raise ValueError("memory 配置必须是 table")
    memory = cast(dict[str, object], raw_memory)
    return memory.get("enabled") is True and memory.get("engine") == "akasha"


def rebuild_akasha_v9(_connection: object) -> None:
    """Back up, rebuild, and atomically publish Akasha v9 derived state."""

    # 1. 非 Akasha installation 只记录迁移回执，不加载完整 runtime。
    current = current_migration_context()
    if not _uses_akasha(current.config_path):
        return

    # 2. 用当前公共迁移 owner 重建并发布双 sidecar。
    from agent.migrations.akasha_sidecar import rebuild_akasha_sidecars

    backup_dir = (
        current.workspace
        / "backups"
        / "akasha-sparse-index-v9"
        / uuid4().hex
    )
    _ = rebuild_akasha_sidecars(
        config_path=current.config_path,
        workspace=current.workspace,
        backup_dir=backup_dir,
        accepted_versions={_LEGACY_INDEX_VERSION},
    )


steps = [step(rebuild_akasha_v9)]

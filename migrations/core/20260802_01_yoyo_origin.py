from __future__ import annotations

import shutil
from pathlib import Path

from yoyo import step

from agent.migrations.context import current_migration_context


__depends__: set[str] = set()
__transactional__ = False


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def remove_legacy_git_migration_state(_connection: object) -> None:
    """删除退役的 Git cursor 状态，不触碰业务数据。"""

    # 1. 只定位所选配置对应的三类历史 companion
    config_path = current_migration_context().config_path
    legacy_paths = (
        config_path.with_name(f"{config_path.name}.migration-cursor"),
        config_path.with_name(f"{config_path.name}.migration-lock"),
        config_path.with_name(f"{config_path.name}.migration-backups"),
    )

    # 2. 删除精确目标，部分失败后仍可安全重试
    for path in legacy_paths:
        _remove_path(path)

    # 3. 任何目标残留时在 Yoyo 落账前失败
    remaining = [str(path) for path in legacy_paths if path.exists() or path.is_symlink()]
    if remaining:
        raise RuntimeError(f"旧迁移状态删除失败: {remaining}")


steps = [step(remove_legacy_git_migration_state)]

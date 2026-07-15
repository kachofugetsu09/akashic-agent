#!/usr/bin/env python3
"""把旧全局插件数据复制到显式 workspace，并保留原目录。"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from agent.plugins.manifest import (
    ensure_workspace_plugin_data_dir,
    validate_workspace_plugin_data_path,
)
from bootstrap.workspace_lock import WorkspaceInstanceLock


_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def _copy_sqlite(source: Path, destination: Path) -> None:
    """使用 SQLite 在线备份生成一致副本并立即校验。"""

    source_uri = f"file:{source}?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True)) as source_db:
        with closing(sqlite3.connect(destination)) as destination_db:
            source_db.backup(destination_db, pages=256, sleep=0.1)
            result = destination_db.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                raise sqlite3.DatabaseError(
                    f"迁移数据库完整性检查失败: {source} ({result})"
                )
            destination_db.commit()


def _copy_tree(source: Path, destination: Path) -> None:
    """复制一个插件数据目录，拒绝符号链接和特殊文件。"""

    destination.mkdir(parents=True)
    for entry in sorted(source.iterdir(), key=lambda item: item.name):
        if entry.is_symlink():
            raise ValueError(f"插件数据迁移不接受符号链接: {entry}")
        target = destination / entry.name
        if entry.is_dir():
            _copy_tree(entry, target)
        elif entry.is_file():
            if entry.suffix.lower() in _SQLITE_SUFFIXES:
                _copy_sqlite(entry, target)
            else:
                shutil.copy2(entry, target)
        else:
            raise ValueError(f"插件数据迁移不接受特殊文件: {entry}")


def migrate_plugin_data(*, workspace: Path, plugins_home: Path) -> list[Path]:
    """持有 workspace 独占锁复制旧插件数据，失败时不保留目标。"""

    lock = WorkspaceInstanceLock(workspace)
    lock.acquire()
    try:
        return _migrate_plugin_data_locked(
            workspace=workspace,
            plugins_home=plugins_home,
        )
    finally:
        lock.release()


def _migrate_plugin_data_locked(
    *,
    workspace: Path,
    plugins_home: Path,
) -> list[Path]:
    """先完整预复制全部目录，再把验证后的结果发布到 workspace。"""

    # 1. 校验源和目标，避免半迁移后才发现名称冲突
    source_root = plugins_home.expanduser().resolve() / "data"
    workspace_root = workspace.expanduser().resolve()
    target_root = workspace_root / "plugin-data"
    validate_workspace_plugin_data_path(target_root, workspace_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"旧插件数据目录不存在: {source_root}")
    entries = sorted(source_root.iterdir(), key=lambda entry: entry.name)
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"插件数据迁移不接受符号链接: {entry}")
    sources = [entry for entry in entries if entry.is_dir()]
    if not sources:
        raise ValueError(f"旧插件数据目录为空: {source_root}")
    if target_root.exists():
        raise FileExistsError(f"目标插件数据根已存在，拒绝覆盖: {target_root}")

    # 2. 独占锁内完成全部复制和 SQLite 校验，不暴露半成品
    staging_root = workspace_root / f".plugin-data-migrate-{uuid.uuid4().hex}"
    ensure_workspace_plugin_data_dir(staging_root, workspace_root)
    try:
        for source in sources:
            _copy_tree(source, staging_root / source.name)

        # 3. 用一次原子 rename 发布整个数据根，发布后不再执行可失败步骤
        os.replace(staging_root, target_root)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return [target_root / source.name for source in sources]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plugins-home", type=Path, required=True)
    args = parser.parse_args()
    migrated = migrate_plugin_data(
        workspace=args.workspace,
        plugins_home=args.plugins_home,
    )
    for path in migrated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

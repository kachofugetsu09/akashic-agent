#!/usr/bin/env python3
"""把旧全局插件数据复制到显式 workspace，并保留原目录。"""

from __future__ import annotations

import argparse
import os
import re
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
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_PLUGIN_DATA_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


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
            if _is_sqlite_sidecar(entry):
                continue
            if entry.suffix.lower() in _SQLITE_SUFFIXES:
                _copy_sqlite(entry, target)
            else:
                _ = shutil.copy2(entry, target)
        else:
            raise ValueError(f"插件数据迁移不接受特殊文件: {entry}")


def _is_sqlite_sidecar(path: Path) -> bool:
    """识别由 SQLite 主库在线备份取代的事务 sidecar。"""

    name = path.name.lower()
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        if not name.endswith(suffix):
            continue
        database_name = name[: -len(suffix)]
        return Path(database_name).suffix in _SQLITE_SUFFIXES
    return False


def migrate_plugin_data(*, workspace: Path, plugins_home: Path) -> list[Path]:
    """持有 workspace 独占锁复制旧插件数据，失败时不保留目标。"""

    workspace_root = workspace.expanduser().resolve()
    plugins_root = plugins_home.expanduser().resolve()
    lock = WorkspaceInstanceLock(workspace_root)
    lock.acquire()
    try:
        return _migrate_plugin_data_locked(
            workspace=workspace_root,
            plugins_home=plugins_root,
        )
    finally:
        lock.release()


def replace_plugin_data(
    *,
    workspace: Path,
    plugins_home: Path,
    plugin_names: tuple[str, ...],
) -> tuple[list[Path], Path]:
    """离线替换指定插件数据，并保留原 workspace 数据作为备份。"""

    workspace_root = workspace.expanduser().resolve()
    plugins_root = plugins_home.expanduser().resolve()
    lock = WorkspaceInstanceLock(workspace_root)
    lock.acquire()
    try:
        return _replace_plugin_data_locked(
            workspace=workspace_root,
            plugins_home=plugins_root,
            plugin_names=plugin_names,
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
    source_root = plugins_home / "data"
    workspace_root = workspace
    target_root = workspace_root / "plugin-data"
    validate_workspace_plugin_data_path(target_root, workspace_root)
    if source_root.is_symlink() or not source_root.is_dir():
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


def _replace_plugin_data_locked(
    *,
    workspace: Path,
    plugins_home: Path,
    plugin_names: tuple[str, ...],
) -> tuple[list[Path], Path]:
    """预复制并原子替换显式选择的插件目录。"""

    # 1. 完整校验选择范围，禁止路径逃逸或隐式全量覆盖
    if not plugin_names:
        raise ValueError("至少指定一个要替换的插件数据目录")
    if len(set(plugin_names)) != len(plugin_names):
        raise ValueError("插件数据目录不能重复指定")
    for name in plugin_names:
        if _PLUGIN_DATA_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"插件数据目录名称无效: {name!r}")

    source_root = plugins_home / "data"
    workspace_root = workspace
    target_root = workspace_root / "plugin-data"
    validate_workspace_plugin_data_path(target_root, workspace_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise FileNotFoundError(f"旧插件数据根不存在或不安全: {source_root}")
    sources = [source_root / name for name in plugin_names]
    for source in sources:
        if source.is_symlink() or not source.is_dir():
            raise FileNotFoundError(f"旧插件数据目录不存在或不安全: {source}")
        target = target_root / source.name
        validate_workspace_plugin_data_path(target, workspace_root)
        if target.exists() and not target.is_dir():
            raise ValueError(f"workspace 插件数据目标不是目录: {target}")

    # 2. 在独占锁内先构造完整副本，任何复制失败都不触碰线上目录
    operation_id = uuid.uuid4().hex
    staging_root = workspace_root / f".plugin-data-replace-{operation_id}"
    backup_root = workspace_root / "backups" / f"plugin-data-before-replace-{operation_id}"
    backup_data_root = backup_root / "plugin-data"
    ensure_workspace_plugin_data_dir(staging_root, workspace_root)
    validate_workspace_plugin_data_path(backup_root, workspace_root)
    try:
        for source in sources:
            _copy_tree(source, staging_root / source.name)
        ensure_workspace_plugin_data_dir(target_root, workspace_root)
        backup_root.mkdir(parents=True)
        backup_data_root.mkdir()
        _ = (backup_root / "selection.txt").write_text(
            "\n".join(plugin_names) + "\n",
            encoding="utf-8",
        )

        # 3. 每个目录先移入备份，再发布已校验副本；失败时恢复原状态
        backed_up: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            for source in sources:
                target = target_root / source.name
                backup = backup_data_root / source.name
                if target.exists():
                    backed_up.append((backup, target))
                    os.replace(target, backup)
                published.append(target)
                os.replace(staging_root / source.name, target)
        except BaseException:
            for target in reversed(published):
                if target.exists():
                    shutil.rmtree(target)
            for backup, target in reversed(backed_up):
                if backup.exists():
                    os.replace(backup, target)
            if backup_root.exists():
                shutil.rmtree(backup_root)
            raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return [target_root / source.name for source in sources], backup_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--workspace", type=Path, required=True)
    _ = parser.add_argument("--plugins-home", type=Path, required=True)
    _ = parser.add_argument(
        "--replace-plugin",
        action="append",
        default=[],
        metavar="DATA_DIR_NAME",
        help="离线替换一个已存在的插件数据目录，可重复指定",
    )
    args = parser.parse_args()
    if args.replace_plugin:
        migrated, backup_root = replace_plugin_data(
            workspace=args.workspace,
            plugins_home=args.plugins_home,
            plugin_names=tuple(args.replace_plugin),
        )
        print(f"原 workspace 数据备份: {backup_root}")
    else:
        migrated = migrate_plugin_data(
            workspace=args.workspace,
            plugins_home=args.plugins_home,
        )
    for path in migrated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

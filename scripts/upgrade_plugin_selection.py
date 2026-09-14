#!/usr/bin/env python3
"""离线显式建立空 stable；先备份运行选择元数据，不验证或启动组合。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from agent.plugin_composition.archive import sync_directory
from agent.plugins.manifest import load_plugin_manifest
from agent.plugins.selection import PluginSelection, SelectionFormatError, SelectionWriteError
from bootstrap.workspace_lock import PluginPublicationLock, WorkspaceInstanceLock


def _plain(path: Path, *, directory: bool = False) -> None:
    if path.is_symlink() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"元数据路径必须是普通{'目录' if directory else '文件'}: {path}")


def _metadata(workspace: Path, home: Path) -> list[tuple[Path, str]]:
    """仅枚举已知选择文件，不进入制品、归档、消息或 plugin-data。"""
    files = [(home / "manifest.toml", "plugins/manifest.toml")]
    cache = home / "cache"
    if cache.exists() or cache.is_symlink():
        _plain(cache, directory=True)
        for marketplace in sorted(cache.iterdir()):
            _plain(marketplace, directory=True)
            for plugin in sorted(marketplace.iterdir()):
                _plain(plugin, directory=True)
                files.append((plugin / ".pointers.json", f"plugins/cache/{marketplace.name}/{plugin.name}/.pointers.json"))
    files.append((workspace / "runtime/plugin-reloads.sqlite3", "workspace/runtime/plugin-reloads.sqlite3"))
    return files


def _save(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("xb") as stream:
        path.chmod(0o600)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != content:
        raise RuntimeError(f"备份校验失败: {path}")


def initialize_selection(*, workspace: Path, plugins_home: Path, backup_dir: Path) -> Path:
    """持原 workspace 与安装锁，保存恢复点后只创建明确 null。"""
    _plain(workspace, directory=True)
    _plain(plugins_home, directory=True)
    workspace, home = workspace.resolve(), plugins_home.resolve()
    backup_dir = backup_dir.absolute()
    if backup_dir.is_relative_to(workspace) or backup_dir.is_relative_to(home):
        raise ValueError("恢复点必须在 workspace 和 plugin-home 之外")
    _plain(backup_dir.parent, directory=True)
    backup_dir = backup_dir.parent.resolve() / backup_dir.name
    if backup_dir.is_relative_to(workspace) or backup_dir.is_relative_to(home):
        raise ValueError("恢复点父目录不能链接到运行数据内")
    workspace_lock = WorkspaceInstanceLock(workspace)
    workspace_lock.acquire()
    try:
        install_lock = PluginPublicationLock(home)
        install_lock.acquire()
        try:
            selection = PluginSelection(workspace)
            runtime = workspace / "runtime"
            if runtime.exists() or runtime.is_symlink():
                _plain(runtime, directory=True)
            if selection.path.exists() or selection.path.is_symlink():
                raise SelectionFormatError("stable 已存在；不支持覆盖、force 或自动修复")

            # 1. 核对已知元数据；不沿 artifact 指针加载代码。
            files = _metadata(workspace, home)
            for source, _ in files:
                if source.exists() or source.is_symlink():
                    _plain(source)
                    if source.suffix == ".sqlite3":
                        for suffix in ("-wal", "-shm", "-journal"):
                            sidecar = source.with_name(source.name + suffix)
                            if sidecar.exists() or sidecar.is_symlink():
                                _plain(sidecar)
                    if source.name == ".pointers.json":
                        raw = json.loads(source.read_text(encoding="utf-8"))
                        if not isinstance(raw, dict) or set(raw) != {"stable", "latest"} or any(
                            value is not None and not isinstance(value, str) for value in raw.values()
                        ):
                            raise ValueError(f"安装指针格式无效: {source}")
            load_plugin_manifest(home)
            backup_dir.mkdir(mode=0o700)
            sync_directory(backup_dir.parent)

            # 2. SQLite 用只读连接的 backup 纳入 WAL，不复制或 checkpoint 源文件。
            entries = []
            for source, relative in files:
                if not source.exists():
                    entries.append({"source": str(source), "backup": None})
                    continue
                target = backup_dir / relative
                if source.suffix == ".sqlite3":
                    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    target.touch(mode=0o600, exist_ok=False)
                    original = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
                    try:
                        saved = sqlite3.connect(target)
                        try:
                            original.backup(saved)
                            if saved.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                                raise RuntimeError("reload journal 备份完整性检查失败")
                        finally:
                            saved.close()
                    finally:
                        original.close()
                    with target.open("rb") as stream:
                        os.fsync(stream.fileno())
                else:
                    content = source.read_bytes()
                    _save(target, content)
                    if source.read_bytes() != content:
                        raise RuntimeError(f"备份期间元数据变化: {source}")
                entries.append({"source": str(source), "backup": relative,
                                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            _save(backup_dir / "recovery.json", json.dumps({
                "workspace": str(workspace), "plugins_home": str(home),
                "previous_selection": "absent", "files": entries,
                "purpose": "explicit-null-initialization; not a validated composition",
            }, ensure_ascii=False, indent=2).encode())
            for current, _, _ in os.walk(backup_dir, topdown=False):
                sync_directory(Path(current))

            # 3. 恢复点完整耐久后才进入原 primitive；失败不删除备份或回写指针。
            selection.initialize()
            return backup_dir
        finally:
            install_lock.release()
    finally:
        workspace_lock.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--plugins-home", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path,
                        help="运行目录之外的新目录；父目录须已存在")
    args = parser.parse_args()
    try:
        backup = initialize_selection(workspace=args.workspace, plugins_home=args.plugins_home,
                                      backup_dir=args.backup_dir)
    except SelectionWriteError as error:
        parser.exit(1, f"初始化写入失败；恢复点 {args.backup_dir}；outcome={error.outcome}；"
                    f"observed_ref={error.observed_ref}；observation_error={error.observation_error!r}\n")
    print(f"已显式创建 null stable，恢复点：{backup}。尚未验证或晋升任何组合。")


if __name__ == "__main__":
    main()

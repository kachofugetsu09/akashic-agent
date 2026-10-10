#!/usr/bin/env python3
"""离线建立空选择或转换旧归档指针；先备份元数据，不启动插件。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from collections.abc import Mapping
import re
import sys
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.plugins.files import sync_directory
from agent.plugins.manifest import load_plugin_manifest, validate_workspace_plugin_data_path
from agent.plugins.distribution_sources import distribution_plugin_sources
from agent.plugins.source_resolver import scan_plugin_sources
from agent.plugins.input_preparation import _source_revision
from agent.plugin_composition.config_input import load_config
from agent.plugins.python_environment import ENVIRONMENT_FILE, PythonEnvironments
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


def initialize_selection(*, workspace: Path, plugins_home: Path, backup_dir: Path,
                         from_archive: bool = False, plugin_dirs: tuple[Path, ...] = (),
                         distribution: Path | None = None) -> Path:
    """持有离线锁，先保存恢复点再提交选择格式变换。"""
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
            changes: dict[Path, bytes] = {}
            upgraded: dict[str, object] | None = None
            if from_archive:
                upgraded, changes = _plan_archive_upgrade(workspace, home, plugin_dirs, distribution)
            elif selection.path.exists() or selection.path.is_symlink():
                raise SelectionFormatError("stable 已存在；不支持覆盖、force 或自动修复")

            # 1. 核对已知元数据；不沿 artifact 指针加载代码。
            files = _metadata(workspace, home)
            if from_archive:
                files.append((selection.path, "workspace/runtime/plugin-stable.json"))
                for index, path in enumerate(changes):
                    files.append((path, f"changed-metadata/{index}"))
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
                "previous_selection": "archived" if from_archive else "absent", "files": entries,
                "purpose": "explicit-installed-selection-upgrade" if from_archive else "explicit-null-initialization",
                "writes": [str(path) for path in changes],
            }, ensure_ascii=False, indent=2).encode())
            for current, _, _ in os.walk(backup_dir, topdown=False):
                sync_directory(Path(current))

            # 3. 恢复点完整耐久后才进入原 primitive；失败不删除备份或回写指针。
            if upgraded is None:
                selection.initialize()
            else:
                for path, content in changes.items():
                    _replace_metadata(path, content)
                _replace_metadata(selection.path, json.dumps(upgraded, ensure_ascii=False).encode())
                if selection.read() != upgraded["root_ref"]:
                    raise RuntimeError("升级后的选择无法读取；恢复点保留，请勿启动")
            return backup_dir
        finally:
            install_lock.release()
    finally:
        workspace_lock.release()


def _read_old_record(workspace: Path, ref: object) -> dict[str, object]:
    """旧格式只在此离线入口读取，不提供运行归档接口。"""
    if not isinstance(ref, str) or re.fullmatch(r"[0-9a-f]{64}", ref) is None:
        raise ValueError("旧归档记录身份无效")
    root = workspace / "runtime/plugin-archives"
    _plain(root, directory=True)
    path = root / f"{ref}.json"
    _plain(path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != ref:
        raise ValueError("旧归档记录摘要不符")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("旧归档记录必须是对象")
    return value


def _check_old_owners(workspace: Path, components: tuple[str, ...]) -> None:
    """旧运行 owner 未结算时拒绝转换，不能抹去原来的恢复事实。"""
    path = workspace / "runtime/plugin-reloads.sqlite3"
    if not path.exists():
        return
    _plain(path)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as source:
        if source.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise RuntimeError("旧 journal 内容损坏")
        try:
            pending = source.execute("SELECT tx_id FROM reload_transactions WHERE phase NOT IN ('complete','aborted','recovered')").fetchall()
            armed = source.execute("SELECT update_id FROM plugin_updates WHERE phase='armed'").fetchall()
            configs = source.execute("SELECT state,input_ref FROM config_updates WHERE state != 'active'").fetchall()
        except sqlite3.OperationalError as error:
            raise RuntimeError(f"旧 journal 无法读取 owner 状态: {path} ({error})；请先用原 Core 恢复后再升级，不会修改旧库") from error
        if pending or armed or any(state == "accepted" or ref in components for state, ref in configs):
            raise RuntimeError("旧安装、配置或 runtime owner 未结算；请先用原 Core 恢复，不会自动重放")


def _plan_archive_upgrade(workspace: Path, home: Path, plugin_dirs: tuple[Path, ...],
                          distribution: Path | None) -> tuple[dict[str, object], dict[Path, bytes]]:
    """只选择实际安装来源，转换环境引用，不复制代码或历史配置。"""
    # 1. 读取唯一旧选择，保留身份供原回执继续查询。
    path = workspace / "runtime/plugin-stable.json"
    _plain(path)
    old = json.loads(path.read_text())
    if not isinstance(old, dict) or set(old) != {"version", "root_ref"} or old["version"] != 1:
        raise SelectionFormatError("只支持 v1 归档指针升级，不能覆盖当前选择")
    ref = old["root_ref"]
    root = _read_old_record(workspace, ref) if ref is not None else {"components": [], "previous": None}
    components = root["components"]
    if not isinstance(components, list) or len(set(components)) != len(components):
        raise ValueError("旧完整选择格式无效")
    _check_old_owners(workspace, tuple(components))
    configured = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION")
    distribution = distribution or (Path(configured) if configured else None)
    fixed = distribution_plugin_sources(distribution.resolve()) if distribution else ()
    scan = scan_plugin_sources(plugin_dirs, installed_cache_root=home / "cache", fixed_sources=fixed)
    if scan.failures:
        raise RuntimeError(f"实际安装来源不可读: {scan.failures}")
    sources = {(source.source_type, source.plugin_name + ("@" + source.marketplace if source.marketplace else "")): source
               for source in scan.sources}
    changes: dict[Path, bytes] = {}
    converted: dict[str, str] = {}
    environments = workspace / "runtime/plugin-python-environments"

    def environment(old_ref: str) -> str:
        if old_ref in converted:
            return converted[old_ref]
        if re.fullmatch(r"(?:[0-9a-f]{32}|[0-9a-f]{64})", old_ref) is None:
            raise ValueError("旧环境引用无效")
        _plain(environments, directory=True)
        existing = environments / old_ref / "environment.json"
        if existing.exists() or existing.is_symlink():
            PythonEnvironments(workspace).open(old_ref)
            converted[old_ref] = old_ref
            return old_ref
        record = _read_old_record(workspace, old_ref)
        location = record["location"]
        if record["version"] != 1 or not isinstance(location, str) or re.fullmatch(r"[0-9a-f]{32}", location) is None:
            raise ValueError("旧环境目录身份无效")
        _plain(environments, directory=True)
        target = environments / location
        _plain(target, directory=True)
        metadata = target / "environment.json"
        content = json.dumps({"version": 2, "input": record["input"]}, sort_keys=True).encode()
        if metadata.exists() or metadata.is_symlink():
            _plain(metadata)
            if json.loads(metadata.read_bytes()) != json.loads(content):
                raise ValueError("环境元数据与旧安装事实不一致")
        changes[metadata] = content
        converted[old_ref] = location
        return location

    # 2. 缓存索引和实际安装环境文件一起改为目录身份；venv 不移动。
    if environments.exists():
        _plain(environments, directory=True)
        for pointer in sorted(environments.glob("*.ref")):
            _plain(pointer)
            changes[pointer] = environment(pointer.read_text()).encode()
    for source in scan.sources:
        pointer = source.plugin_root / ENVIRONMENT_FILE
        if pointer.exists():
            _plain(pointer)
            refs = json.loads(pointer.read_text())
            if not isinstance(refs, dict):
                raise ValueError("安装环境引用不是对象")
            changes[pointer] = json.dumps({key: environment(value) for key, value in refs.items()}).encode()
    inputs: dict[str, object] = {}
    for component in components:
        record = _read_old_record(workspace, component)
        if record["version"] not in {4, 5}:
            raise ValueError("插件输入版本不支持，须先用原 Core 完成此前升级")
        try:
            source = sources[(cast(str, record["source_type"]), cast(str, record["plugin_id"]))]
        except KeyError as error:
            raise SelectionFormatError(f"当前安装来源缺失: {record['plugin_id']}；请核对 --plugin-dir/--distribution，不会加载旧代码归档") from error
        data_dir = workspace / cast(str, record["data_dir"])
        validate_workspace_plugin_data_path(data_dir, workspace)
        _, revision = load_config(data_dir)
        # 当前配置文件拥有配置；旧 config 正文既不执行，也不复制。
        value = {key: item for key, item in record.items() if key != "config"}
        manifest = source.static_manifest
        if manifest is None:
            raise RuntimeError("已扫描来源缺少 static manifest")
        value.update(version=6, code=str(source.plugin_root.resolve()), source_revision=_source_revision(source.plugin_root),
                     entrypoints=dict(manifest.entrypoints),
                     config_revision=revision,
                     python_environments={key: environment(cast(str, item))
                                          for key, item in cast(Mapping[str, object], record["python_environments"]).items()})
        if source.distribution_source:
            value["distribution_source"] = source.distribution_source
        # 写入任何元数据前，由正常选择边界验证转换结果。
        PluginSelection(workspace).prepare(value)
        inputs[component] = value
    adoption_ref = root.get("distribution_adoption_ref")
    adoption = _read_old_record(workspace, adoption_ref) if adoption_ref is not None else None
    return {"version": 2, "root_ref": ref, "inputs": inputs, "distribution_adoption": adoption,
            "transition": {"base": root["previous"], "components": components} if ref is not None else None}, changes


def _replace_metadata(path: Path, content: bytes) -> None:
    """替换已列入恢复点的元数据；代码、消息与业务配置不在写入范围。"""
    temporary = path.with_name(f".{path.name}.upgrade-{os.getpid()}")
    _save(temporary, content)
    os.replace(temporary, path)
    sync_directory(path.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--plugins-home", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path,
                        help="运行目录之外的新目录；父目录须已存在")
    parser.add_argument("--from-archive", action="store_true", help="显式转换 v1 归档指针，不删除旧归档")
    parser.add_argument("--plugin-dir", action="append", type=Path, default=[], help="当前原生内置插件目录，可重复")
    parser.add_argument("--distribution", type=Path, help="当前发行版；也可用 AKASHIC_PLUGIN_DISTRIBUTION")
    args = parser.parse_args()
    try:
        backup = initialize_selection(workspace=args.workspace, plugins_home=args.plugins_home,
                                      backup_dir=args.backup_dir, from_archive=args.from_archive,
                                      plugin_dirs=tuple(args.plugin_dir), distribution=args.distribution)
    except SelectionWriteError as error:
        parser.exit(1, f"初始化写入失败；恢复点 {args.backup_dir}；outcome={error.outcome}；"
                    f"observed_ref={error.observed_ref}；observation_error={error.observation_error!r}\n")
    print(f"选择元数据已提交，恢复点：{backup}。插件尚未启动；旧归档未删除。")


if __name__ == "__main__":
    main()

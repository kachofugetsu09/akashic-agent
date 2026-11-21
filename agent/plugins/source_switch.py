"""换来源：把发行版内置提供的插件原子切换为已安装的 git artifact，失败可按恢复点还原。"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast
from uuid import uuid4

from agent.plugins.artifacts import ARTIFACTS_DIR, pointer_state_path
from agent.plugins.files import sync_directory
from agent.plugins.input_preparation import prepare_plugin_input
from agent.plugins.install import (
    PluginInstallResult,
    install_git_plugin,
    read_git_plugin_manifest,
    set_artifact_origin,
)
from agent.plugins.manifest import plugins_root
from agent.plugins.python_environment import OfflineWheels
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.selection import PluginSelection, SelectionWriteError
from agent.plugins.static_manifest import load_static_plugin_manifest
from infra.persistence.json_store import atomic_write_text


class SourceSwitchError(RuntimeError):
    """换来源的前置条件不成立，或还原时发现状态已被其他操作改变。"""


# 切换前的完整可恢复状态：选择文件、cache 指针、用户 patch，以及本次会新建的目录和 artifact。
@dataclass(frozen=True)
class SourceSwitchRestorePoint:
    plugin_id: str
    plugins_home: str
    update_id: str
    base_root_ref: str
    selection_text: str
    pointers_text: str | None
    patch_text: str | None
    artifacts_before: tuple[str, ...]
    absent_dirs: tuple[str, ...]
    committed_root_ref: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> SourceSwitchRestorePoint:
        raw = cast(dict[str, str | None | list[str]], json.loads(text))
        return cls(
            plugin_id=cast(str, raw["plugin_id"]), plugins_home=cast(str, raw["plugins_home"]),
            update_id=cast(str, raw["update_id"]), base_root_ref=cast(str, raw["base_root_ref"]),
            selection_text=cast(str, raw["selection_text"]), pointers_text=cast(str | None, raw["pointers_text"]),
            patch_text=cast(str | None, raw["patch_text"]),
            artifacts_before=tuple(cast(list[str], raw["artifacts_before"])),
            absent_dirs=tuple(cast(list[str], raw["absent_dirs"])),
            committed_root_ref=cast(str | None, raw["committed_root_ref"]),
        )


@dataclass(frozen=True)
class SourceSwitchResult:
    install: PluginInstallResult
    root_ref: str
    restore_point: SourceSwitchRestorePoint


# 离线换来源；调用方已持有工作区维护锁与插件发布锁，且实例未在运行。
def switch_plugin_source(
    *, workspace: Path, plugin_id: str, source: str, commit: str, upstream: str | None = None,
    plugins_home: Path | None = None, offline_wheels: OfflineWheels | None = None,
) -> SourceSwitchResult:
    home = (plugins_home or plugins_root()).resolve(strict=False)
    name, _, marketplace = plugin_id.rpartition("@")
    # 1. 准备阶段：只读取代码与选择，不写任何持久状态。
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SourceSwitchError("换来源必须指定完整 40 位提交")
    manifest, revision = read_git_plugin_manifest(source=source, ref_name=commit)
    if not name or not marketplace or manifest.name != name or revision != commit:
        raise SourceSwitchError(f"来源的插件身份与目标不一致: {plugin_id}")
    selection = PluginSelection(workspace)
    base_ref = selection.read()
    if base_ref is None:
        raise SourceSwitchError("选择为空，没有可切换的内置输入")
    old_ref, old = _builtin_input(selection, base_ref, plugin_id)
    # 2. 恢复点：在 install_git_plugin 写任何东西之前保存。
    update_id = uuid4().hex
    point = _capture_restore_point(workspace, home, plugin_id, update_id, base_ref)
    try:
        # 3. 建 artifact 与依赖环境；origin 改回真实上游；一次选择 CAS。
        installed = install_git_plugin(
            workspace=workspace, source=source, marketplace=marketplace, ref_name=commit,
            plugins_home=home, update_id=update_id, offline_wheels=offline_wheels,
            takeover=frozenset({plugin_id}),
        )
        set_artifact_origin(installed.installed_path, upstream or source)
        root_ref = _commit_switch(workspace, selection, installed, base_ref, old_ref, old, update_id)
    except BaseException:
        # 4. CAS 之前的任何失败：还原到切换前，结果与从未尝试过一致。
        restore_source_switch(point, workspace=workspace)
        raise
    return SourceSwitchResult(installed, root_ref, replace(point, committed_root_ref=root_ref))


# 找到目标插件当前的内置输入；已经是 installed 来源就没有可切换的东西。
def _builtin_input(
    selection: PluginSelection, base_ref: str, plugin_id: str,
) -> tuple[str, Mapping[str, object]]:
    for ref in selection.components(base_ref):
        record = selection.read_input(ref)
        if record["plugin_id"] == plugin_id:
            if record["source_type"] != "builtin":
                raise SourceSwitchError(f"目标当前不是内置来源: {plugin_id}")
            return ref, record
    raise SourceSwitchError(f"选择里没有目标插件: {plugin_id}")


# 组装 installed 输入并一次 CAS 提交；同一次提交里移除该插件的历史归属条目。
def _commit_switch(
    workspace: Path, selection: PluginSelection, installed: PluginInstallResult,
    base_ref: str, old_ref: str, old: Mapping[str, object], update_id: str,
) -> str:
    plugin_id = f"{installed.plugin_name}@{installed.marketplace}"
    artifact = installed.installed_path
    prepared = prepare_plugin_input(
        {"name": installed.plugin_name, "plugin_root": str(artifact), "module_path": str(artifact / "plugin.py"),
         "manifest_digest": load_static_plugin_manifest(artifact).identity_digest,
         "marketplace": installed.marketplace, "source_type": "installed"},
        workspace=workspace, selection=selection,
    )
    new = selection.read_input(prepared.input_ref)
    if prepared.plugin_id != plugin_id or new["data_dir"] != old["data_dir"]:
        raise SourceSwitchError(f"安装输入的身份或数据目录与旧输入不一致: {plugin_id}")
    ReloadJournal(workspace).set_input_ref(update_id, prepared.input_ref)
    components = tuple(prepared.input_ref if ref == old_ref else ref for ref in selection.components(base_ref))
    try:
        return selection.commit(
            components, expected_ref=base_ref,
            distribution_adoption=_without_adoption_entry(selection.adoption(), plugin_id),
            replace_adoption=True,
        )
    except SelectionWriteError as error:
        # 写入结果不确定时按落盘事实判定：基线已变说明 CAS 已发布。
        if error.outcome == "uncertain" and error.observed_ref not in (None, base_ref):
            return cast(str, error.observed_ref)
        raise


# 去掉指定插件的归属条目；条目清空时整条凭证置空，其余条目原样保留。
def _without_adoption_entry(
    adoption: Mapping[str, object] | None, plugin_id: str,
) -> Mapping[str, object] | None:
    if adoption is None:
        return None
    entries = cast(Sequence[Mapping[str, object]], adoption["entries"])
    kept = tuple(item for item in entries if item["plugin_id"] != plugin_id)
    if len(kept) == len(entries):
        return adoption
    return {**adoption, "entries": kept} if kept else None


# 保存切换前状态；install_git_plugin 会写的指针、patch、artifact 与新目录都在这里登记。
def _capture_restore_point(
    workspace: Path, home: Path, plugin_id: str, update_id: str, base_ref: str,
) -> SourceSwitchRestorePoint:
    name, _, marketplace = plugin_id.rpartition("@")
    base = home / "cache" / marketplace / name
    artifacts = base / ARTIFACTS_DIR
    candidates = (
        home / "marketplaces", home / "marketplaces" / marketplace, home / "cache", home / "cache" / marketplace,
        base, artifacts,
    )
    return SourceSwitchRestorePoint(
        plugin_id=plugin_id, plugins_home=str(home), update_id=update_id, base_root_ref=base_ref,
        selection_text=_read_text(PluginSelection(workspace).path) or "",
        pointers_text=_read_text(pointer_state_path(base)),
        patch_text=_read_text(workspace / "bundle.patch.toml"),
        artifacts_before=tuple(sorted(item.name for item in artifacts.iterdir())) if artifacts.is_dir() else (),
        absent_dirs=tuple(str(path) for path in candidates if not path.exists()),
    )


# 按恢复点还原选择、指针、patch，并清掉本次新建的 artifact 与空目录；切换成功后也可调用。
def restore_source_switch(point: SourceSwitchRestorePoint, *, workspace: Path) -> None:
    home = Path(point.plugins_home)
    name, _, marketplace = point.plugin_id.rpartition("@")
    base = home / "cache" / marketplace / name
    selection = PluginSelection(workspace)
    # 1. 选择：只有基线或本次提交的结果才允许还原，其他值说明被别的操作改过。
    current = selection.read()
    if current == point.committed_root_ref and current != point.base_root_ref:
        _write_selection_text(selection.path, point.selection_text)
    elif current != point.base_root_ref:
        raise SourceSwitchError("选择已被其他操作改变，拒绝还原")
    # 2. 指针与 patch 逐字还原；先选择后指针，避免选择引用已被删除的 artifact。
    _restore_text(pointer_state_path(base), point.pointers_text)
    _restore_text(workspace / "bundle.patch.toml", point.patch_text)
    # 3. 删除本次新建的 artifact 与空目录，并结算安装恢复点。
    artifacts = base / ARTIFACTS_DIR
    if artifacts.is_dir():
        for item in artifacts.iterdir():
            if item.name not in point.artifacts_before:
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    for path in reversed(point.absent_dirs):
        try:
            Path(path).rmdir()
        except OSError:
            pass
    ReloadJournal(workspace).revert_committed_install(point.update_id, "source switch restored")


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _restore_text(path: Path, text: str | None) -> None:
    if text is None:
        path.unlink(missing_ok=True)
        if path.parent.is_dir():
            sync_directory(path.parent)
        return
    atomic_write_text(path, text, domain="plugin_source_switch")


# 选择文件的权限与写入方式和 PluginSelection 一致：同目录临时文件、刷盘、原子替换。
def _write_selection_text(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=".plugin-stable-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        _ = stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)
    sync_directory(path.parent)


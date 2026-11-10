"""workspace patch 的组合迁移：插件拆分或改名后把用户选择交给继任 row；清掉和 preset 相同、只是抄来的字段。"""
from __future__ import annotations

import os
from typing import Any, cast

import tomlkit

from agent.migrations.context import current_migration_context
from agent.plugins.bundles import BundleRow, preset_row_for, preset_rows
from core.common.frozen_json import freeze_json
from infra.persistence.json_store import atomic_write_text


# 读取 workspace patch；没有 patch 时返回 None，格式不对时拒绝。
def _read_patch() -> tuple[str, tomlkit.TOMLDocument, dict[str, Any]] | None:
    path = current_migration_context().workspace / "bundle.patch.toml"
    if path.is_symlink():
        raise ValueError(f"组合迁移不接受符号链接: {path}")
    if not path.exists():
        return None
    before = path.read_text()
    patch = tomlkit.parse(before)
    rows = patch.get("rows")
    if set(patch) != {"schema_version", "rows"} or patch["schema_version"] != 1 or not isinstance(rows, dict):
        raise ValueError("bundle patch 格式无效")
    return before, patch, cast(dict[str, Any], rows)


# 先留恢复点再写入；同一标签的恢复点只记录第一次迁移前的内容。
def _write_patch(before: str, patch: tomlkit.TOMLDocument, label: str) -> None:
    workspace = current_migration_context().workspace
    backup = workspace / "runtime" / f"bundle.patch.before-{label}.toml"
    if not backup.exists():
        atomic_write_text(backup, before, domain="bundle_migration")
    atomic_write_text(workspace / "bundle.patch.toml", tomlkit.dumps(patch), domain="bundle_migration")


# 1. 读取 workspace patch；2. 前任有显式 row 时按同样的启用/停用补齐继任 row；3. 先留恢复点再写入。
# 显式启用也要继承：minimal 等默认停用继任的 bundle 上，用户启用前任得到的能力不能因拆分消失。
def inherit_choice(predecessor: str, successors: tuple[str, ...], label: str) -> None:
    read = _read_patch()
    if read is None:
        return
    before, patch, rows = read
    plugins = [str(row["plugin"]) for row in rows.values()]
    chosen = [row for row in rows.values() if str(row["plugin"]).split("@")[0] == predecessor]
    if not chosen:
        return
    disabled = chosen[0].get("disabled") is True
    marketplace = str(chosen[0]["plugin"]).partition("@")[2] or "release"
    added = [name for name in successors
             if name not in rows and all(plugin.split("@")[0] != name for plugin in plugins)]
    if not added:
        return
    for name in added:
        row = tomlkit.table()
        row["plugin"] = f"{name}@{marketplace}"
        if disabled:
            row["disabled"] = True
        rows[name] = row
    _write_patch(before, patch, label)


# 插件改名：前任 row 原样（停用、市场后缀等字段）交给继任；继任已有 row 时只移除前任。
def rename_row(predecessor: str, successor: str, label: str) -> None:
    read = _read_patch()
    if read is None:
        return
    before, patch, rows = read
    found = [key for key, row in rows.items() if str(row["plugin"]).split("@")[0] == predecessor]
    if not found:
        return
    taken = successor in rows or any(str(row["plugin"]).split("@")[0] == successor for row in rows.values())
    for key in found:
        row = rows.pop(key)
        if not taken:
            marketplace = str(row["plugin"]).partition("@")[2]
            row["plugin"] = successor if not marketplace else f"{successor}@{marketplace}"
            rows[successor] = row
            taken = True
    _write_patch(before, patch, label)


# 早期迁移和旧版启停会把 preset 的 disabled/config 原样抄进 patch，挡住 preset 以后的更新。
# 1. 只看覆盖某条 preset row、且插件相同的 row；2. 删掉和 preset 相同的 disabled、config；
# 3. 只剩 plugin 的 row 整行删除；4. 先留恢复点再写入。和 preset 不同的字段是用户的选择，原样保留。
def follow_preset(label: str) -> None:
    bundles = current_migration_context().bundle_directory
    read = _read_patch()
    if bundles is None or read is None:
        return
    before, patch, rows = read
    preset = {row.id: row for row in preset_rows(bundles, mode=os.environ.get("AKASHIC_PLUGIN_BUNDLE", "base"))}
    changed = False
    for identity in list(rows):
        row = rows[identity]
        base = preset_row_for(preset, BundleRow(identity, str(row.get("plugin")), {}))
        if base is None or row.get("plugin") != base.plugin:
            continue
        if "disabled" in row and row["disabled"] == base.disabled:
            del row["disabled"]
            changed = True
        if "config" in row and freeze_json(row["config"].unwrap()) == base.config:
            del row["config"]
            changed = True
        if set(row) == {"plugin"}:
            del rows[identity]
            changed = True
    if changed:
        _write_patch(before, patch, label)

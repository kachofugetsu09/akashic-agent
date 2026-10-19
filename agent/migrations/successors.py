"""插件拆分后，把用户对前任插件的停用选择复制给继任 row；只补缺，不覆盖已有选择。"""
from __future__ import annotations

import tomlkit

from agent.migrations.context import current_migration_context
from infra.persistence.json_store import atomic_write_text


# 1. 读取 workspace patch；2. 前任被停用时补齐继任停用 row；3. 先留恢复点再写入。
def inherit_disabled(predecessor: str, successors: tuple[str, ...], label: str) -> None:
    workspace = current_migration_context().workspace
    path = workspace / "bundle.patch.toml"
    if path.is_symlink():
        raise ValueError(f"组合迁移不接受符号链接: {path}")
    if not path.exists():
        return
    before = path.read_text()
    patch = tomlkit.parse(before)
    rows = patch.get("rows")
    if set(patch) != {"schema_version", "rows"} or patch["schema_version"] != 1 or not isinstance(rows, dict):
        raise ValueError("bundle patch 格式无效")
    plugins = [str(row["plugin"]) for row in rows.values()]
    stopped = [str(row["plugin"]) for row in rows.values()
               if str(row["plugin"]).split("@")[0] == predecessor and row.get("disabled") is True]
    if not stopped:
        return
    marketplace = stopped[0].partition("@")[2] or "release"
    added = [name for name in successors
             if name not in rows and all(plugin.split("@")[0] != name for plugin in plugins)]
    if not added:
        return
    for name in added:
        row = tomlkit.table()
        row["plugin"] = f"{name}@{marketplace}"
        row["disabled"] = True
        rows[name] = row
    backup = workspace / "runtime" / f"bundle.patch.before-{label}.toml"
    if not backup.exists():
        atomic_write_text(backup, before, domain="bundle_migration")
    atomic_write_text(path, tomlkit.dumps(patch), domain="bundle_migration")

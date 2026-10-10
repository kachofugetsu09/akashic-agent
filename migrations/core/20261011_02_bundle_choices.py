"""把旧启停选择复制到 workspace patch，保留原文和可重试的写入计划。"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
import tomllib

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context
from infra.persistence.json_store import atomic_write_text

__depends__ = {"20260928_01_plugin_config_updates"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:@[A-Za-z0-9][A-Za-z0-9._-]*)?")


def _text(path: Path) -> str | None:
    if path.is_symlink():
        raise ValueError(f"组合迁移不接受符号链接: {path}")
    return path.read_text() if path.exists() else None


def _build_plan(config_path: Path, workspace: Path, manifest: Path, bundles: Path | None) -> dict:
    """冻结旧输入和完整目标；未知配置字段不删除。"""
    config_before = config_path.read_text()
    config = tomlkit.parse(config_before)
    saved_config = config_path.with_name(config_path.name + ".before-bundle-choices.toml")
    legacy_config = _text(saved_config)
    if legacy_config is None:
        legacy_config = config_before
    legacy = _text(manifest)
    choices = {}
    if legacy is not None:
        entries = tomllib.loads(legacy)["plugins"]
        if not isinstance(entries, dict):
            raise ValueError("旧 manifest.plugins 必须为表")
        for identity, item in entries.items():
            if _ID.fullmatch(identity) is None or not isinstance(item, dict) or type(item.get("enabled")) is not bool:
                raise ValueError(f"旧启停条目无效: {identity}")
            choices[identity] = item["enabled"]
    # 1. 固定 base 的 row 身份；mode 和用户 patch 仍在它之后覆盖。
    defaults = {} if bundles is None else tomllib.loads((bundles / "base.toml").read_text())["rows"]
    names = {row["plugin"].split("@")[0]: row["plugin"] for row in defaults.values()}
    receipt = workspace / "runtime/distribution-install.json"
    if receipt.exists():
        for row in json.loads(receipt.read_text())["installed"]:
            names.setdefault(row["name"], row["name"] + "@" + row["marketplace"])
    agent = config.get("agent", {})
    plugins = agent.get("plugins", {})
    disabled = tomllib.loads(legacy_config).get("agent", {}).get("plugins", {}).get("disabled_builtin", [])
    if "disabled_builtin" in plugins and plugins["disabled_builtin"] != disabled:
        raise RuntimeError("共享配置的旧启停选择已改变，不能覆盖原恢复点")
    if not isinstance(disabled, list) or any(not isinstance(item, str) or _ID.fullmatch(item) is None for item in disabled):
        raise ValueError("旧 disabled_builtin 必须是插件身份数组")
    for identity in disabled:
        choices[names.get(identity, identity)] = False
    plugins.pop("disabled_builtin", None)
    if not plugins:
        agent.pop("plugins", None)
    if not agent:
        config.pop("agent", None)
    # 2. 显式用户 patch 优先；旧开关只填补没有对应 plugin 的 row。
    patch_before = _text(workspace / "bundle.patch.toml")
    patch = tomlkit.parse(patch_before or "schema_version = 1\n[rows]\n")
    if set(patch) != {"schema_version", "rows"} or patch["schema_version"] != 1:
        raise ValueError("bundle patch 格式无效")
    rows = patch["rows"]
    existing = {row["plugin"] for row in rows.values()}
    by_plugin = {row["plugin"]: (identity, row) for identity, row in defaults.items()}
    for plugin, enabled in sorted(choices.items()):
        if plugin in existing:
            continue
        identity, default = by_plugin.get(plugin, (plugin, {"plugin": plugin}))
        if identity in rows:
            # 用户在同一 row 更换 provider，同样优先于旧开关。
            continue
        rows[identity] = {**default, "disabled": not enabled}
    return {"config_path": str(config_path), "legacy_manifest": legacy, "legacy_config": legacy_config,
            "config_before": config_before, "config_after": tomlkit.dumps(config),
            "patch_before": patch_before, "patch_after": tomlkit.dumps(patch)}


def upgrade(connection: sqlite3.Connection) -> None:
    """先保留恢复计划，再逐个 CAS 替换；中断后只接受原值或已写入的目标值。"""
    context = current_migration_context()
    workspace, config = context.workspace, context.config_path
    journal = workspace / "runtime/plugin-reloads.sqlite3"
    if journal.exists():
        with sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True) as db:
            if db.execute("SELECT 1 FROM plugin_updates WHERE phase='armed' LIMIT 1").fetchone():
                raise RuntimeError("先用旧版本结算 armed 安装，再迁移启停选择")
    manifest = context.plugins_home / "manifest.toml"
    backup = workspace / "runtime/before-bundle-choices.json"
    if not backup.exists():
        plan = _build_plan(config, workspace, manifest, context.bundle_directory)
        # 配置可以被多个 workspace 共享；原字段的恢复点必须跟随配置文件。
        saved_config = config.with_name(config.name + ".before-bundle-choices.toml")
        if _text(saved_config) is None:
            atomic_write_text(saved_config, plan["legacy_config"], domain="bundle_migration")
        if saved_config.read_text() != plan["legacy_config"]:
            raise RuntimeError("共享配置恢复点已变化")
        atomic_write_text(backup, json.dumps(plan, ensure_ascii=False, indent=2) + "\n", domain="bundle_migration")
    plan = json.loads(backup.read_text())
    if plan["config_path"] != str(config) or _text(manifest) != plan["legacy_manifest"]:
        raise RuntimeError("组合迁移的原配置路径或旧清单已变化")
    targets = ((config, "config"), (workspace / "bundle.patch.toml", "patch"))
    for path, label in targets:
        if _text(path) not in (plan[label + "_before"], plan[label + "_after"]):
            raise RuntimeError(f"组合迁移输入已变化: {path}")
    # patch 先提交，删除旧字段前已经保留其全部选择。
    for path, label in reversed(targets):
        if _text(path) != plan[label + "_after"]:
            atomic_write_text(path, plan[label + "_after"], domain="bundle_migration")
    for path, label in targets:
        if path.read_text() != plan[label + "_after"]:
            raise RuntimeError(f"组合迁移写入核对失败: {path}")


steps = [step(upgrade, None)]

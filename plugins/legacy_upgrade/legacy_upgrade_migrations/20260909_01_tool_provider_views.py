"""把技能材料授权迁到实际 standard_tools owner。"""

from __future__ import annotations

import tomllib

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context
from .support.plugin_data import builtin_plugin_data_dir, validate_workspace_plugin_data_path
from agent.plugin_contracts.json_store import atomic_write_text

__depends__ = {"20260908_01_legacy_summaries"}
__transactional__ = False


def move_skill_prompt_owner(_ledger: object) -> None:
    """只改准确的旧 owner，并保存可恢复原文。"""
    context = current_migration_context()
    path = builtin_plugin_data_dir("context", context.workspace) / "config.local.toml"
    validate_workspace_plugin_data_path(path, context.workspace)
    if not path.exists():
        return
    before = path.read_text(encoding="utf-8")
    data = tomllib.loads(before)
    prompt_sources = data.get("prompt_sources")
    if prompt_sources is None:
        return
    if not isinstance(prompt_sources, dict):
        raise ValueError("context.prompt_sources 必须是 TOML table")
    owner = prompt_sources.get("skills")
    if owner != "skills":
        return
    backup = path.with_name("config.before-tool-provider-views.toml")
    if backup.exists():
        if backup.read_text(encoding="utf-8") != before:
            raise ValueError("工具 provider 迁移备份与当前配置不一致")
    else:
        atomic_write_text(backup, before)
    document = tomlkit.parse(before)
    sources = document.get("prompt_sources")
    if not isinstance(sources, dict) or sources.get("skills") != "skills":
        raise ValueError("context.prompt_sources.skills 文档结构与预检不一致")
    sources["skills"] = "standard_tools"
    atomic_write_text(path, tomlkit.dumps(document))


steps = [step(move_skill_prompt_owner)]

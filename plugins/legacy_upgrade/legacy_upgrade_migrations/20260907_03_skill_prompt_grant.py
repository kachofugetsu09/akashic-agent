"""为内置默认材料配置授权技能 Prompt；自定义授权保持原样。"""
from yoyo import step

from agent.migrations.context import current_migration_context
from .support.plugin_data import (
    builtin_plugin_data_dir,
    validate_workspace_plugin_data_path,
)
from agent.plugin_contracts.json_store import atomic_write_text

__depends__ = {"20260907_02_retire_legacy_agent_config"}
__transactional__ = False


def grant_skills(_ledger):
    """只更新已知默认授权，先保存原配置；恢复或重试不会覆盖备份。"""
    context = current_migration_context()
    path = builtin_plugin_data_dir("context", context.workspace) / "config.local.toml"
    validate_workspace_plugin_data_path(path, context.workspace)
    if not path.exists():
        return
    before = path.read_text(encoding="utf-8")
    expected = (
        'prompt_sources = {default_prompt = "prompt", markdown_memory = "markdown_memory"}\n'
        'summary_source = ["compaction", "compaction"]\n'
    )
    if before != expected:
        return
    backup = path.with_name("config.before-skill-prompt-grant.toml")
    if backup.exists():
        if backup.read_text(encoding="utf-8") != before:
            raise ValueError("技能授权备份与当前配置不一致")
    else:
        atomic_write_text(backup, before)
    atomic_write_text(path, (
        'prompt_sources = {default_prompt = "prompt", markdown_memory = "markdown_memory", skills = "skills"}\n'
        'summary_source = ["compaction", "compaction"]\n'
    ))


steps = [step(grant_skills)]

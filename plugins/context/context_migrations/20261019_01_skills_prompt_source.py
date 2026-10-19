"""技能提示词从 standard_tools 移到 skills 插件：把本插件配置里的 skills 授权改给新 owner。"""
from __future__ import annotations

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.plugin_composition.config_input import CONFIG_INPUT, load_config, save_config

__depends__: set[str] = set()
__transactional__ = False


# 只改写指向 standard_tools 的 skills 授权并保留市场后缀；其他来源与用户改过的值不动。
def move_skills_source(connection: object) -> None:
    data_root = current_migration_context().bundle_data_roots["context"]
    if not (data_root / CONFIG_INPUT).exists():
        return
    config, _ = load_config(data_root)
    sources = config.get("prompt_sources")
    if not isinstance(sources, dict):
        return
    current = sources.get("skills")
    if not isinstance(current, str) or current.split("@")[0] != "standard_tools":
        return
    _, at, marketplace = current.partition("@")
    save_config(data_root, {**config, "prompt_sources": {**sources, "skills": "skills" + at + marketplace}})


steps = [step(move_skills_source)]

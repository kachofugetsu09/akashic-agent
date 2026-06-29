from __future__ import annotations

from typing import Any

from agent.skills import SkillsLoader
from agent.tools.base import Tool


class LoadSkillTool(Tool):
    name = "load_skill"
    description = "按 skill 名称加载完整 SKILL.md 指令。用于使用 Skills 目录里列出的技能。"
    parameters = {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "要加载的 skill 名称，例如 memory 或 akasha:memory",
            },
        },
        "required": ["skill"],
    }

    def __init__(self, skills: SkillsLoader) -> None:
        self._skills = skills

    async def execute(self, skill: str, **kwargs: Any) -> str:
        name = skill.strip()
        if not name:
            return "错误：缺少 skill 名称。"

        record = self._skills.load_skill_record(name)
        if record is None:
            available = [
                item.name
                for item in self._skills.list_skill_records(filter_unavailable=False)
            ]
            if not available:
                return f"错误：未找到 skill：{name}。当前没有可用的 skill。"
            return f"错误：未找到 skill：{name}。\n已发现 skill：{', '.join(available)}"

        if not record.available:
            return f"错误：skill 不可用：{name}。\n缺失依赖：{record.missing}"

        content = self._skills.load_skill_body(name)
        if not content:
            return f"错误：skill 内容为空：{name}。"

        return (
            f"# Skill: {record.name}\n\n"
            f"Source: {record.source}\n"
            f"Base directory: {record.root_dir.resolve()}\n\n"
            "如果本 skill 提到相对路径，请按 Base directory 拼接后读取。\n\n"
            "---\n\n"
            f"{content}"
        )

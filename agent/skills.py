import json
import os
import re
import shutil
from pathlib import Path

# 内置技能目录（项目 skills/ 文件夹）
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillsLoader:
    """技能加载器：管理 workspace 和内置两套技能目录，
    负责列举、加载、过滤以及生成 system prompt 摘要。"""

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        self.workspace = workspace
        # 用户自定义技能目录（优先级高于内置）
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """列举所有技能。

        按优先级合并 workspace 技能和内置技能（同名时 workspace 优先），
        可选过滤掉依赖不满足的技能。

        Args:
            filter_unavailable: 为 True 时过滤掉依赖不满足的技能。

        Returns:
            技能信息列表，每项包含 'name'、'path'、'source'。
        """
        skills = []

        # 先扫描用户自定义技能（优先级最高）
        if self.workspace_skills.exists():
            for skill_dir in self.workspace_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "workspace"})

        # 再扫描内置技能，同名的已被 workspace 覆盖则跳过
        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists() and not any(s["name"] == skill_dir.name for s in skills):
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "builtin"})

        # 过滤依赖不满足的技能
        if filter_unavailable:
            return [s for s in skills if self._check_requirements(self._get_skill_config(s["name"]))]
        return skills

    def _check_requirements(self, skill_config: dict) -> bool:
        """检查技能的运行依赖是否满足（CLI 工具 + 环境变量）。"""
        requires = skill_config.get("requires", {})
        # 检查所需 CLI 工具是否在 PATH 中
        for b in requires.get("bins", []):
            if not shutil.which(b):
                return False
        # 检查所需环境变量是否已设置
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """将指定技能的完整内容加载到 system prompt 中（剥除 frontmatter）。

        用于 always=true 技能或用户显式请求的技能，直接内嵌到上下文里。

        Args:
            skill_names: 要加载的技能名称列表。

        Returns:
            格式化后的技能内容字符串，多个技能之间用分隔线隔开。
        """
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                content = self._strip_frontmatter(content)
                parts.append(f"### Skill: {name}\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def load_skill(self, name: str) -> str | None:
        """按名称读取技能的原始 SKILL.md 内容（含 frontmatter）。

        查找顺序：workspace 技能 → 内置技能。

        Args:
            name: 技能目录名，如 "weather"、"summarize"。

        Returns:
            SKILL.md 的文本内容，未找到则返回 None。
        """
        # 优先查 workspace 自定义技能
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            return workspace_skill.read_text(encoding="utf-8")

        # 回退到内置技能
        if self.builtin_skills:
            builtin_skill = self.builtin_skills / name / "SKILL.md"
            if builtin_skill.exists():
                return builtin_skill.read_text(encoding="utf-8")

        return None

    def _strip_frontmatter(self, content: str) -> str:
        """剥除 SKILL.md 头部的 YAML frontmatter，只保留正文。"""
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end():].strip()
        return content

    def get_skill_metadata(self, name: str) -> dict | None:
        """读取技能 frontmatter 中的键值对（简单 YAML 解析）。

        Args:
            name: 技能名称。

        Returns:
            frontmatter 键值字典，未找到或解析失败则返回 None。
        """
        content = self.load_skill(name)
        if not content:
            return None

        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                # 简单逐行解析，足够处理 SKILL.md 中的扁平结构
                metadata = {}
                for line in match.group(1).split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"\'')
                return metadata

        return None

    def _get_skill_config(self, name: str) -> dict:
        """从 frontmatter 的 metadata 字段中提取技能配置（always、requires 等）。"""
        meta = self.get_skill_metadata(name) or {}
        return self._parse_skill_config(meta.get("metadata", ""))

    def _parse_skill_config(self, raw: str) -> dict:
        """解析 frontmatter 中 metadata 字段的 JSON 配置。

        配置结构示例：
            {"akasic": {"always": true, "requires": {"bins": ["curl"]}}}

        为兼容其他格式，同时识别 "akasic" 和 "skill" 顶层键。

        Args:
            raw: frontmatter 中 metadata 字段的原始字符串。

        Returns:
            技能配置字典，解析失败则返回空字典。
        """
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {}
            # 依次尝试已知的顶层键
            for key in ("akasic", "skill"):
                if key in data:
                    return data[key]
            # 兼容：直接返回整个对象（旧格式）
            return data
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_always_skills(self) -> list[str]:
        """返回所有标记了 always=true 且依赖满足的技能名称列表。

        always 技能在每轮对话都会直接注入完整内容到 system prompt，
        适合需要无条件生效的核心指令（如记忆、个人助手规则等）。
        """
        result = []
        for s in self.list_skills(filter_unavailable=True):
            meta = self.get_skill_metadata(s["name"]) or {}
            skill_config = self._parse_skill_config(meta.get("metadata", ""))
            if skill_config.get("always") or meta.get("always"):
                result.append(s["name"])
        return result

    def build_skills_summary(self) -> str:
        """生成所有技能的 XML 摘要，注入到 system prompt 的 Skills 章节。

        每条技能包含名称、描述、SKILL.md 文件路径和可用状态。
        模型看到摘要后，可通过 read_file 读取对应 SKILL.md 获取完整指令。

        Returns:
            XML 格式的技能列表字符串，无技能时返回空字符串。
        """
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for s in all_skills:
            name = escape_xml(s["name"])
            path = s["path"]
            desc = escape_xml(self._get_skill_description(s["name"]))
            skill_config = self._get_skill_config(s["name"])
            available = self._check_requirements(skill_config)

            lines.append(f"  <skill available=\"{str(available).lower()}\">")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")

            # 不可用时列出缺失的依赖，方便用户排查
            if not available:
                missing = self._get_missing_requirements(skill_config)
                if missing:
                    lines.append(f"    <requires>{escape_xml(missing)}</requires>")

            lines.append(f"  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    def _get_missing_requirements(self, skill_config: dict) -> str:
        """返回未满足的依赖描述（CLI 工具或环境变量）。"""
        missing = []
        requires = skill_config.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                missing.append(f"CLI: {b}")
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)

    def _get_skill_description(self, name: str) -> str:
        """从 frontmatter 读取技能描述，未设置时回退为技能名称。"""
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name

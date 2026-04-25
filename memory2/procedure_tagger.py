"""
memory2/procedure_tagger.py — 为 procedure 条目生成 trigger_tags。

与打标迁移脚本（scripts/tag_procedures.py）使用相同的 prompt，
保证线上写入质量和离线迁移质量一致。
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent.provider import LLMProvider

logger = logging.getLogger(__name__)

# 与 scripts/tag_procedures.py 保持一致
KNOWN_TOOLS = [
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "shell",
    "web_fetch",
    "web_search",
    "memorize",
    "message_push",
    "schedule",
    "cancel_schedule",
    "list_schedules",
    "spawn",
    "tool_search",
    "update_now",
    "mcp_fitbit__fitbit_health_snapshot",
    "mcp_fitbit__fitbit_sleep_report",
]

_SYSTEM_PROMPT = "你是一个记忆标注助手，输出严格的 JSON，不加任何额外文字。"

_USER_PROMPT_TEMPLATE = """\
你的任务是分析一条操作规范（procedure），为检索系统生成触发关键字标签。

## 系统注册的工具（tool）列表
{tool_list}

## 系统可用的技能（skill）列表
{skill_list}

## 待标注的操作规范
{summary}

## 输出格式（仅输出 JSON，不加其他内容）
{{
  "tools": [],      // 触发该规范的工具名，必须严格来自上面工具列表，没有则为 []
  "skills": [],     // 触发该规范的技能名，必须严格来自上面技能列表，没有则为 []
  "keywords": [],   // shell 命令关键词或路径关键词（如 "pacman"、"pip install"、"git"），不与 tools/skills 重复
  "scope": "tool_triggered" // 或 "global"：global 表示无论执行什么操作都应遵守的通用规范
}}

## 示例

规范：Agent 执行技能安装缺失依赖时，必须使用 `sudo pacman -S --noconfirm` 静默安装
输出：{{"tools": ["shell"], "skills": [], "keywords": ["pacman", "--noconfirm"], "scope": "tool_triggered"}}

规范：用户希望搜索时自动执行先搜后抓全文流程
输出：{{"tools": ["web_search", "web_fetch"], "skills": [], "keywords": [], "scope": "tool_triggered"}}

规范：生成 RSS 订阅链接前必须先读取 rsshub-route-finder 技能的 SKILL.md
输出：{{"tools": [], "skills": ["rsshub-route-finder"], "keywords": [], "scope": "tool_triggered"}}

规范：工具尝试失败两次后必须收敛并反馈用户，禁止无限试错
输出：{{"tools": [], "skills": [], "keywords": [], "scope": "global"}}

## 注意
- 只输出 JSON 对象，不加 markdown 代码块或任何说明
- tools 必须严格来自工具列表，不可自创
- skills 必须严格来自技能列表，不可自创
- keywords 聚焦于 shell 命令词、路径关键词等可精确匹配的字符串
"""

MAX_TOKENS = 128  # JSON 输出很小，128 足够


class ProcedureTagger:
    """为 procedure 条目生成 trigger_tags，保证写入质量与离线迁移脚本一致。"""

    def __init__(
        self,
        provider: "LLMProvider",
        model: str,
        skills_fn: Callable[[], list[str]],
        tools: list[str] | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._skills_fn = skills_fn
        self._tools = tools or KNOWN_TOOLS

    async def tag(self, summary: str) -> dict | None:
        """为一条 procedure summary 生成 trigger_tags。失败时返回 None。"""
        skills = self._skills_fn()
        prompt = _USER_PROMPT_TEMPLATE.format(
            tool_list="\n".join(f"- {t}" for t in self._tools),
            skill_list="\n".join(f"- {s}" for s in skills) if skills else "（暂无）",
            summary=summary,
        )
        try:
            resp = await self._provider.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=[],
                model=self._model,
                max_tokens=MAX_TOKENS,
            )
            raw = (resp.content or "").strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            tag = json.loads(raw)
            return _validate(tag, set(self._tools), set(skills))
        except Exception as e:
            logger.warning("[procedure_tagger] tag 失败: %s", e)
            return None


def _validate(tag: dict, valid_tools: set[str], valid_skills: set[str]) -> dict:
    tools = [t for t in (tag.get("tools") or []) if t in valid_tools]
    skills = [s for s in (tag.get("skills") or []) if s in valid_skills]
    keywords = [
        k
        for k in (tag.get("keywords") or [])
        if isinstance(k, str) and len(k.strip()) >= 3
    ]
    scope = tag.get("scope", "tool_triggered")
    if scope not in ("tool_triggered", "global"):
        scope = (
            "global" if not tools and not skills and not keywords else "tool_triggered"
        )
    if scope == "global" and (tools or skills or keywords):
        scope = "tool_triggered"
    return {"tools": tools, "skills": skills, "keywords": keywords, "scope": scope}

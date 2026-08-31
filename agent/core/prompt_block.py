from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from agent.persona import read_veda
from agent.prompting import PromptSectionRender, SectionCache
from prompts.agent import (
    build_agent_behavior_rules_prompt,
    build_agent_session_context_prompt,
    build_agent_static_identity_prompt,
    build_skills_catalog_prompt,
)

if TYPE_CHECKING:
    from agent.skills import SkillsLoader


@dataclass
class TurnContext:
    workspace: Path
    skills: "SkillsLoader"
    skill_names: list[str]
    channel: str | None
    chat_id: str | None


class PromptBlock(Protocol):
    priority: int
    label: str
    is_static: bool

    def render(
        self, ctx: TurnContext, cached_signature: str | None = None
    ) -> str | None: ...

    def cache_signature(self, ctx: TurnContext) -> str | None: ...


# ─── Prompt Block 渲染顺序（priority 升序 = system prompt 拼接顺序）────────────
#   5 VedaPromptBlock         → memory/VEDA.md
#                              来源：用户明确维护的 workspace 人格真源
#                              时机：每次组装都重新读取，本轮修改从下一轮生效
#  10 IdentityPromptBlock      → build_agent_static_identity_prompt(workspace)
#                              来源：工作区路径、memory/* 文件索引
#                              时机：仅 workspace 变化时才变，最稳定
#  15 BehaviorRulesPromptBlock → build_agent_behavior_rules_prompt(workspace)
#                              来源：prompts/agent.py 里的固定行为规范
#                              时机：仅代码或 workspace 变化时才变，最稳定
#  20 SkillsCatalogPromptBlock → skills.build_skills_summary()
#                              来源：skills/ 目录扫描结果、技能描述、依赖可用性
#                              时机：技能文件或环境依赖变化时才变，低频
#  30 plugin prompt sections   → self model
#                              来源：普通插件追加的有序 prompt section
#                              时机：自我认知被写回时才变，低频
#  35 plugin prompt sections   → long-term memory
#                              来源：普通插件追加的有序 prompt section
#                              时机：长期记忆 consolidate 或人工更新时才变，低频
#  40 SessionContextPromptBlock→ 环境 + 当前 session
#                              来源：platform.machine() + channel + chat_id
#                              时机：切换机器架构、channel、chat_id 时才变；同 session 基本稳定
#  45 ActiveSkillsPromptBlock  → active skill 内容
#                              来源：always skills + 本轮命中的 skill_names
#                              时机：本轮技能命中集合变化时就会变，中频
#  50 MemoryBlockPromptBlock   → 本轮语义检索注入
#                              来源：普通 Prompt lifecycle section
#                              时机：每轮 retrieval 结果都可能不同，最高频
# ─────────────────────────────────────────────────────────────────────────────
class VedaPromptBlock:
    priority = 5
    label = "veda"
    is_static = False

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str:
        return read_veda(ctx.workspace)

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return None


class IdentityPromptBlock:
    priority = 10
    label = "identity"
    is_static = True

    def __init__(self, render_fn=build_agent_static_identity_prompt) -> None:
        self._render_fn = render_fn

    def render(
        self, ctx: TurnContext, cached_signature: str | None = None
    ) -> str | None:
        return self._render_fn(workspace=ctx.workspace)

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return str(ctx.workspace.expanduser().resolve())


class BehaviorRulesPromptBlock:
    priority = 15
    label = "behavior_rules"
    is_static = True

    def __init__(self, render_fn=build_agent_behavior_rules_prompt) -> None:
        self._render_fn = render_fn

    def render(
        self, ctx: TurnContext, cached_signature: str | None = None
    ) -> str | None:
        return self._render_fn(workspace=ctx.workspace)

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return str(ctx.workspace.expanduser().resolve())


class SkillsCatalogPromptBlock:
    priority = 20
    label = "skills_catalog"
    is_static = True

    def __init__(self, render_fn=build_skills_catalog_prompt) -> None:
        self._render_fn = render_fn

    def render(
        self, ctx: TurnContext, cached_signature: str | None = None
    ) -> str | None:
        summary = cached_signature or ""
        if not summary:
            return None
        return self._render_fn(summary)

    def cache_signature(self, ctx: TurnContext) -> str | None:
        summary = ctx.skills.build_skills_summary()
        return summary or None


class SessionContextPromptBlock:
    priority = 40
    label = "session_context"
    is_static = False

    def __init__(self, render_fn=build_agent_session_context_prompt) -> None:
        self._render_fn = render_fn

    def render(
        self, ctx: TurnContext, cached_signature: str | None = None
    ) -> str | None:
        return self._render_fn(
            channel=ctx.channel,
            chat_id=ctx.chat_id,
        )

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return None


class ActiveSkillsPromptBlock:
    priority = 50
    label = "active_skills"
    is_static = False

    def render(
        self, ctx: TurnContext, cached_signature: str | None = None
    ) -> str | None:
        always_skills = ctx.skills.get_always_skills()
        names: list[str] = []
        seen: set[str] = set()
        for name in [*always_skills, *ctx.skill_names]:
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
        if not names:
            return None
        content = ctx.skills.load_skills_for_context(names)
        if not content:
            return None
        return f"# Active Skills\n\n{content}"

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return None


class SystemPromptBuilder:
    """
    ┌──────────────────────────────────────┐
    │ SystemPromptBuilder                  │
    ├──────────────────────────────────────┤
    │ 1. 按 priority 遍历 prompt blocks    │
    │ 2. 读取 static block cache           │
    │ 3. 渲染启用的 blocks                 │
    │ 4. 输出有序 prompt sections          │
    └──────────────────────────────────────┘
    """

    def __init__(
        self,
        blocks: list[PromptBlock],
        cache: SectionCache | None = None,
    ) -> None:
        self._blocks = sorted(blocks, key=lambda block: block.priority)
        self._cache = cache or SectionCache()

    def build(
        self,
        ctx: TurnContext,
        *,
        disabled_sections: set[str] | None = None,
    ) -> list[PromptSectionRender]:
        # 1. 先准备输出容器和禁用集合。
        renders: list[PromptSectionRender] = []
        disabled = disabled_sections or set()
        cache_scope = str(ctx.workspace.expanduser().resolve())

        # 2. 再逐个渲染 prompt block。
        for block in self._blocks:
            if block.label in disabled:
                continue
            cache_hit = False
            rendered: str | None = None
            signature = block.cache_signature(ctx) if block.is_static else None

            # 3. static block 先查缓存，避免重复读文件或重复构造。
            if signature:
                rendered = self._cache.get(cache_scope, block.label, signature)
                cache_hit = rendered is not None
            if rendered is None:
                rendered = block.render(ctx, cached_signature=signature)
                if rendered and signature:
                    self._cache.set(cache_scope, block.label, signature, rendered)

            # 4. 最后只收录真正有内容的 block。
            if rendered:
                renders.append(
                    PromptSectionRender(
                        name=block.label,
                        content=rendered,
                        is_static=block.is_static,
                        cache_hit=cache_hit,
                        order=block.priority,
                    )
                )

        return renders

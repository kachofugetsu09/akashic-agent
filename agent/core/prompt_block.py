from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from agent.memes.catalog import MemeCatalog
from agent.prompting import PromptSectionMeta, PromptSectionRender, SectionCache
from prompts.agent import (
    build_agent_behavior_rules_prompt,
    build_agent_session_context_prompt,
    build_agent_static_identity_prompt,
    build_skills_catalog_prompt,
)

if TYPE_CHECKING:
    from agent.skills import SkillsLoader
    from core.memory.profile import ProfileReader

logger = logging.getLogger("agent.core.prompt_block")


@dataclass
class TurnContext:
    workspace: Path
    memory: "ProfileReader"
    skills: "SkillsLoader"
    skill_names: list[str]
    channel: str | None
    chat_id: str | None
    message_timestamp: datetime | None
    retrieved_memory_block: str


class PromptBlock(Protocol):
    priority: int
    label: str
    is_static: bool

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None: ...

    def cache_signature(self, ctx: TurnContext) -> str | None: ...


# ─── Prompt Block 渲染顺序（priority 升序 = system prompt 拼接顺序）────────────
#  10 IdentityPromptBlock      → 工作区路径 + 文件索引         (static, cacheable)
#  15 BehaviorRulesPromptBlock → 行为规范 + 历史检索协议        (static, cacheable)
#  20 MemoryBlockPromptBlock   → 本轮语义检索注入              (dynamic)
#  30 LongTermMemoryPromptBlock→ 长期 profile                 (dynamic)
#  40 SelfModelPromptBlock     → SELF.md                     (dynamic)
#  50 SessionContextPromptBlock→ 当前时间 + 环境 + channel     (dynamic)
#  60 ActiveSkillsPromptBlock  → active skill 内容            (dynamic)
#  65 MemesPromptBlock         → meme catalog                 (dynamic)
#  70 SkillsCatalogPromptBlock → 技能目录                     (static, cacheable)
# ─────────────────────────────────────────────────────────────────────────────
class IdentityPromptBlock:
    priority = 10
    label = "identity"
    is_static = True

    def __init__(self, render_fn=build_agent_static_identity_prompt) -> None:
        self._render_fn = render_fn

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None:
        return self._render_fn(workspace=ctx.workspace)

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return str(ctx.workspace.expanduser().resolve())


class BehaviorRulesPromptBlock:
    priority = 15
    label = "behavior_rules"
    is_static = True

    def __init__(self, render_fn=build_agent_behavior_rules_prompt) -> None:
        self._render_fn = render_fn

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None:
        return self._render_fn(workspace=ctx.workspace)

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return str(ctx.workspace.expanduser().resolve())


class MemoryBlockPromptBlock:
    priority = 20
    label = "retrieved_memory"
    is_static = False

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None:
        block = (ctx.retrieved_memory_block or "").strip()
        return block or None

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return None


class LongTermMemoryPromptBlock:
    priority = 30
    label = "long_term_memory"
    is_static = False

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None:
        memory = ctx.memory.read_profile()
        return str(memory).strip() if memory else None

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return None


class SelfModelPromptBlock:
    priority = 40
    label = "self_model"
    is_static = False

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None:
        self_content = ctx.memory.read_self()
        if not self_content:
            return None
        return f"## Akashic 自我认知\n\n{self_content}"

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return None


class RecentContextPromptBlock:
    priority = 45
    label = "recent_context"
    is_static = False

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None:
        content = ctx.memory.read_recent_context()
        if not content:
            return None
        # Strip ## Recent Turns section — it mirrors the sliding window and causes overlap.
        marker = "\n## Recent Turns"
        cut = content.find(marker)
        trimmed = content[:cut].strip() if cut != -1 else content.strip()
        return trimmed if trimmed else None

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return None


class SessionContextPromptBlock:
    priority = 50
    label = "session_context"
    is_static = False

    def __init__(self, render_fn=build_agent_session_context_prompt) -> None:
        self._render_fn = render_fn

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None:
        return self._render_fn(
            message_timestamp=ctx.message_timestamp,
            channel=ctx.channel,
            chat_id=ctx.chat_id,
        )

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return None


class ActiveSkillsPromptBlock:
    priority = 60
    label = "active_skills"
    is_static = False

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None:
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


class MemesPromptBlock:
    priority = 65
    label = "memes"
    is_static = False

    def __init__(self, catalog: MemeCatalog) -> None:
        self._catalog = catalog

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None:
        block = self._catalog.build_prompt_block()
        if not block:
            return None
        return f"# Memes\n\n{block}"

    def cache_signature(self, ctx: TurnContext) -> str | None:
        return None


class SkillsCatalogPromptBlock:
    priority = 70
    label = "skills_catalog"
    is_static = True

    def __init__(self, render_fn=build_skills_catalog_prompt) -> None:
        self._render_fn = render_fn

    def render(self, ctx: TurnContext, cached_signature: str | None = None) -> str | None:
        summary = cached_signature or ""
        if not summary:
            return None
        return self._render_fn(summary)

    def cache_signature(self, ctx: TurnContext) -> str | None:
        summary = ctx.skills.build_skills_summary()
        return summary or None


@dataclass
class SystemPromptBuildResult:
    system_sections: list[PromptSectionRender]
    system_prompt: str
    debug_breakdown: list[PromptSectionMeta]


class SystemPromptBuilder:
    """
    ┌──────────────────────────────────────┐
    │ SystemPromptBuilder                  │
    ├──────────────────────────────────────┤
    │ 1. 按 priority 遍历 prompt blocks    │
    │ 2. 读取 static block cache           │
    │ 3. 渲染启用的 blocks                 │
    │ 4. 汇总 system prompt               │
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
    ) -> SystemPromptBuildResult:
        # 1. 先准备输出容器和禁用集合。
        renders: list[PromptSectionRender] = []
        breakdown: list[PromptSectionMeta] = []
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
                    )
                )
                breakdown.append(
                    PromptSectionMeta(
                        name=block.label,
                        chars=len(rendered),
                        est_tokens=max(1, len(rendered) // 3),
                        is_static=block.is_static,
                        cache_hit=cache_hit,
                    )
                )

        return SystemPromptBuildResult(
            system_sections=renders,
            system_prompt="\n\n---\n\n".join(item.content for item in renders),
            debug_breakdown=breakdown,
        )

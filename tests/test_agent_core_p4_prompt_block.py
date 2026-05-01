from __future__ import annotations
from typing import Any, cast

from pathlib import Path
from types import SimpleNamespace

from agent.core.prompt_block import (
    ActiveSkillsPromptBlock,
    BehaviorRulesPromptBlock,
    IdentityPromptBlock,
    LongTermMemoryPromptBlock,
    MemoryBlockPromptBlock,
    RecentContextPromptBlock,
    SelfModelPromptBlock,
    SessionContextPromptBlock,
    SkillsCatalogPromptBlock,
    SystemPromptBuilder,
    TurnContext,
)
from prompts.agent import build_agent_static_identity_prompt


class _Memory:
    def read_profile(self) -> str:
        return "memory block"

    def read_self(self) -> str:
        return "self note"


class _Skills:
    def get_always_skills(self) -> list[str]:
        return ["always"]

    def load_skills_for_context(self, names: list[str]) -> str:
        return "\n".join(names)

    def build_skills_summary(self) -> str:
        return "summary"


def test_system_prompt_builder_uses_prompt_blocks_and_static_cache(tmp_path: Path):
    builder = SystemPromptBuilder(
        [
            IdentityPromptBlock(render_fn=lambda **_: "identity"),
            MemoryBlockPromptBlock(),
        ]
    )
    ctx = TurnContext(
        workspace=tmp_path,
        memory=cast(Any, _Memory()),
        skills=cast(Any, _Skills()),
        skill_names=[],
        channel=None,
        chat_id=None,
        retrieved_memory_block="retrieved",
    )

    first = builder.build(ctx)
    second = builder.build(ctx)

    assert first.system_prompt == "identity\n\n---\n\nretrieved"
    assert [item.name for item in first.system_sections] == ["identity", "retrieved_memory"]
    assert second.debug_breakdown[0].cache_hit is True


def test_system_prompt_builder_respects_disabled_sections(tmp_path: Path):
    builder = SystemPromptBuilder(
        [
            IdentityPromptBlock(render_fn=lambda **_: "identity"),
            MemoryBlockPromptBlock(),
        ]
    )
    ctx = TurnContext(
        workspace=tmp_path,
        memory=cast(Any, _Memory()),
        skills=cast(Any, _Skills()),
        skill_names=[],
        channel=None,
        chat_id=None,
        retrieved_memory_block="retrieved",
    )

    built = builder.build(ctx, disabled_sections={"retrieved_memory"})

    assert built.system_prompt == "identity"
    assert [item.name for item in built.system_sections] == ["identity"]


def test_static_identity_prompt_is_not_hardcoded_to_specific_user(tmp_path: Path):
    prompt = build_agent_static_identity_prompt(workspace=tmp_path)

    assert "花月的长期 AI 伙伴" not in prompt
    assert "用户的长期 AI 伙伴" in prompt


def test_prompt_block_priorities_leave_spacing_for_future_inserts():
    priorities = [
        (IdentityPromptBlock.label, IdentityPromptBlock.priority),
        (BehaviorRulesPromptBlock.label, BehaviorRulesPromptBlock.priority),
        (SkillsCatalogPromptBlock.label, SkillsCatalogPromptBlock.priority),
        (SelfModelPromptBlock.label, SelfModelPromptBlock.priority),
        (LongTermMemoryPromptBlock.label, LongTermMemoryPromptBlock.priority),
        (SessionContextPromptBlock.label, SessionContextPromptBlock.priority),
        (RecentContextPromptBlock.label, RecentContextPromptBlock.priority),
        (ActiveSkillsPromptBlock.label, ActiveSkillsPromptBlock.priority),
        (MemoryBlockPromptBlock.label, MemoryBlockPromptBlock.priority),
    ]

    assert priorities == [
        ("identity", 10),
        ("behavior_rules", 15),
        ("skills_catalog", 20),
        ("self_model", 30),
        ("long_term_memory", 35),
        ("session_context", 40),
        ("recent_context", 45),
        ("active_skills", 50),
        ("retrieved_memory", 55),
    ]

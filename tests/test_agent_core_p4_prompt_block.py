from __future__ import annotations
from typing import Any, cast

from pathlib import Path
from types import SimpleNamespace

from agent.core.prompt_block import (
    ActiveSkillsPromptBlock,
    BehaviorRulesPromptBlock,
    IdentityPromptBlock,
    SessionContextPromptBlock,
    SkillsCatalogPromptBlock,
    SystemPromptBuilder,
    TurnContext,
    VedaPromptBlock,
)
from prompts.agent import (
    build_agent_static_identity_prompt,
    build_current_session_prompt,
)


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
        ]
    )
    ctx = TurnContext(
        workspace=tmp_path,
        skills=cast(Any, _Skills()),
        skill_names=[],
        channel=None,
        chat_id=None,
    )

    first = builder.build(ctx)
    second = builder.build(ctx)

    assert [item.content for item in first] == ["identity"]
    assert [item.name for item in first] == ["identity"]
    assert second[0].cache_hit is True


def test_system_prompt_builder_respects_disabled_sections(tmp_path: Path):
    builder = SystemPromptBuilder(
        [
            IdentityPromptBlock(render_fn=lambda **_: "identity"),
        ]
    )
    ctx = TurnContext(
        workspace=tmp_path,
        skills=cast(Any, _Skills()),
        skill_names=[],
        channel=None,
        chat_id=None,
    )

    built = builder.build(ctx, disabled_sections={"identity"})

    assert built == []


def test_static_identity_prompt_exposes_veda_edit_boundary(tmp_path: Path):
    prompt = build_agent_static_identity_prompt(workspace=tmp_path)

    assert f"{tmp_path.resolve()}/memory/VEDA.md" in prompt
    assert "只有用户明确要求修改人格或 Veda 时" in prompt
    assert "用户的长期 AI 伙伴" not in prompt


def test_veda_prompt_block_reloads_after_each_turn_build(tmp_path: Path):
    path = tmp_path / "memory/VEDA.md"
    path.parent.mkdir(parents=True)
    path.write_text("first veda", encoding="utf-8")
    builder = SystemPromptBuilder([VedaPromptBlock()])
    ctx = TurnContext(
        workspace=tmp_path,
        skills=cast(Any, _Skills()),
        skill_names=[],
        channel=None,
        chat_id=None,
    )

    first = builder.build(ctx)
    path.write_text("second veda", encoding="utf-8")
    second = builder.build(ctx)

    assert [item.content for item in first] == ["first veda"]
    assert [item.content for item in second] == ["second veda"]
    assert second[0].cache_hit is False


def test_current_session_prompt_distinguishes_web_and_android_surfaces():
    web = build_current_session_prompt(channel="web", chat_id="desktop-chat")
    mobile = build_current_session_prompt(channel="mobile", chat_id="phone-chat")

    assert "Channel: web" in web
    assert "Chat ID: desktop-chat" in web
    assert "Client Surface: WebChat" in web
    assert "Client Device Context: 电脑网页端" in web
    assert "Channel: mobile" in mobile
    assert "Chat ID: phone-chat" in mobile
    assert "Client Surface: Akashic Android" in mobile
    assert "Client Device Context: Android 手机端" in mobile


def test_current_session_prompt_does_not_guess_unknown_channel_surface():
    prompt = build_current_session_prompt(
        channel="custom_web_bridge",
        chat_id="raw-chat-id",
    )

    assert "Channel: custom_web_bridge" in prompt
    assert "Chat ID: raw-chat-id" in prompt
    assert "Client Surface: Unknown" in prompt
    assert "Client Device Context: Unknown" in prompt
    assert "Client Surface: WebChat" not in prompt
    assert "Client Surface: Akashic Android" not in prompt


def test_prompt_block_priorities_leave_spacing_for_future_inserts():
    priorities = [
        (VedaPromptBlock.label, VedaPromptBlock.priority),
        (IdentityPromptBlock.label, IdentityPromptBlock.priority),
        (BehaviorRulesPromptBlock.label, BehaviorRulesPromptBlock.priority),
        (SkillsCatalogPromptBlock.label, SkillsCatalogPromptBlock.priority),
        (SessionContextPromptBlock.label, SessionContextPromptBlock.priority),
        (ActiveSkillsPromptBlock.label, ActiveSkillsPromptBlock.priority),
    ]

    assert priorities == [
        ("veda", 5),
        ("identity", 10),
        ("behavior_rules", 15),
        ("skills_catalog", 20),
        ("session_context", 40),
        ("active_skills", 50),
    ]

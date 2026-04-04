from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from agent.core.prompt_block import (
    IdentityPromptBlock,
    MemoryBlockPromptBlock,
    SystemPromptBuilder,
    TurnContext,
)


class _Memory:
    def get_memory_context(self) -> str:
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
        memory=_Memory(),
        skills=_Skills(),
        skill_names=[],
        message_timestamp=datetime(2026, 4, 4, 21, 0, 0),
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
        memory=_Memory(),
        skills=_Skills(),
        skill_names=[],
        message_timestamp=None,
        retrieved_memory_block="retrieved",
    )

    built = builder.build(ctx, disabled_sections={"retrieved_memory"})

    assert built.system_prompt == "identity"
    assert [item.name for item in built.system_sections] == ["identity"]

from __future__ import annotations

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest

from agent.plugin_composition import (
    SKILLS,
    CompositionError,
    CompositionRoot,
    Context,
    PluginRuntime,
    PluginSkills,
)


class _LeakedPluginSkills(PluginSkills):
    def _register(
        self,
        plugin_id: str,
        kind: Literal["skill", "drift_skill"],
        path: Path,
    ) -> Callable[[], None]:
        _ = super()._register(plugin_id, kind, path)
        return lambda: None


def _runtime(plugin_dir: Path) -> PluginRuntime:
    return PluginRuntime(
        plugin_id=plugin_dir.name,
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        workspace=plugin_dir / "workspace",
        config=object(),
    )


@pytest.mark.asyncio
async def test_plugin_skills_freezes_normal_and_drift_lanes(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "skills_probe"
    normal = plugin_dir / "skills"
    drift = plugin_dir / "drift" / "skills"
    normal.mkdir(parents=True)
    drift.mkdir(parents=True)
    root = CompositionRoot("skills-lanes")
    skills = PluginSkills()
    _ = await root.context.provide(SKILLS, skills)
    plugin_ctx: Context | None = None

    async def plugin(ctx: Context) -> None:
        nonlocal plugin_ctx
        plugin_ctx = ctx
        service = ctx.require(SKILLS)
        await service.register(ctx, "skills")
        await service.register(ctx, "drift/skills", drift=True)

    _ = await root.mount(
        plugin,
        name="skills-probe",
        inject=(SKILLS,),
        runtime=_runtime(plugin_dir),
    )
    assert plugin_ctx is not None

    frozen = skills.freeze()

    assert frozen["skills_probe"].skill_roots == (normal,)
    assert frozen["skills_probe"].drift_skill_roots == (drift,)
    with pytest.raises(CompositionError) as caught:
        _ = await skills.register(plugin_ctx, "skills")
    assert caught.value.code == "PLUGIN_SKILLS_FROZEN"
    await root.dispose()
    assert root.receipt().effects == ()


@pytest.mark.asyncio
async def test_plugin_skills_rejects_symlink_escape(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "skills_escape"
    outside = tmp_path / "outside"
    plugin_dir.mkdir()
    outside.mkdir()
    (plugin_dir / "skills").symlink_to(outside, target_is_directory=True)
    root = CompositionRoot("skills-escape")
    skills = PluginSkills()
    _ = await root.context.provide(SKILLS, skills)

    async def plugin(ctx: Context) -> None:
        await ctx.require(SKILLS).register(ctx, "skills")

    _ = await root.mount(
        plugin,
        name="skills-escape",
        inject=(SKILLS,),
        runtime=_runtime(plugin_dir),
    )

    receipt = root.receipt()
    assert receipt.ready is False
    assert any("Skill root 路径越界" in error for error in receipt.errors)
    await root.dispose()


@pytest.mark.asyncio
async def test_plugin_skills_rolls_back_duplicate_root(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "skills_duplicate"
    (plugin_dir / "skills").mkdir(parents=True)
    root = CompositionRoot("skills-duplicate")
    skills = PluginSkills()
    _ = await root.context.provide(SKILLS, skills)

    async def plugin(ctx: Context) -> None:
        service = ctx.require(SKILLS)
        await service.register(ctx, "skills")
        await service.register(ctx, "skills")

    _ = await root.mount(
        plugin,
        name="skills-duplicate",
        inject=(SKILLS,),
        runtime=_runtime(plugin_dir),
    )

    receipt = root.receipt()
    assert receipt.ready is False
    assert any("Skill root 重复" in error for error in receipt.errors)
    assert dict(skills.freeze()) == {}
    await root.dispose()


@pytest.mark.asyncio
async def test_plugin_skills_oracle_kills_leaked_registration_mutant(
    tmp_path: Path,
) -> None:
    correct = await _disposed_registration_fixture(tmp_path / "correct", PluginSkills)
    mutant = await _disposed_registration_fixture(
        tmp_path / "mutant",
        _LeakedPluginSkills,
    )

    assert correct is False
    assert mutant is True


async def _disposed_registration_fixture(
    plugin_dir: Path,
    service_type: type[PluginSkills],
) -> bool:
    """Dispose one provider Fiber before freezing its remaining declarations."""

    skill_root = plugin_dir / "skills"
    skill_root.mkdir(parents=True)
    root = CompositionRoot(f"skills-dispose:{service_type.__name__}")
    skills = service_type()
    _ = await root.context.provide(SKILLS, skills)

    async def plugin(ctx: Context) -> None:
        await ctx.require(SKILLS).register(ctx, "skills")

    fiber = await root.mount(
        plugin,
        name="skills-owner",
        inject=(SKILLS,),
        runtime=_runtime(plugin_dir),
    )
    await fiber.dispose()
    leaked = bool(skills.freeze())
    await root.dispose()
    return leaked

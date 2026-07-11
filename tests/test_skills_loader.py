from pathlib import Path

import pytest

from agent.skills import SkillsLoader
from agent.tools.skill_loader import LoadSkillTool


def _write_skill(
    skills_dir: Path,
    name: str,
    *,
    description: str = "测试技能",
    body: str = "正文",
    extra_frontmatter: str = "",
) -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    extra = f"{extra_frontmatter}\n" if extra_frontmatter else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"{extra}"
        f"---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def test_skill_index_prefers_workspace_over_builtin(tmp_path: Path):
    workspace = tmp_path / "workspace"
    builtin = tmp_path / "builtin"
    _write_skill(builtin, "memory", description="builtin", body="builtin body")
    _write_skill(
        workspace / "skills",
        "memory",
        description="workspace",
        body="workspace body",
    )

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    records = loader.list_skill_records(filter_unavailable=False)
    assert [record.name for record in records] == ["memory"]
    assert records[0].source == "workspace"
    assert loader.load_skill_body("memory") == "workspace body"


def test_skills_summary_hides_file_locations(tmp_path: Path):
    workspace = tmp_path / "workspace"
    skill_dir = _write_skill(
        workspace / "skills",
        "memory",
        description="处理记忆任务时使用。",
        body="body",
        extra_frontmatter="when_to_use: 用户询问记忆时。",
    )

    summary = SkillsLoader(workspace, builtin_skills_dir=tmp_path / "builtin").build_skills_summary()

    assert '<skill name="memory" available="true" source="workspace">' in summary
    assert "<when_to_use>用户询问记忆时。</when_to_use>" in summary
    assert "<location>" not in summary
    assert str(skill_dir / "SKILL.md") not in summary


def test_skill_frontmatter_uses_yaml_parser(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace / "skills",
        "memory",
        description="处理记忆任务时使用。",
        body="body",
        extra_frontmatter=(
            "when_to_use: |\n"
            "  用户询问记忆时。\n"
            "metadata:\n"
            "  akashic:\n"
            "    always: true"
        ),
    )

    loader = SkillsLoader(workspace, builtin_skills_dir=tmp_path / "builtin")
    record = loader.list_skill_records()[0]

    assert record.when_to_use == "用户询问记忆时。\n"
    assert record.always is True


@pytest.mark.asyncio
async def test_load_skill_tool_returns_body_and_base_directory(tmp_path: Path):
    workspace = tmp_path / "workspace"
    skill_dir = _write_skill(
        workspace / "skills",
        "memory",
        description="处理记忆任务时使用。",
        body="读取 guides/intro.md。",
    )
    tool = LoadSkillTool(SkillsLoader(workspace, builtin_skills_dir=tmp_path / "builtin"))

    result = await tool.execute(skill="memory")

    assert "# Skill: memory" in result
    assert f"Base directory: {skill_dir.resolve()}" in result
    assert "读取 guides/intro.md。" in result
    assert "description:" not in result


@pytest.mark.asyncio
async def test_load_skill_tool_blocks_unavailable_skill(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace / "skills",
        "needs-bin",
        body="hidden body",
        extra_frontmatter=(
            'metadata: {"akashic": {"requires": '
            '{"bins": ["definitely-missing-akashic-test-bin"]}}}'
        ),
    )
    tool = LoadSkillTool(SkillsLoader(workspace, builtin_skills_dir=tmp_path / "builtin"))

    result = await tool.execute(skill="needs-bin")

    assert "skill 不可用" in result
    assert "definitely-missing-akashic-test-bin" in result
    assert "hidden body" not in result


def test_always_skill_still_loads_into_context(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace / "skills",
        "memory",
        body="always body",
        extra_frontmatter='metadata: {"akashic": {"always": true}}',
    )
    loader = SkillsLoader(workspace, builtin_skills_dir=tmp_path / "builtin")

    assert loader.get_always_skills() == ["memory"]
    assert "always body" in loader.load_skills_for_context(["memory"])


def test_plugin_roots_do_not_depend_on_workspace_symlinks(tmp_path: Path):
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugin-skills"
    skill_dir = _write_skill(plugin_root, "plugin-skill", body="plugin body")
    workspace_skills = workspace / "skills"
    workspace_skills.mkdir(parents=True)
    (workspace_skills / "legacy-link").symlink_to(skill_dir, target_is_directory=True)
    personal_target = _write_skill(
        tmp_path / "personal-skills",
        "personal-target",
        body="personal body",
    )
    (workspace_skills / "personal-link").symlink_to(
        personal_target,
        target_is_directory=True,
    )

    loader = SkillsLoader(
        workspace,
        builtin_skills_dir=None,
        plugin_roots={"demo": (plugin_root,)},
        ignored_workspace_symlink_roots=(plugin_root,),
    )

    records = loader.build_index().records
    assert set(records) == {"personal-link", "plugin-skill"}
    assert records["personal-link"].source == "workspace"
    assert records["plugin-skill"].source == "plugin"
    assert records["plugin-skill"].source_id == "demo"

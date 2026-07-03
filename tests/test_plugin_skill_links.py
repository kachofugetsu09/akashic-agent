from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.plugins.manager import ActivePluginInfo
from agent.plugins.skill_links import PluginSkillLinker
from agent.skills import SkillsLoader


def _write_plugin_skill(
    plugin_root: Path,
    plugin_id: str,
    skill_name: str,
    *,
    body: str = "plugin skill body",
) -> Path:
    skill_dir = plugin_root / plugin_id / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {skill_name}\n"
        "description: 插件技能\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return plugin_root / plugin_id


def _plugin_info(
    plugin_id: str,
    plugin_dir: Path,
    manifest: dict[str, object] | None = None,
) -> ActivePluginInfo:
    return ActivePluginInfo(
        plugin_id=plugin_id,
        plugin_dir=plugin_dir,
        manifest=manifest or {},
        module_path=f"test_{plugin_id}",
    )


def _memory_engine(name: str) -> object:
    return SimpleNamespace(describe=lambda: SimpleNamespace(name=name))


def test_plugin_skill_linker_creates_workspace_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([_plugin_info("foo", plugin_dir)])

    link = workspace / "skills" / "foo:bar"
    assert result.expected == 1
    assert result.created == 1
    assert link.is_symlink()
    loader = SkillsLoader(workspace, builtin_skills_dir=tmp_path / "builtin")
    assert loader.load_skill_body("foo:bar") == "plugin skill body"


def test_plugin_skill_linker_removes_stale_link(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")
    linker = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    )
    linker.sync([_plugin_info("foo", plugin_dir)])

    result = linker.sync([])

    assert result.removed == 1
    assert not (workspace / "skills" / "foo:bar").exists()


def test_plugin_skill_linker_removes_broken_plugin_link(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    skills_dir = workspace / "skills"
    skills_dir.mkdir(parents=True)
    link = skills_dir / "gone:bar"
    link.symlink_to(plugin_root / "gone" / "skills" / "bar", target_is_directory=True)

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([])

    assert result.removed == 1
    assert not link.is_symlink()


def test_plugin_skill_linker_does_not_overwrite_user_skill(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "foo", "bar")
    user_skill = workspace / "skills" / "foo:bar"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text("user body", encoding="utf-8")

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([_plugin_info("foo", plugin_dir)])

    assert result.skipped == 1
    assert not user_skill.is_symlink()
    assert (user_skill / "SKILL.md").read_text(encoding="utf-8") == "user body"


def test_plugin_skill_linker_filters_by_memory_engine(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugins"
    plugin_dir = _write_plugin_skill(plugin_root, "akasha", "memory")
    manifest: dict[str, object] = {
        "skills": {
            "enabled_when": {
                "kind": "memory_engine",
                "engine": "akasha",
            }
        }
    }
    plugin = _plugin_info("akasha", plugin_dir, manifest)

    disabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("default"),
    ).sync([plugin])
    enabled = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("akasha"),
    ).sync([plugin])
    removed = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=_memory_engine("default"),
    ).sync([plugin])

    assert disabled.expected == 0
    assert enabled.expected == 1
    assert removed.removed == 1
    assert not (workspace / "skills" / "akasha:memory").is_symlink()


def test_meme_plugin_skill_is_exposed_with_plugin_prefix(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = Path(__file__).parents[1] / "plugins"
    plugin_dir = plugin_root / "meme"

    result = PluginSkillLinker(
        workspace=workspace,
        plugin_roots=[plugin_root],
        memory_engine=None,
    ).sync([_plugin_info("meme", plugin_dir)])

    loader = SkillsLoader(workspace, builtin_skills_dir=tmp_path / "builtin")
    assert result.expected >= 1
    assert (workspace / "skills" / "meme:meme-manage").is_symlink()
    assert "表情包库管理" in (loader.load_skill_body("meme:meme-manage") or "")

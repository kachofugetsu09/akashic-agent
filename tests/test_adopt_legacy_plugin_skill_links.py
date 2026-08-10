from pathlib import Path

import pytest

from infra.persistence.json_store import load_json
from scripts.adopt_legacy_plugin_skill_links import adopt_legacy_links


def test_adopts_only_verified_legacy_plugin_links(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugin-home" / "cache"
    target = plugin_root / "github" / "demo" / "skills" / "hello"
    target.mkdir(parents=True)
    skills = workspace / "skills"
    skills.mkdir(parents=True)
    (skills / "hello").symlink_to(target, target_is_directory=True)
    (skills / "user-skill").mkdir()

    links = adopt_legacy_links(workspace=workspace, plugin_roots=(plugin_root,))

    assert links == {"skills/hello": str(target.resolve())}
    assert load_json(
        workspace / "runtime" / "plugin-skill-links.json",
        default=None,
        domain="test",
    ) == {"version": 1, "links": links, "pending": {}}
    assert (skills / "user-skill").is_dir()


def test_rejects_outside_target_without_writing_registry(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugin-home" / "cache"
    plugin_root.mkdir(parents=True)
    outside = tmp_path / "user-skill"
    outside.mkdir()
    skills = workspace / "skills"
    skills.mkdir(parents=True)
    (skills / "user-skill").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="不在 plugin roots 内"):
        adopt_legacy_links(workspace=workspace, plugin_roots=(plugin_root,))

    assert not (workspace / "runtime" / "plugin-skill-links.json").exists()


def test_refuses_to_overwrite_existing_registry(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin_root = tmp_path / "plugin-home" / "cache"
    plugin_root.mkdir(parents=True)
    registry = workspace / "runtime" / "plugin-skill-links.json"
    registry.parent.mkdir(parents=True)
    registry.write_text('{"version":1,"links":{},"pending":{}}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        adopt_legacy_links(workspace=workspace, plugin_roots=(plugin_root,))

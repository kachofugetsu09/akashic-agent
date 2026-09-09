from pathlib import Path

import pytest

from agent.plugins.archive import PluginArchive
from agent.plugins.doctor import _check_expected_links


def test_skill_links_accept_verified_archive_and_detect_changed_content(tmp_path):
    source = tmp_path / "source"
    skill = source / "skills" / "example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("original")
    root = tmp_path / "runtime" / "plugin-archives"
    archive = PluginArchive(root)
    identity = archive.save(source)
    archived = archive.open(identity) / "skills" / "example"
    target = tmp_path / "skills"
    target.mkdir()
    (target / "example").symlink_to(archived)
    assert _check_expected_links(target, {"example": skill}, root) == ([], [])
    (skill / "SKILL.md").write_text("changed")
    assert _check_expected_links(target, {"example": skill}, root) == ([], ["example"])
    (archived / "SKILL.md").chmod(0o644)
    (archived / "SKILL.md").write_text("tampered")
    with pytest.raises(RuntimeError, match="文件树损坏"):
        _check_expected_links(target, {"example": skill}, root)


def test_read_only_archive_does_not_create_directory(tmp_path):
    root = tmp_path / "missing"
    PluginArchive(root, create=False)
    assert not root.exists()

"""单次遍历的源码摘要与分别遍历的完整/运行摘要逐字节一致。"""
import hashlib
import os
from pathlib import Path

import pytest

from agent.plugins import input_preparation
from agent.plugins.input_preparation import SOURCE_EXCLUDED_NAMES, _source_revision, source_revisions


# 改动前的分别遍历实现，作为字节序列的参照。
def _reference(plugin_dir: Path, *, runtime_only: bool) -> str:
    digest = hashlib.sha256()
    root = plugin_dir.resolve(strict=False)
    excluded = SOURCE_EXCLUDED_NAMES if runtime_only else SOURCE_EXCLUDED_NAMES - {".akashic-source.json"}
    for current, directories, filenames in os.walk(plugin_dir, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in excluded)
        current_path = Path(current)
        for name in [*directories, *sorted(filenames)]:
            if name in excluded:
                continue
            path = current_path / name
            relative = path.relative_to(plugin_dir)
            if path.is_symlink():
                resolved = path.resolve(strict=False)
                assert resolved.is_relative_to(root)
                digest.update(str(relative).encode())
                digest.update(os.readlink(path).encode())
                if resolved.is_file():
                    digest.update(resolved.read_bytes())
                continue
            if not path.is_file():
                continue
            digest.update(str(relative).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _tree(root: Path) -> Path:
    plugin = root / "plugin"
    (plugin / "pkg/inner").mkdir(parents=True)
    (plugin / "__pycache__").mkdir()
    (plugin / "node_modules/dep").mkdir(parents=True)
    (plugin / ".akashic-source.json/nested").mkdir(parents=True)
    _ = (plugin / "plugin.py").write_text("VALUE = 1\n")
    _ = (plugin / "pkg/inner/mod.py").write_text("X = 2\n")
    _ = (plugin / "__pycache__/plugin.cpython-313.pyc").write_bytes(b"ignored")
    _ = (plugin / "node_modules/dep/index.js").write_text("ignored")
    _ = (plugin / ".akashic-source.json/nested/label.txt").write_text("source label dir")
    _ = (plugin / "pkg/.akashic-source.json").write_text("{\"tag\": 1}")
    (plugin / "pkg/link.py").symlink_to("inner/mod.py")
    (plugin / "pkg/dir-link").symlink_to("inner")
    return plugin


def test_single_walk_matches_separate_walks(tmp_path: Path):
    plugin = _tree(tmp_path)
    expected = (_reference(plugin, runtime_only=False), _reference(plugin, runtime_only=True))
    assert source_revisions(plugin) == expected
    assert _source_revision(plugin) == expected[0]
    assert _source_revision(plugin, runtime_only=True) == expected[1]
    assert expected[0] != expected[1]


def test_source_label_only_changes_full_revision(tmp_path: Path):
    plugin = _tree(tmp_path)
    before = source_revisions(plugin)
    _ = (plugin / "pkg/.akashic-source.json").write_text("{\"tag\": 2}")
    after = source_revisions(plugin)
    assert after[0] != before[0] and after[1] == before[1]


def test_escaping_symlink_is_rejected(tmp_path: Path):
    plugin = _tree(tmp_path)
    _ = (tmp_path / "outside.py").write_text("SECRET = 1\n")
    (plugin / "escape.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(RuntimeError, match="越界"):
        _ = source_revisions(plugin)


def test_single_revision_walk_hashes_only_requested_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plugin = _tree(tmp_path)
    expected = (_reference(plugin, runtime_only=True), _reference(plugin, runtime_only=False))
    created: list[object] = []
    original = hashlib.sha256

    def counting_sha256() -> object:
        digest = original()
        created.append(digest)
        return digest

    monkeypatch.setattr(input_preparation.hashlib, "sha256", counting_sha256)
    assert (_source_revision(plugin, runtime_only=True), _source_revision(plugin)) == expected
    assert len(created) == 2

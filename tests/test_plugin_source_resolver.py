from __future__ import annotations

from pathlib import Path

import pytest

from agent.plugins.source_resolver import resolve_plugin_sources


def test_installed_resolver_does_not_follow_cache_symlinks(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    outside = tmp_path / "outside" / "feed" / "1.0.0"
    outside.mkdir(parents=True)
    (outside / "plugin.py").write_text("", encoding="utf-8")

    (cache / "lab").mkdir(parents=True)
    (cache / "lab" / "feed").symlink_to(outside.parent, target_is_directory=True)
    (cache / "lab" / "safe" / "1.0.0").mkdir(parents=True)
    (cache / "lab" / "safe" / "1.0.0" / "plugin.py").write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="符号链接"):
        resolve_plugin_sources([], installed_cache_root=cache)


def test_installed_resolver_ignores_transaction_directories(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "lab" / "feed"
    (cache / ".1.0.0-backup-123").mkdir(parents=True)
    (cache / ".1.0.0-backup-123" / "plugin.py").write_text("", encoding="utf-8")
    (cache / ".feed-install-123").mkdir(parents=True)
    (cache / ".feed-install-123" / "plugin.py").write_text("", encoding="utf-8")

    assert resolve_plugin_sources([], installed_cache_root=tmp_path / "cache") == []


def test_installed_resolver_rejects_ambiguous_visible_versions(tmp_path: Path) -> None:
    plugin_root = tmp_path / "cache" / "lab" / "feed"
    for version in ("1.0.0", "2.0.0"):
        (plugin_root / version).mkdir(parents=True)
        (plugin_root / version / "plugin.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="版本冲突"):
        resolve_plugin_sources([], installed_cache_root=tmp_path / "cache")


def test_installed_resolver_rejects_missing_plugin_file(tmp_path: Path) -> None:
    version_root = tmp_path / "cache" / "lab" / "feed" / "1.0.0"
    version_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="缺少 plugin.py"):
        resolve_plugin_sources([], installed_cache_root=tmp_path / "cache")

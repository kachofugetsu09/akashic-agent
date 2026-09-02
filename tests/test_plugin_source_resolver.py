from __future__ import annotations

from pathlib import Path

import pytest

from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.source_resolver import resolve_plugin_sources


def _write_artifact(plugin_base: Path, artifact_id: str) -> Path:
    artifact = plugin_base / ".artifacts" / artifact_id
    artifact.mkdir(parents=True)
    (artifact / "plugin.py").write_text("", encoding="utf-8")
    (artifact / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        'name = "feed"\n'
        'version = "3.0.0"\n'
        "api_version = 3\n"
        'entrypoint = "plugin.py"\n',
        encoding="utf-8",
    )
    return artifact


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


def test_installed_resolver_rejects_legacy_visible_versions(tmp_path: Path) -> None:
    plugin_root = tmp_path / "cache" / "lab" / "feed"
    for version in ("1.0.0", "2.0.0"):
        (plugin_root / version).mkdir(parents=True)
        (plugin_root / version / "plugin.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="不受支持的旧版可见目录"):
        resolve_plugin_sources([], installed_cache_root=tmp_path / "cache")


def test_installed_resolver_rejects_missing_static_manifest(tmp_path: Path) -> None:
    plugin_base = tmp_path / "cache" / "lab" / "feed"
    artifact = plugin_base / ".artifacts" / "1.0.0-aaaa"
    artifact.mkdir(parents=True)
    (artifact / "plugin.py").write_text("", encoding="utf-8")
    (plugin_base / ".pointers.json").write_text(
        '{"stable":".artifacts/1.0.0-aaaa",'
        '"latest":".artifacts/1.0.0-aaaa"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="缺少静态 manifest"):
        resolve_plugin_sources([], installed_cache_root=tmp_path / "cache")


def test_installed_resolver_selects_stable_or_latest_artifact(tmp_path: Path) -> None:
    plugin_base = tmp_path / "cache" / "lab" / "feed"
    stable = _write_artifact(plugin_base, "1.0.0-aaaa")
    latest = _write_artifact(plugin_base, "2.0.0-bbbb")
    _ = write_pointers(
        plugin_base,
        stable=ArtifactPointer(".artifacts/1.0.0-aaaa"),
        latest=ArtifactPointer(".artifacts/2.0.0-bbbb"),
    )

    stable_sources = resolve_plugin_sources(
        [],
        installed_cache_root=tmp_path / "cache",
    )
    latest_sources = resolve_plugin_sources(
        [],
        installed_cache_root=tmp_path / "cache",
        installed_selector="latest",
    )

    assert [(item.plugin_name, item.plugin_root) for item in stable_sources] == [
        ("feed", stable)
    ]
    assert [(item.plugin_name, item.plugin_root) for item in latest_sources] == [
        ("feed", latest)
    ]


def test_installed_resolver_allows_candidate_before_first_promotion(
    tmp_path: Path,
) -> None:
    plugin_base = tmp_path / "cache" / "lab" / "feed"
    latest = _write_artifact(plugin_base, "1.0.0-aaaa")
    _ = write_pointers(
        plugin_base,
        stable=ArtifactPointer(None),
        latest=ArtifactPointer(".artifacts/1.0.0-aaaa"),
    )

    assert (
        resolve_plugin_sources(
            [],
            installed_cache_root=tmp_path / "cache",
        )
        == []
    )
    assert (
        resolve_plugin_sources(
            [],
            installed_cache_root=tmp_path / "cache",
            installed_selector="latest",
        )[0].plugin_root
        == latest
    )


def test_installed_resolver_rejects_invalid_or_escaping_pointer_state(
    tmp_path: Path,
) -> None:
    plugin_base = tmp_path / "cache" / "lab" / "feed"
    _ = _write_artifact(plugin_base, "1.0.0-aaaa")
    (plugin_base / ".pointers.json").write_text(
        '{"stable":".artifacts/1.0.0-aaaa"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="结构无效"):
        resolve_plugin_sources([], installed_cache_root=tmp_path / "cache")

    (plugin_base / ".pointers.json").write_text(
        '{"stable":".artifacts/1.0.0-aaaa","latest":"../outside"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pointer 越界"):
        resolve_plugin_sources(
            [],
            installed_cache_root=tmp_path / "cache",
            installed_selector="latest",
        )


def test_installed_resolver_rejects_artifact_symlink(tmp_path: Path) -> None:
    plugin_base = tmp_path / "cache" / "lab" / "feed"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "plugin.py").write_text("", encoding="utf-8")
    artifacts = plugin_base / ".artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "escape").symlink_to(outside, target_is_directory=True)
    (plugin_base / ".pointers.json").write_text(
        '{"stable":".artifacts/escape","latest":".artifacts/escape"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不能经过符号链接"):
        resolve_plugin_sources([], installed_cache_root=tmp_path / "cache")

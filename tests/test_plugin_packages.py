from pathlib import Path
from typing import Any

import pytest

from agent.plugins.manifest import (
    load_package_manifest,
    write_package_manifest,
    write_plugin_manifest,
)
from agent.plugins.packages import discover_plugin_packages, enabled_plugin_packages
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus


def test_repository_declares_two_proactive_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    packages = discover_plugin_packages(root)

    assert set(packages) == {"default-proactive", "wake-proactive"}
    assert packages["default-proactive"].members == (
        "default_proactive",
        "proactive_flow",
        "drift_flow",
    )
    assert packages["wake-proactive"].members == (
        "wake_proactive",
        "wake_proactive_flow",
        "wake_drift_flow",
    )


def test_manager_discover_reads_each_package_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    manager = PluginManager(
        [root / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )
    original_read_text = Path.read_text
    reads: list[Path] = []

    def record_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.name == "package.toml":
            reads.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read)

    mods = manager.discover()

    assert reads == [
        root / "plugin_packages" / "default-proactive" / "package.toml",
        root / "plugin_packages" / "wake-proactive" / "package.toml",
    ]
    assert [(mod["name"], mod["package_id"], mod["source_type"]) for mod in mods] == [
        ("akasha", "", "builtin"),
        ("default_memory", "", "builtin"),
    ]


def test_proactive_runtime_is_exclusive(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="proactive.runtime"):
        enabled_plugin_packages(
            root,
            {"default-proactive": True, "wake-proactive": True},
        )


def test_plugin_manifest_write_preserves_packages(tmp_path: Path) -> None:
    (tmp_path / "manifest.toml").write_text(
        '[plugins]\n\n[packages."wake-proactive"]\nenabled = true\n',
        encoding="utf-8",
    )

    write_plugin_manifest({"feed@lab": True}, plugins_home=tmp_path)

    assert load_package_manifest(tmp_path) == {"wake-proactive": True}

    write_package_manifest({"wake-proactive": False}, plugins_home=tmp_path)

    assert load_package_manifest(tmp_path) == {"wake-proactive": False}


def test_sync_manifest_migrates_legacy_proactive_members(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "manifest.toml").write_text(
        '[plugins]\n\n[plugins."wake_proactive"]\nenabled = true\n'
        '[plugins."default_proactive"]\nenabled = false\n',
        encoding="utf-8",
    )
    manager = PluginManager(
        [root / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    manager.sync_manifest(plugins_home=tmp_path)

    assert load_package_manifest(tmp_path) == {
        "default-proactive": False,
        "wake-proactive": True,
    }
    manifest = (tmp_path / "manifest.toml").read_text(encoding="utf-8")
    assert 'plugins."wake_proactive"' not in manifest
    assert 'plugins."default_proactive"' not in manifest


def test_package_manifest_rejects_non_schema_values(tmp_path: Path) -> None:
    package_dir = tmp_path / "plugin_packages" / "broken"
    package_dir.mkdir(parents=True)

    (package_dir / "package.toml").write_text(
        '[package]\nid = "broken"\nmembers = ["broken"]\n'
        'dashboard = "false"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dashboard 无效"):
        discover_plugin_packages(tmp_path)

    (package_dir / "package.toml").write_text(
        '[package]\nid = "broken"\nmembers = ["broken"]\n'
        'provides = "proactive.runtime"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provides 无效"):
        discover_plugin_packages(tmp_path)

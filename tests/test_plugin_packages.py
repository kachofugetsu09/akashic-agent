from pathlib import Path
from typing import Any

import pytest

from agent.plugins.manifest import (
    load_package_manifest,
    write_package_manifest,
    write_plugin_manifest,
)
from agent.plugins.packages import discover_plugin_packages
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus


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

    assert reads == []
    assert {mod["name"] for mod in mods} == {
        "akasha",
        "compaction",
        "eventmail",
        "models",
        "markdown_memory",
        "openai-compatible",
        "opencode-go",
        "codex",
        "conversation-ui",
        "drift",
        "runtime-ui",
        "scheduler",
        "shell-ui",
        "subagent",
        "wake",
        "workbench-ui",
    }


def test_manager_can_disable_one_builtin_without_hiding_installed_plugins(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    manager = PluginManager(
        [root / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
        disabled_builtin_plugins=frozenset({"subagent"}),
    )

    assert "subagent" not in {item["name"] for item in manager.discover()}


def test_plugin_manifest_write_preserves_packages(tmp_path: Path) -> None:
    (tmp_path / "manifest.toml").write_text(
        '[plugins]\n\n[packages."example-bundle"]\nenabled = true\n',
        encoding="utf-8",
    )

    write_plugin_manifest({"feed@lab": True}, plugins_home=tmp_path)

    assert load_package_manifest(tmp_path) == {"example-bundle": True}

    write_package_manifest({"example-bundle": False}, plugins_home=tmp_path)

    assert load_package_manifest(tmp_path) == {"example-bundle": False}


def test_package_manifest_rejects_non_schema_values(tmp_path: Path) -> None:
    package_dir = tmp_path / "plugin_packages" / "broken"
    package_dir.mkdir(parents=True)

    (package_dir / "package.toml").write_text(
        '[package]\nid = "broken"\nmembers = ["broken"]\n' 'dashboard = "false"\n',
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

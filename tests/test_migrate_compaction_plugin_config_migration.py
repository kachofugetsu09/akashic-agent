from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest
import yoyo

from agent.migrations.context import bind_migration_context

_ROOT = Path(__file__).parents[1]
_MIGRATION = _ROOT / "migrations/yoyo/20260831_01_migrate_compaction_plugin_config.py"


def _load():
    spec = importlib.util.spec_from_file_location("compaction_plugin_config_migration", _MIGRATION)
    assert spec is not None and spec.loader is not None
    original = yoyo.step
    yoyo.step = lambda callback: callback  # type: ignore[assignment]
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        yoyo.step = original


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.migrate_compaction_plugin_config(None)


def test_moves_exact_policy_with_verified_before_images(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    before = (
        '[agent]\nsystem_prompt = "keep"\n\n'
        "[agent.context.compaction]\nkeep_recent_tokens = 21000\n"
    ).encode()
    config.write_bytes(before)

    _run(_load(), config, workspace)

    assert "compaction" not in tomllib.loads(config.read_text())["agent"].get("context", {})
    plugin = workspace / "plugin-data/compaction-builtin/config.local.toml"
    assert tomllib.loads(plugin.read_text()) == {"keep_recent_tokens": 21000}
    backups = list((workspace / "backups/migrate-compaction-plugin-config").glob("*/manifest.json"))
    assert len(backups) == 1
    manifest = json.loads(backups[0].read_text())
    assert manifest["sources"][0]["sha256"]
    assert (backups[0].parent / "source-0.before").read_bytes() == before

    _run(_load(), config, workspace)
    assert len(list((workspace / "backups/migrate-compaction-plugin-config").glob("*/manifest.json"))) == 1


def test_conflicting_plugin_policy_fails_before_any_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plugin = workspace / "plugin-data/compaction-builtin/config.local.toml"
    plugin.parent.mkdir(parents=True)
    plugin.write_text("keep_recent_tokens = 22000\n")
    config = tmp_path / "config.toml"
    config.write_text("[agent.context.compaction]\nkeep_recent_tokens = 21000\n")
    before_config = config.read_bytes()
    before_plugin = plugin.read_bytes()

    with pytest.raises(RuntimeError, match="冲突"):
        _run(_load(), config, workspace)

    assert config.read_bytes() == before_config
    assert plugin.read_bytes() == before_plugin
    assert not (workspace / "backups").exists()


def test_rerun_forward_completes_prepared_intent_after_first_publish(
    tmp_path: Path,
) -> None:
    module = _load()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        "[agent.context.compaction]\nkeep_recent_tokens = 21000\n"
    )
    plugin = workspace / "plugin-data/compaction-builtin/config.local.toml"
    config_snapshot = module._snapshot(config)
    plugin_snapshot = module._snapshot(plugin)
    rendered = module._render(config_snapshot.content, plugin_snapshot.content)
    assert rendered is not None
    publication = workspace / "backups/migrate-compaction-plugin-config/interrupted"
    module._backup((config_snapshot, plugin_snapshot), rendered, publication)
    module._write(config_snapshot.target, rendered[0], config_snapshot.mode)

    _run(module, config, workspace)

    assert (publication / "complete").read_text() == "complete\n"
    migrated_config = tomllib.loads(config.read_text())
    assert "agent" not in migrated_config or "compaction" not in migrated_config[
        "agent"
    ].get("context", {})
    assert tomllib.loads(plugin.read_text()) == {"keep_recent_tokens": 21000}


def test_rejects_plugin_data_symlink_before_read_or_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "plugin-data").symlink_to(outside, target_is_directory=True)
    config = tmp_path / "config.toml"
    config.write_text(
        "[agent.context.compaction]\nkeep_recent_tokens = 21000\n"
    )
    before = config.read_bytes()

    with pytest.raises(ValueError, match="符号链接"):
        _run(_load(), config, workspace)

    assert config.read_bytes() == before
    assert not list(outside.rglob("*"))


def test_publication_conflict_never_overwrites_newer_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        "[agent.context.compaction]\nkeep_recent_tokens = 21000\n"
    )
    plugin = workspace / "plugin-data/compaction-builtin/config.local.toml"
    original_complete = module._complete_intent

    def drift_then_complete(root, snapshots):
        plugin.parent.mkdir(parents=True, exist_ok=True)
        plugin.write_text("keep_recent_tokens = 33000\n")
        original_complete(root, snapshots)

    monkeypatch.setattr(module, "_complete_intent", drift_then_complete)

    with pytest.raises(RuntimeError, match="目标已变化"):
        _run(module, config, workspace)

    assert plugin.read_text() == "keep_recent_tokens = 33000\n"
    assert config.read_text().endswith("keep_recent_tokens = 21000\n")
    conflicts = list(
        (workspace / "backups/migrate-compaction-plugin-config").glob("*/conflict")
    )
    assert len(conflicts) == 1


def test_publication_preserves_same_content_symlink_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        "[agent.context.compaction]\nkeep_recent_tokens = 21000\n"
    )
    plugin = workspace / "plugin-data/compaction-builtin/config.local.toml"
    outside = tmp_path / "outside.toml"
    outside.write_text("keep_recent_tokens = 21000\n")
    original_complete = module._complete_intent

    def drift_then_complete(root, snapshots):
        plugin.parent.mkdir(parents=True, exist_ok=True)
        plugin.symlink_to(outside)
        original_complete(root, snapshots)

    monkeypatch.setattr(module, "_complete_intent", drift_then_complete)

    with pytest.raises(RuntimeError, match="path 冲突"):
        _run(module, config, workspace)

    assert plugin.is_symlink()
    assert plugin.resolve() == outside
    assert outside.read_text() == "keep_recent_tokens = 21000\n"

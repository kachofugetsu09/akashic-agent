from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

import pytest
import tomlkit

from agent.config import Config
from agent.migrations import migrate_installation
from bootstrap import app as bootstrap_app
from bootstrap.init_workspace import init_workspace
from bootstrap.tools import build_core_runtime
from core.net.http import SharedHttpResources


class _FakeServer:
    def __init__(self) -> None:
        self.should_exit = False

    async def serve(self) -> None:
        while not self.should_exit:
            await asyncio.sleep(0)


def _prepare_fresh_case(root: Path) -> tuple[Path, Path, Path]:
    """Create one fresh workspace using the ordinary Akasha plugin default."""

    home = root / "home"
    config_path = root / "config.toml"
    workspace = root / "workspace"
    home.mkdir()
    _ = init_workspace(config_path=config_path, workspace=workspace)
    return home, config_path, workspace


@pytest.mark.asyncio
async def test_fresh_init_core_configuration_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    _home, config_path, workspace = _prepare_fresh_case(tmp_path)
    config = Config.load(config_path, workspace=workspace)
    resources = SharedHttpResources()
    runtime = build_core_runtime(config, workspace, resources)

    try:
        await runtime.start()
        assert runtime.plugin_manager.current_snapshot is not None
        active = {item.plugin_id for item in runtime.plugin_manager.active_plugins()}
        assert "akasha" in active
        assert "default_memory" not in active
    finally:
        await runtime.stop()
        await resources.aclose()


@pytest.mark.asyncio
async def test_fresh_init_runtime_start_stop_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    _home, config_path, workspace = _prepare_fresh_case(tmp_path)
    monkeypatch.setattr(
        bootstrap_app,
        "build_dashboard_server",
        lambda **_kwargs: _FakeServer(),
    )
    monkeypatch.setattr(
        bootstrap_app,
        "build_chat_server",
        lambda **_kwargs: _FakeServer(),
    )
    config = Config.load(config_path, workspace=workspace)
    runtime = bootstrap_app.AppRuntime(config, workspace)

    try:
        await runtime.start()
        assert runtime.core is not None
        assert runtime.core.plugin_manager.current_snapshot is not None
        active = {
            item.plugin_id for item in runtime.core.plugin_manager.active_plugins()
        }
        assert "akasha" in active
        assert "default_memory" not in active
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_legacy_akasha_config_migrates_before_plugin_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the production startup boundary from legacy selection to Akasha."""

    # 1. Recreate the exact legacy Akasha selector from a valid installation.
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    _home, config_path, workspace = _prepare_fresh_case(tmp_path)
    document = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    document["memory"]["engine"] = "akasha"
    config_path.write_text(tomlkit.dumps(document), encoding="utf-8")

    # 2. Apply startup Yoyo and load the ordinary plugin configuration.
    outcome = migrate_installation(config_path, workspace)
    assert "20260825_02_select_akasha_embedding_plugin" in outcome.migrations
    migrated = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "engine" not in migrated["memory"]
    assert migrated["agent"]["plugins"]["disabled_builtin"] == []

    # 3. Prove the migrated runtime starts with only Akasha claiming memory.
    config = Config.load(config_path, workspace=workspace)
    resources = SharedHttpResources()
    runtime = build_core_runtime(config, workspace, resources)
    try:
        await runtime.start()
        active = {item.plugin_id for item in runtime.plugin_manager.active_plugins()}
        assert "akasha" in active
        assert "default_memory" not in active
    finally:
        await runtime.stop()
        await resources.aclose()


@pytest.mark.asyncio
async def test_disabled_legacy_memory_does_not_start_or_replay_akasha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an opted-out legacy workspace outside Akasha activation."""

    # 1. Recreate an explicitly disabled legacy Default Memory selection.
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    _home, config_path, workspace = _prepare_fresh_case(tmp_path)
    document = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    document["memory"]["enabled"] = False
    document["memory"]["engine"] = "default"
    config_path.write_text(tomlkit.dumps(document), encoding="utf-8")

    # 2. Migrate the selector before runtime plugin discovery.
    _ = migrate_installation(config_path, workspace)
    migrated = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["memory"]["enabled"] is False
    assert "akasha" in migrated["agent"]["plugins"]["disabled_builtin"]

    # 3. Boot normally and prove no Akasha owner or derived database appeared.
    config = Config.load(config_path, workspace=workspace)
    resources = SharedHttpResources()
    runtime = build_core_runtime(config, workspace, resources)
    try:
        await runtime.start()
        active = {item.plugin_id for item in runtime.plugin_manager.active_plugins()}
        assert "akasha" not in active
        assert not (workspace / "memory" / "akasha.db").exists()
        assert not (workspace / "memory" / "akasha-v2-index.db").exists()
    finally:
        await runtime.stop()
        await resources.aclose()

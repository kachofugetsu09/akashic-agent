from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.config import Config
from bootstrap import app as bootstrap_app
from bootstrap.init_workspace import init_workspace


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

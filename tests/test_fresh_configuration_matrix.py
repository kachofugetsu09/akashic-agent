from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.config import Config
from bootstrap import app as bootstrap_app
from bootstrap.init_workspace import init_workspace
from bootstrap.tools import build_core_runtime
from core.net.http import SharedHttpResources

MEMORY_CASES = (
    ("disabled", False, ""),
    ("default", True, ""),
    ("akasha", True, "akasha"),
)


class _FakeServer:
    def __init__(self) -> None:
        self.should_exit = False

    async def serve(self) -> None:
        while not self.should_exit:
            await asyncio.sleep(0)


def _set_toml_value(text: str, section: str, key: str, value: str) -> str:
    """Update one TOML field inside an existing section."""

    lines = text.splitlines()
    start = lines.index(f"[{section}]") + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("[")),
        len(lines),
    )
    prefix = f"{key} ="
    for index in range(start, end):
        if lines[index].startswith(prefix):
            lines[index] = f"{key} = {value}"
            break
    else:
        lines.insert(end, f"{key} = {value}")
    return "\n".join(lines) + "\n"


def _prepare_fresh_case(
    root: Path,
    *,
    memory_enabled: bool,
    memory_engine: str,
) -> tuple[Path, Path, Path]:
    """Create a fresh isolated workspace for one memory configuration."""

    home = root / "home"
    config_path = root / "config.toml"
    workspace = root / "workspace"
    home.mkdir()
    _ = init_workspace(config_path=config_path, workspace=workspace)
    text = config_path.read_text(encoding="utf-8")
    text = _set_toml_value(text, "memory", "enabled", str(memory_enabled).lower())
    text = _set_toml_value(text, "memory", "engine", f'"{memory_engine}"')
    config_path.write_text(text, encoding="utf-8")
    return home, config_path, workspace


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_name", "memory_enabled", "memory_engine"), MEMORY_CASES
)
async def test_fresh_init_core_configuration_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_name: str,
    memory_enabled: bool,
    memory_engine: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    _home, config_path, workspace = _prepare_fresh_case(
        tmp_path,
        memory_enabled=memory_enabled,
        memory_engine=memory_engine,
    )
    config = Config.load(config_path, workspace=workspace)
    resources = SharedHttpResources()
    runtime = build_core_runtime(config, workspace, resources)

    try:
        await runtime.start()
        assert runtime.memory_runtime.engine.describe().name == memory_name
        assert runtime.plugin_manager.current_snapshot is not None
    finally:
        await runtime.stop()
        await runtime.memory_runtime.aclose()
        await resources.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_name", "memory_enabled", "memory_engine"), MEMORY_CASES[1:]
)
async def test_fresh_init_runtime_start_stop_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_name: str,
    memory_enabled: bool,
    memory_engine: str,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    _home, config_path, workspace = _prepare_fresh_case(
        tmp_path,
        memory_enabled=memory_enabled,
        memory_engine=memory_engine,
    )
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
        assert runtime.memory_runtime.engine.describe().name == memory_name
        assert runtime.core is not None
        assert runtime.core.plugin_manager.current_snapshot is not None
    finally:
        await runtime.shutdown()

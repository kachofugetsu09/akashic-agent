from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.registry import plugin_registry
from agent.tool_hooks import ToolExecutionRequest, ToolExecutor
from bus.event_bus import EventBus

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins" / "shell_safety"


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


async def _invoke(tool_name: str, arguments: dict[str, Any]) -> Any:
    return {"tool": tool_name, "arguments": dict(arguments)}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _make_plugin_root(tmp_path: Path) -> Path:
    root = tmp_path / "plugins"
    root.mkdir()
    (root / "shell_safety").symlink_to(PLUGIN_DIR, target_is_directory=True)
    return root


def _run_shell(root: Path, command: str) -> Any:
    bus = EventBus()
    mgr = PluginManager(plugin_dirs=[root], event_bus=bus)
    _run(mgr.load_all())
    return _run(
        ToolExecutor(mgr.tool_hooks).execute(
            ToolExecutionRequest(
                call_id="c1",
                tool_name="shell",
                arguments={"command": command, "description": "测试命令"},
                source="passive",
            ),
            _invoke,
        )
    )


def test_shell_safety_blocks_sudo_without_non_interactive(tmp_path: Path) -> None:
    result = _run_shell(_make_plugin_root(tmp_path), "sudo pacman -Syu --noconfirm")

    assert result.status == "denied"
    assert "sudo -n" in result.output


def test_shell_safety_blocks_interactive_editor(tmp_path: Path) -> None:
    result = _run_shell(_make_plugin_root(tmp_path), "sudo -n vim /etc/example.service")

    assert result.status == "denied"
    assert "vim" in result.output


def test_shell_safety_blocks_package_write_without_noconfirm(tmp_path: Path) -> None:
    result = _run_shell(_make_plugin_root(tmp_path), "pacman -Syu")

    assert result.status == "denied"
    assert "--noconfirm" in result.output


def test_shell_safety_allows_non_interactive_package_write(tmp_path: Path) -> None:
    result = _run_shell(_make_plugin_root(tmp_path), "sudo -n pacman -Syu --noconfirm")

    assert result.status == "success"
    assert result.final_arguments["command"] == "sudo -n pacman -Syu --noconfirm"


def test_shell_safety_allows_package_query(tmp_path: Path) -> None:
    result = _run_shell(_make_plugin_root(tmp_path), "pacman -Q")

    assert result.status == "success"
    assert result.final_arguments["command"] == "pacman -Q"

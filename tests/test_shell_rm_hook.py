from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from agent.tool_hooks import ToolExecutionRequest, ToolExecutor
from bus.event_bus import EventBus
from agent.plugins.manager import PluginManager


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "plugins"


def _make_manager(plugin_dirs: list[Path], *, event_bus: EventBus, tools: Any = None) -> PluginManager:
    return PluginManager(plugin_dirs=plugin_dirs, event_bus=event_bus, tool_registry=tools)


async def _invoke(tool_name: str, arguments: dict[str, Any]) -> Any:
    return {"tool": tool_name, "arguments": dict(arguments)}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_shell_rm_hook_rewrites_rm_and_creates_restore_dir(tmp_path: Path) -> None:
    restore_dir = tmp_path / "restore"
    os.environ["AKASIC_RESTORE_DIR"] = str(restore_dir)
    try:
        bus = EventBus()
        mgr = _make_manager([FIXTURES_DIR], event_bus=bus)
        _run(mgr.load_all())
        executor = ToolExecutor(mgr.tool_hooks)

        result = _run(
            executor.execute(
                ToolExecutionRequest(
                    call_id="c1",
                    tool_name="shell",
                    arguments={
                        "command": "rm -rf foo bar",
                        "description": "删除文件",
                    },
                    source="passive",
                ),
                _invoke,
            )
        )

        assert result.status == "success"
        assert restore_dir.is_dir()
        assert result.final_arguments["command"] == f"mv -- foo bar {restore_dir}"
        assert result.output["arguments"]["command"] == f"mv -- foo bar {restore_dir}"
    finally:
        os.environ.pop("AKASIC_RESTORE_DIR", None)


def test_shell_rm_hook_rewrites_sudo_rm(tmp_path: Path) -> None:
    restore_dir = tmp_path / "restore"
    os.environ["AKASIC_RESTORE_DIR"] = str(restore_dir)
    try:
        bus = EventBus()
        mgr = _make_manager([FIXTURES_DIR], event_bus=bus)
        _run(mgr.load_all())

        result = _run(
            ToolExecutor(mgr.tool_hooks).execute(
                ToolExecutionRequest(
                    call_id="c1",
                    tool_name="shell",
                    arguments={
                        "command": "sudo rm -f /tmp/a",
                        "description": "删除文件",
                    },
                    source="passive",
                ),
                _invoke,
            )
        )

        assert result.status == "success"
        assert result.final_arguments["command"] == f"sudo mv -- /tmp/a {restore_dir}"
    finally:
        os.environ.pop("AKASIC_RESTORE_DIR", None)


def test_shell_rm_hook_skips_non_rm_command(tmp_path: Path) -> None:
    restore_dir = tmp_path / "restore"
    os.environ["AKASIC_RESTORE_DIR"] = str(restore_dir)
    try:
        bus = EventBus()
        mgr = _make_manager([FIXTURES_DIR], event_bus=bus)
        _run(mgr.load_all())

        result = _run(
            ToolExecutor(mgr.tool_hooks).execute(
                ToolExecutionRequest(
                    call_id="c1",
                    tool_name="shell",
                    arguments={
                        "command": "ls -la",
                        "description": "列目录",
                    },
                    source="passive",
                ),
                _invoke,
            )
        )

        assert result.status == "success"
        assert not restore_dir.exists()
        assert result.final_arguments["command"] == "ls -la"
    finally:
        os.environ.pop("AKASIC_RESTORE_DIR", None)

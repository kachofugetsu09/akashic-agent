from __future__ import annotations

import importlib.util
from pathlib import Path

from agent.plugins.manager import PluginManager


def test_workspace_mcp_legacy_modules_are_physically_unreachable() -> None:
    for module_name in (
        "agent.mcp.declarations",
        "agent.mcp.watcher",
        "agent.mcp.admin",
        "agent.mcp.generation",
        "agent.tools.workspace_mcp",
    ):
        assert importlib.util.find_spec(module_name) is None


def test_workspace_mcp_manager_owner_and_builtin_skill_are_removed() -> None:
    for name in (
        "active_workspace_mcp",
        "prepared_workspace_mcp",
        "prepare_workspace_mcp",
        "publish_workspace_mcp",
        "discard_workspace_mcp_candidate",
    ):
        assert not hasattr(PluginManager, name)
    assert not (
        Path(__file__).parents[1] / "skills/manage-workspace-mcp" / "SKILL.md"
    ).exists()


def test_workspace_init_does_not_create_removed_mcp_directories() -> None:
    source = (Path(__file__).parents[1] / "bootstrap/init_workspace.py").read_text(
        encoding="utf-8"
    )

    assert '"mcp"' not in source
    assert '"mcp/servers"' not in source

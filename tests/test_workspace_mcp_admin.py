from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent.mcp.admin import WorkspaceMcpAdmin
from agent.mcp.watcher import WorkspaceMcpWatcher
from agent.plugins.manager import PluginManager
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus


def _server(root: Path, tool_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "server.py"
    path.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        " message=json.loads(line)\n"
        " if 'id' not in message: continue\n"
        " method=message.get('method')\n"
        " if method == 'initialize': result={'protocolVersion':'2025-11-25'}\n"
        f" elif method == 'tools/list': result={{'tools':[{{'name':'{tool_name}','description':'probe','inputSchema':{{'type':'object','properties':{{}}}}}}]}}\n"
        " else: result={'content':[{'type':'text','text':'ok'}]}\n"
        " print(json.dumps({'jsonrpc':'2.0','id':message['id'],'result':result}),flush=True)\n",
        encoding="utf-8",
    )
    return path


def _manager(workspace: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[workspace / "plugins"],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(),
        workspace=workspace,
        installed_cache_root=workspace / "cache",
    )


@pytest.mark.asyncio
async def test_admin_apply_status_failed_update_rollback_and_remove(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    manager = _manager(workspace)
    watcher = WorkspaceMcpWatcher(
        manager,
        workspace / "mcp" / "servers",
        mcp_root=workspace / "mcp",
    )
    admin = WorkspaceMcpAdmin(workspace, watcher)
    await watcher.reconcile()
    server = _server(workspace / "mcp" / "probe", "inspect")

    applied = await admin.apply(
        name="desktop",
        command=[sys.executable, str(server)],
        cwd=None,
        env={"SECRET_TOKEN": "never-return-this"},
        watch_paths=["../probe/server.py"],
    )

    declaration = workspace / "mcp" / "servers" / "desktop.toml"
    original = declaration.read_text(encoding="utf-8")
    assert applied["status"] == "active"
    assert applied["effectiveFrom"] == "next_turn"
    assert applied["runtime"]["servers"] == ["desktop"]
    assert applied["runtime"]["tools"] == ["mcp_desktop__inspect"]
    status = admin.status("desktop")
    assert status["declarations"][0]["envKeys"] == ["SECRET_TOKEN"]
    assert "never-return-this" not in json.dumps(status)

    active = manager.active_workspace_mcp
    with pytest.raises(RuntimeError, match="声明已回滚"):
        await admin.apply(
            name="desktop",
            command=[sys.executable, str(workspace / "mcp" / "missing.py")],
            cwd=None,
            env={},
            watch_paths=[],
        )
    assert declaration.read_text(encoding="utf-8") == original
    assert manager.active_workspace_mcp is active
    assert list((workspace / "mcp" / "backups" / "desktop").glob("*.toml"))

    removed = await admin.remove("desktop")
    assert removed["status"] == "removed"
    assert removed["runtime"]["servers"] == []
    assert not declaration.exists()
    assert admin.status("desktop")["declarations"] == []
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_admin_rejects_invalid_name_and_outside_watch_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    manager = _manager(workspace)
    watcher = WorkspaceMcpWatcher(
        manager,
        workspace / "mcp" / "servers",
        mcp_root=workspace / "mcp",
    )
    admin = WorkspaceMcpAdmin(workspace, watcher)
    await watcher.reconcile()
    server = _server(workspace / "mcp" / "probe", "inspect")

    with pytest.raises(ValueError, match="MCP name"):
        await admin.apply(
            name="../escape",
            command=[sys.executable, str(server)],
            cwd=None,
            env={},
            watch_paths=[],
        )
    with pytest.raises(RuntimeError, match="声明已回滚"):
        await admin.apply(
            name="escape",
            command=[sys.executable, str(server)],
            cwd=None,
            env={},
            watch_paths=["../../../outside"],
        )
    assert not (workspace / "mcp" / "servers" / "escape.toml").exists()
    await manager.terminate_all()

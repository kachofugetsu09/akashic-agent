"""stable_view 插件：/stable 命令渲染当前组合的 ASCII 依赖树。"""
import ast
from pathlib import Path
import shutil
import sys
from typing import Any, cast

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.commands import COMMANDS
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.commands import plugin as commands_plugin
from plugins.stable_view import plugin as stable_view
from plugins.stable_view.plugin import format_stable_catalog


def _catalog() -> dict[str, object]:
    return {
        "snapshot_id": "0123456789abcdef",
        "plugins": [
            {
                "id": "commands",
                "revision": "revcommands01",
                "generation_id": "gen-commands",
                "api_version": 3,
                "composition": {
                    "ready": True,
                    "fibers": [
                        {"name": "commands", "parent": None, "state": "active",
                         "required": True, "dependencies": [],
                         "missing_services": [], "error": None},
                    ],
                    "incident_count": 0,
                },
            },
            {
                "id": "stable_view",
                "revision": "revstable0001",
                "generation_id": "gen-stable",
                "api_version": 3,
                "composition": {
                    "ready": True,
                    "fibers": [
                        {"name": "stable_view", "parent": None, "state": "active",
                         "required": True,
                         "dependencies": ["core.commands"],
                         "missing_services": [], "error": None},
                        {"name": "stable_view.child", "parent": "stable_view",
                         "state": "active", "required": False,
                         "dependencies": [], "missing_services": [], "error": None},
                    ],
                    "incident_count": 0,
                },
            },
        ],
        "mcp_servers": [{"name": "feed"}],
    }


def test_format_stable_catalog_renders_tree() -> None:
    text = format_stable_catalog(_catalog())
    assert "current runtime graph 0123456789abcdef" in text
    assert "stable_view" in text
    assert "└─ stable_view.child" in text
    assert "core.commands" in text
    assert "feed" in text


def test_format_stable_catalog_mcp_unavailable() -> None:
    catalog = _catalog()
    catalog.pop("mcp_servers")
    catalog["mcp_unavailable"] = {
        "code": "mcp_catalog_unavailable", "message": "no session",
    }
    text = format_stable_catalog(catalog)
    assert "mcp_catalog_unavailable" in text
    assert "stable_view" in text


def test_format_stable_catalog_shows_pre_fiber_failure_and_cleanup_pending() -> None:
    catalog = _catalog()
    catalog["plugins"].append({
        "id": "broken",
        "revision": "revbroken",
        "generation_id": "gen-broken",
        "api_version": 3,
        "load_error": "import blocked",
        "cleanup_pending": True,
        "composition": {"ready": False, "fibers": [], "incident_count": 0},
    })
    text = format_stable_catalog(catalog)
    assert "broken" in text
    assert "load error: import blocked" in text
    assert "cleanup pending" in text


async def _mount_stable_view(tmp_path, reader) -> CompositionRoot:
    root = CompositionRoot("selected")
    await root.context.provide(RUNTIME_CATALOG, reader)
    await root.mount(commands_plugin.apply, name="commands")
    await root.mount(
        stable_view.apply,
        name="stable_view",
        inject=stable_view.inject,
        runtime=PluginRuntime(
            plugin_id="stable_view", generation_id="gen-stable",
            plugin_dir=tmp_path, data_dir=tmp_path / "data",
            workspace=tmp_path, config={},
        ),
    )
    return root


@pytest.mark.asyncio
async def test_stable_command_returns_tree(tmp_path) -> None:
    root = await _mount_stable_view(tmp_path, lambda _ctx: _catalog())
    try:
        commands = root.context.require(COMMANDS).freeze()
        result = await commands.execute(
            "/stable", session_key="chat", channel="test",
            chat_id="room", sender="user",
        )
        assert result is not None
        assert result.result.kind == "success"
        assert "stable_view.child" in result.result.text
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_stable_command_reports_reader_failure(tmp_path) -> None:
    def broken(_ctx) -> dict[str, object]:
        raise RuntimeError("no scope")

    root = await _mount_stable_view(tmp_path, broken)
    try:
        commands = root.context.require(COMMANDS).freeze()
        result = await commands.execute(
            "/stable", session_key="chat", channel="test",
            chat_id="room", sender="user",
        )
        assert result is not None
        assert result.result.kind == "error"
        assert "no scope" in result.result.text
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_stable_command_uses_real_manager_commands_and_live_catalog(tmp_path) -> None:
    """Manager 装配真实 Commands 与 stable_view，并由 live catalog 显示 MCP 缺失。"""
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus
    from tests.fixtures.plugin_workspace import initialize_plugin_workspace

    source = tmp_path / "plugins"
    shutil.copytree(
        Path(__file__).parents[1] / "plugins/commands",
        source / "commands",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        Path(__file__).parents[1] / "plugins/stable_view",
        source / "stable_view",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    manager = PluginManager(
        [source],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    try:
        await manager.load_all()
        root = manager._live_root
        assert root is not None
        commands = root.context.require(COMMANDS).freeze()
        result = await commands.execute(
            "/stable", session_key="chat", channel="test",
            chat_id="room", sender="user",
        )
        assert result is not None
        assert result.result.kind == "success"
        assert "current runtime graph" in result.result.text
        assert "commands" in result.result.text
        assert "stable_view" in result.result.text
        assert "mcp_provider_unavailable" in result.result.text
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_stable_command_projects_real_failed_and_cleanup_pending_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live stable consumer renders one failed owner and its cleanup state."""
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus
    from tests.fixtures.plugin_workspace import initialize_plugin_workspace

    source = tmp_path / "plugins"
    shutil.copytree(
        Path(__file__).parents[1] / "plugins/commands",
        source / "commands",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        Path(__file__).parents[1] / "plugins/stable_view",
        source / "stable_view",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    def write_checked(name: str, text: str) -> Path:
        entry = source / name / "plugin.py"
        entry.parent.mkdir()
        tree = ast.parse(text, filename=str(entry))
        compile(tree, str(entry), "exec")
        entry.write_text(text, encoding="utf-8")
        return entry

    write_checked(
        "bad_import",
        "import os\n"
        "api_version = 3\nname = 'bad_import'\nversion = '1.0.0'\n"
        "inject = ()\n"
        "if os.environ.get('STABLE_IMPORT_FAIL') == 'yes':\n"
        "    raise ImportError('stable import blocked')\n"
        "async def apply(ctx):\n    return None\n",
    )
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    monkeypatch.setenv("STABLE_IMPORT_FAIL", "no")
    first_bus = EventBus()
    first = PluginManager(
        [source], event_bus=first_bus, workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    try:
        try:
            await first.load_all()
            selected = first._selection.read()
            assert selected is not None
            bad_refs = tuple(
                ref for ref in first._selection_components(selected)
                if first._archive.read_descriptor(ref)["plugin_id"] == "bad_import"
            )
            assert len(bad_refs) == 1
            bad_ref = bad_refs[0]
        finally:
            await first.terminate_all()
    finally:
        await first_bus.aclose()

    monkeypatch.setenv("STABLE_IMPORT_FAIL", "yes")
    cleanup_fail = True
    cleanup_attempts = 0
    trace: list[tuple[str, str]] = []
    observed: Any = None
    event_bus = EventBus()
    manager = PluginManager(
        [source], event_bus=event_bus, workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    original_load_live = manager._load_live_generation

    async def observe_load_live(generation: Any) -> None:
        nonlocal observed, cleanup_attempts
        if generation.plugin_id == "bad_import":
            trace.append(("load", generation.generation_id))
        if observed is None and generation.plugin_id == "bad_import":
            observed = generation

            async def cleanup() -> None:
                nonlocal cleanup_attempts
                cleanup_attempts += 1
                if cleanup_fail:
                    raise RuntimeError("stable cleanup blocked")
                trace.append(("cleanup-done", generation.generation_id))

            generation.scope.defer("stable-bad-cleanup", cleanup)
        await original_load_live(generation)

    monkeypatch.setattr(manager, "_load_live_generation", observe_load_live)
    try:
        await manager.load_all()
        failed = observed
        assert failed is not None and failed is manager.generation("bad_import")
        assert failed.archive_ref == bad_ref
        assert failed.state == "failed"
        assert failed.fiber is None
        assert failed.load_error is not None
        assert str(failed.load_error) == "stable import blocked"
        old_load_error = failed.load_error
        assert failed.scope.closed is False
        assert manager._draining_generations["bad_import"] == [failed]
        assert failed.module_path in sys.modules
        assert failed.module_path in manager._fresh_importer._roots
        assert cleanup_attempts == 1

        root = manager._live_root
        assert root is not None
        commands = root.context.require(COMMANDS).freeze()
        result = await commands.execute(
            "/stable", session_key="chat", channel="test",
            chat_id="room", sender="user",
        )
        assert result is not None
        assert result.result.kind == "success"
        assert "bad_import" in result.result.text
        assert "stable import blocked" in result.result.text
        assert "cleanup pending" in result.result.text
        assert result.result.text.index("bad_import") < result.result.text.index("stable import blocked")
        assert result.result.text.index("stable import blocked") < result.result.text.index("cleanup pending")
        assert "mcp_provider_unavailable" in result.result.text

        from agent.plugin_composition.runtime_catalog import build_runtime_catalog

        catalog = build_runtime_catalog(
            root, manager._active_generations, manager._draining_generations,
        )
        item = next(
            item for item in cast(list[dict[str, object]], catalog["plugins"])
            if item["id"] == "bad_import"
        )
        assert item["load_error"] == "stable import blocked"
        assert item["cleanup_pending"] is True

        cleanup_fail = False
        with pytest.raises(ImportError, match="stable import blocked"):
            await manager.retry_runtime_recovery("bad_import")
        fresh_failed = manager.generation("bad_import")
        assert fresh_failed is not None and fresh_failed is not failed
        assert failed.scope.closed
        assert failed.load_error is old_load_error
        assert failed.module_path not in sys.modules
        assert failed.module_path not in manager._fresh_importer._roots
        assert fresh_failed.load_error is not None
        assert trace.index(("cleanup-done", failed.generation_id)) < trace.index(
            ("load", fresh_failed.generation_id)
        )
        fresh_catalog = build_runtime_catalog(
            root, manager._active_generations, manager._draining_generations,
        )
        fresh_item = next(
            item for item in cast(list[dict[str, object]], fresh_catalog["plugins"])
            if item["id"] == "bad_import"
        )
        assert fresh_item["load_error"] == "stable import blocked"
        assert fresh_item["cleanup_pending"] is False

        monkeypatch.setenv("STABLE_IMPORT_FAIL", "no")
        await manager.retry_runtime_recovery("bad_import")
        recovered = manager.generation("bad_import")
        assert recovered is not None and recovered is not fresh_failed
        assert recovered.state == "active"
        assert recovered.fiber is not None
        assert recovered.load_error is None
    finally:
        cleanup_fail = False
        try:
            await manager.terminate_all()
        finally:
            await event_bus.aclose()

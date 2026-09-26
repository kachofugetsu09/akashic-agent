from __future__ import annotations
import ast
from pathlib import Path
from typing import cast
import pytest
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from agent.plugin_composition import FiberState
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus

def _v3_source(
    name: str,
    *,
    version: str = "1.0.0",
    body: str = "    return None\n",
    exports: str = "",
) -> str:
    """Create one minimal v3 module with the exact Core admission contract."""

    return (
        "api_version = 3\n"
        f"name = {name!r}\n"
        f"version = {version!r}\n"
        f"{exports}"
        "async def apply(ctx):\n"
        f"{body}"
    )

def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir

def _write_checked_plugin(root: Path, name: str, source: str) -> Path:
    """Parse and compile a dynamic fixture before writing its source."""
    entry = root / name / "plugin.py"
    tree = ast.parse(source, filename=str(entry))
    compile(tree, str(entry), "exec")
    return _write_plugin(root, name, source)

def _manager(
    tmp_path: Path,
    *,
    workspace: Path | None = None,
) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=workspace or tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )

@pytest.mark.asyncio
async def test_live_selection_compile_abort_then_commit(tmp_path: Path) -> None:
    """A bad source keeps A; a valid update commits B in the same Root."""
    plugin = _write_plugin(
        tmp_path / "plugins", "selection", _v3_source("selection", version="release-a"),
    )
    initialize_plugin_workspace(tmp_path / "workspace")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        root = manager.live_root
        old = manager.generation("selection")
        selected_a = manager._selection.read()
        assert root is not None and old is not None and selected_a is not None

        (plugin / "plugin.py").write_text("def invalid(:\n", encoding="utf-8")
        result = await manager.reconcile_changed()
        assert result[0]["publication_state"] == "source_unavailable"
        assert manager._selection.read() == selected_a
        assert manager.generation("selection") is old
        assert old.fiber is not None and old.fiber.state == FiberState.ACTIVE

        (plugin / "plugin.py").write_text(
            _v3_source("selection", version="release-b"), encoding="utf-8",
        )
        committed = await manager.reconcile_changed()
        current = manager.generation("selection")
        selected_b = manager._selection.read()
        assert committed[0]["publication_state"] == "active"
        assert selected_b is not None and selected_b != selected_a
        assert current is not None and current is not old
        assert current.instance.version == "release-b"
        assert old.scope.closed
        assert manager.live_root is root
    finally:
        await manager.terminate_all()

@pytest.mark.asyncio
async def test_cold_selected_import_failure_retains_failed_generation_and_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected pre-Fiber import failure keeps B while a peer still starts."""
    bad_source = (
        "import os\n"
        + _v3_source(
            "cold_bad",
            exports=(
                "if os.environ.get('COLD_IMPORT_FAIL') == 'yes':\n"
                "    raise RuntimeError('cold import blocked')\n"
            ),
        )
    )
    _write_checked_plugin(tmp_path / "plugins", "cold_bad", bad_source)
    _write_checked_plugin(tmp_path / "plugins", "cold_peer", _v3_source("cold_peer"))
    initialize_plugin_workspace(tmp_path / "workspace")

    monkeypatch.setenv("COLD_IMPORT_FAIL", "no")
    first = _manager(tmp_path)
    try:
        await first.load_all()
        selected = first._selection.read()
        assert selected is not None
    finally:
        await first.terminate_all()

    monkeypatch.setenv("COLD_IMPORT_FAIL", "yes")
    manager = _manager(tmp_path)
    try:
        await manager.load_all()
        failed = manager.generation("cold_bad")
        peer = manager.generation("cold_peer")
        assert failed is not None
        assert failed.state == "failed"
        assert failed.load_error is not None
        assert str(failed.load_error) == "cold import blocked"
        assert failed.fiber is None
        assert failed.scope.closed
        assert manager._selection.read() == selected
        assert not manager._draining_generations.get("cold_bad")
        assert peer is not None and peer.fiber is not None
        assert peer.fiber.state == FiberState.ACTIVE

        status = next(
            item for item in cast(list[dict[str, object]], manager.plugin_status()["plugins"])
            if item["plugin_id"] == "cold_bad"
        )
        assert status["state"] == "failed"
        assert status["load_error"] == "cold import blocked"
        assert status["cleanup_pending"] is False

        from agent.plugin_composition.runtime_catalog import build_runtime_catalog

        root = manager._live_root
        assert root is not None
        catalog = build_runtime_catalog(
            root,
            manager._active_generations,
            manager._draining_generations,
        )
        view = next(
            item for item in cast(list[dict[str, object]], catalog["plugins"])
            if item["id"] == "cold_bad"
        )
        assert view["api_version"] == 3
        assert view["archive_ref"] == failed.archive_ref
        assert view["state"] == "failed"
        assert view["load_error"] == "cold import blocked"
        assert view["cleanup_pending"] is False
        assert cast(dict[str, object], view["composition"])["ready"] is False
        assert cast(dict[str, object], view["composition"])["fibers"] == []
    finally:
        await manager.terminate_all()

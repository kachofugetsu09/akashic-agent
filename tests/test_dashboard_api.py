from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.manager import PluginManager
from bootstrap.dashboard_api import build_dashboard_server
from bootstrap.dashboard_api import create_dashboard_app
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


def test_retired_proactive_dashboard_routes_do_not_open_legacy_database(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "proactive.db"
    legacy_bytes = b"legacy proactive database, not sqlite\x00"
    legacy.write_bytes(legacy_bytes)
    before = (legacy.stat().st_ino, hashlib.sha256(legacy_bytes).hexdigest())

    with TestClient(create_dashboard_app(tmp_path)) as client:
        for route in (
            "/api/dashboard/proactive/overview",
            "/api/dashboard/proactive/tick_logs",
            "/api/dashboard/proactive/tick_logs/legacy/steps",
        ):
            assert client.get(route).status_code == 404

    assert (
        legacy.stat().st_ino,
        hashlib.sha256(legacy.read_bytes()).hexdigest(),
    ) == before


def test_core_dashboard_has_no_privileged_memory_routes(tmp_path: Path) -> None:
    with TestClient(create_dashboard_app(tmp_path)) as client:
        assert client.get("/api/dashboard/memory/engine-info").status_code == 404
        assert client.get("/api/dashboard/memories").status_code == 404
        assert client.delete("/api/dashboard/interactions/turn:1").status_code == 404


def _test_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_standalone_dashboard_does_not_import_plugin_backend(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    plugin_base = home / ".akashic-plugin" / "cache" / "github" / "observe"
    plugin_dir = plugin_base / ".artifacts" / "1.0.0-test"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "db.py").write_text(
        "def ping():\n" "    return 'ok'\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('backend-imported').write_text('yes')\n"
        "from .db import ping\n"
        "def register(app, context):\n"
        "    @app.get('/api/dashboard/test-relative-import')\n"
        "    def route():\n"
        "        return {'value': ping()}\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'observe'\n"
        "version = '1.0.0'\n"
        "from importlib import import_module\nfrom agent.plugin_composition.ui import UI\ninject = (UI,)\n"
        "async def apply(ctx):\n    await ctx.require(UI).register(ctx, dashboard=lambda: import_module('.dashboard', __package__))\n",
        encoding="utf-8",
    )
    pointer = ArtifactPointer(".artifacts/1.0.0-test")
    _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
    manifest_path = home / ".akashic-plugin" / "manifest.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '[plugins."observe@github"]\nenabled = true\n', encoding="utf-8"
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))
    source_before = _test_tree_digest(plugin_dir)

    with TestClient(create_dashboard_app(tmp_path)) as client:
        response = client.get("/api/dashboard/test-relative-import")
    assert response.status_code == 404
    assert not (plugin_dir / "backend-imported").exists()
    assert _test_tree_digest(plugin_dir) == source_before


@pytest.mark.asyncio
async def test_dashboard_server_supplies_same_app_routes_to_real_manager(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    manager = PluginManager(
        [],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home",
    )
    try:
        server = build_dashboard_server(workspace=workspace, plugin_manager=manager)
        app = server.config.app
        assert manager._dashboard_routes == tuple(app.routes)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError, match="Dashboard host routes 不能重复配置"):
            manager.configure_dashboard_routes(tuple(app.routes))
        await manager.load_all()
        assert manager.live_root is not None
        with pytest.raises(RuntimeError, match="Dashboard host routes 必须在 live Root 前配置"):
            manager.configure_dashboard_routes(tuple(app.routes))
    finally:
        await manager.terminate_all()

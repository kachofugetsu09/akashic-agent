from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.plugins.manager import PluginManager
from bootstrap.app import AppRuntime
from bus.event_bus import EventBus


@pytest.mark.asyncio
async def test_runtime_install_waits_until_latest_is_leasable(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin" / "baseline"
    builtin.mkdir(parents=True)
    (builtin / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class BaselinePlugin(Plugin):\n"
        "    name = 'baseline'\n",
        encoding="utf-8",
    )
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class CandidatePlugin(Plugin):\n"
        "    name = 'candidate'\n"
        "    version = '1.0.0'\n",
        encoding="utf-8",
    )
    _commit(source)
    bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[builtin.parent],
        event_bus=bus,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "plugins-home" / "cache",
    )
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None
    app = object.__new__(AppRuntime)
    app.workspace = tmp_path / "workspace"
    app.core = SimpleNamespace(plugin_manager=manager)
    app._plugin_management_lock = asyncio.Lock()

    result = await app._install_plugin(str(source), "lab", "", [])

    assert result["pluginId"] == "candidate@lab"
    assert result["publicationState"] == "latest_ready"
    assert manager.current_snapshot is stable
    assert "candidate@lab" not in stable.generations
    assert manager.latest_snapshot is not None
    assert "candidate@lab" in manager.latest_snapshot.generations
    latest_lease = manager.snapshot_store.lease(selector="latest")
    await latest_lease.release()

    promoted = await app._promote_plugin("candidate@lab")

    assert promoted["publication_state"] == "promoted"
    assert manager.current_snapshot is manager.latest_snapshot
    await manager.terminate_all()
    await bus.aclose()


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "plugin.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "candidate"], cwd=repo, check=True)

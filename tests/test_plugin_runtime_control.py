from __future__ import annotations

import asyncio
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.install import PluginInstallResult, install_git_plugin
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

    result = await app._install_plugin(str(source), "lab", "", [])

    assert result["pluginId"] == "candidate@lab"
    assert result["publicationState"] == "latest_ready"
    candidate = result["candidate"]
    assert isinstance(candidate, dict)
    assert candidate["candidateRuntimeRevision"] == manager.candidate_status()[
        "candidate_source_revision"
    ]
    assert candidate["candidateReloadTransactionId"]
    assert candidate["candidateError"] == ""
    assert manager.current_snapshot is stable
    assert "candidate@lab" not in stable.generations
    assert manager.latest_snapshot is not None
    assert "candidate@lab" in manager.latest_snapshot.generations
    latest_lease = manager.snapshot_store.lease(selector="latest")
    await latest_lease.release()

    blocked_source = tmp_path / "blocked"
    blocked_source.mkdir()
    (blocked_source / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class BlockedPlugin(Plugin):\n"
        "    name = 'blocked'\n"
        "    version = '1.0.0'\n",
        encoding="utf-8",
    )
    _commit(blocked_source)
    with pytest.raises(
        RuntimeError,
        match="已有插件候选等待处理: plugin=candidate@lab phase=latest_ready",
    ):
        await app._install_plugin(str(blocked_source), "lab", "", [])
    assert not (manager.installed_plugins_home / "cache" / "lab" / "blocked").exists()

    promoted = await app._promote_plugin("candidate@lab")

    assert promoted["publication_state"] == "promoted"
    assert manager.current_snapshot is manager.latest_snapshot
    await manager.terminate_all()
    await bus.aclose()


@pytest.mark.asyncio
async def test_runtime_install_and_watcher_share_candidate_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builtin = tmp_path / "builtin" / "baseline"
    builtin.mkdir(parents=True)
    (builtin / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class BaselinePlugin(Plugin):\n"
        "    name = 'baseline'\n",
        encoding="utf-8",
    )
    sources: dict[str, Path] = {}
    for name in ("alpha", "beta"):
        source = tmp_path / name
        source.mkdir()
        (source / "plugin.py").write_text(
            "from agent.plugins import Plugin\n"
            f"class {name.title()}Plugin(Plugin):\n"
            f"    name = '{name}'\n"
            "    version = '1.0.0'\n",
            encoding="utf-8",
        )
        _commit(source)
        sources[name] = source
    bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[builtin.parent],
        event_bus=bus,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "plugins-home" / "cache",
    )
    await manager.load_all()
    _ = await asyncio.to_thread(
        install_git_plugin,
        workspace=tmp_path / "workspace",
        source=str(sources["alpha"]),
        marketplace="lab",
        plugins_home=manager.installed_plugins_home,
        stage_candidate=True,
    )
    app = object.__new__(AppRuntime)
    app.workspace = tmp_path / "workspace"
    app.core = SimpleNamespace(plugin_manager=manager)

    install, watcher = await asyncio.gather(
        app._install_plugin(str(sources["beta"]), "lab", "", []),
        manager.reconcile_changed(),
        return_exceptions=True,
    )

    assert isinstance(install, RuntimeError)
    assert "plugin=alpha@lab phase=latest_ready" in str(install)
    assert not (manager.installed_plugins_home / "cache" / "lab" / "beta").exists()
    assert not isinstance(watcher, BaseException)
    assert manager.candidate_status()["candidate_plugin_id"] == "alpha@lab"
    await manager.discard_latest_candidate("alpha@lab")

    entered = threading.Event()
    release = threading.Event()

    def blocking_install(**kwargs: Any) -> PluginInstallResult:
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release plugin install")
        return install_git_plugin(**kwargs)

    monkeypatch.setattr("agent.plugins.manager.install_git_plugin", blocking_install)
    cancelled_install = asyncio.create_task(
        app._install_plugin(str(sources["beta"]), "lab", "", [])
    )
    assert await asyncio.to_thread(entered.wait, 5)
    cancelled_install.cancel()
    waiting_watcher = asyncio.create_task(manager.reconcile_changed())
    await asyncio.sleep(0)
    assert not waiting_watcher.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_install
    _ = await waiting_watcher
    assert manager.candidate_status()["candidate_plugin_id"] == "beta@lab"
    assert manager.candidate_status()["candidate_state"] == "latest_ready"
    await manager.discard_latest_candidate("beta@lab")
    await manager.terminate_all()
    await bus.aclose()


def _commit(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "plugin.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "candidate"], cwd=repo, check=True)

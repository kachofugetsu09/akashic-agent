from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from agent.supervisor import _SupervisorLock
from agent.plugins.artifacts import read_pointers
from agent.plugins.trusted_install import (
    install_trusted_plugin_batch,
    load_trusted_plugin_batch,
)
from bootstrap.tools import CoreRuntime
from bootstrap.workspace_lock import (
    PluginPublicationLock,
    WorkspaceInstanceLock,
    WorkspaceMaintenanceLock,
)


def test_trusted_batch_installs_exact_v3_plugins_as_stable(tmp_path: Path) -> None:
    repositories = [
        _create_plugin_repository(tmp_path / "citation", "citation"),
        _create_plugin_repository(tmp_path / "steam", "steam"),
    ]
    batch_path = tmp_path / "trusted.json"
    batch_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plugins": [
                    {
                        "source": str(repository),
                        "marketplace": "lab",
                        "ref": _git_output(repository, "rev-parse", "HEAD"),
                    }
                    for repository in repositories
                ],
            }
        ),
        encoding="utf-8",
    )

    home = tmp_path / "plugins-home"
    receipt = install_trusted_plugin_batch(
        workspace=tmp_path / "workspace",
        batch_path=batch_path,
        plugins_home=home,
    )

    assert receipt["mode"] == "operator_trusted_offline_batch"
    assert receipt["programmaticValidation"] == "bypassed_by_operator_trust"
    installed = receipt["plugins"]
    assert isinstance(installed, list)
    assert [item["pluginId"] for item in installed] == [
        "citation@lab",
        "steam@lab",
    ]
    for item in installed:
        plugin_name = str(item["pluginId"]).split("@", maxsplit=1)[0]
        pointers = read_pointers(home / "cache" / "lab" / plugin_name)
        assert pointers is not None
        assert pointers.stable == pointers.latest
        assert str(item["sourceRevision"])[:16] in str(item["installedPath"])


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {
                "schema_version": 1,
                "plugins": [
                    {
                        "source": "https://example.invalid/plugin.git",
                        "marketplace": "github",
                        "ref": "main",
                    }
                ],
            },
            "完整 commit SHA",
        ),
        (
            {"schema_version": 1, "plugins": [], "future": True},
            "只接受 schema_version 和 plugins",
        ),
    ],
)
def test_trusted_batch_rejects_ambiguous_or_unknown_input(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    batch_path = tmp_path / "trusted.json"
    batch_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_trusted_plugin_batch(batch_path)


def test_maintenance_lock_fences_both_runtime_owners(tmp_path: Path) -> None:
    maintenance = WorkspaceMaintenanceLock(tmp_path)
    supervisor = _SupervisorLock(tmp_path)
    supervisor.acquire()
    with pytest.raises(RuntimeError, match="生命周期 owner"):
        maintenance.acquire()
    supervisor.release()

    runtime = WorkspaceInstanceLock(tmp_path)
    runtime.acquire()
    with pytest.raises(RuntimeError, match="生命周期 owner"):
        maintenance.acquire()
    runtime.release()

    maintenance.acquire()
    maintenance.release()


@pytest.mark.asyncio
async def test_core_runtime_holds_shared_plugin_home_publication_lock(
    tmp_path: Path,
) -> None:
    home = tmp_path / "shared-plugin-home"
    observed: list[str] = []

    async def noop() -> None:
        return None

    class PluginManager:
        loaded_count = 0

        async def load_all(self) -> None:
            competitor = PluginPublicationLock(home)
            with pytest.raises(RuntimeError, match="发布或消费 owner"):
                competitor.acquire()
            observed.append("publication-fenced")

        async def terminate_all(self) -> None:
            observed.append("plugins-stopped")

    runtime = object.__new__(CoreRuntime)
    runtime.plugin_manager = PluginManager()  # type: ignore[assignment]
    runtime.workspace = None
    runtime.plugin_publication_lock = PluginPublicationLock(home)
    runtime._plugin_publication_locked = False
    runtime.tools = SimpleNamespace(get_tool=lambda _name: None)
    runtime.loop = SimpleNamespace(shutdown_compaction=noop)
    runtime.event_bus = SimpleNamespace(aclose=noop)
    runtime.session_manager = SimpleNamespace(close=lambda: None)

    await runtime.start()

    assert observed == ["publication-fenced"]
    await runtime.stop()
    assert observed == ["publication-fenced", "plugins-stopped"]
    after_shutdown = PluginPublicationLock(home)
    after_shutdown.acquire()
    after_shutdown.release()


@pytest.mark.asyncio
async def test_inspect_modules_holds_plugin_publication_lock(tmp_path: Path) -> None:
    home = tmp_path / "shared-plugin-home"

    async def load_and_stop() -> None:
        competitor = PluginPublicationLock(home)
        with pytest.raises(RuntimeError, match="发布或消费 owner"):
            competitor.acquire()
        raise RuntimeError("stop after inspect lock proof")

    async def noop() -> None:
        return None

    runtime = object.__new__(CoreRuntime)
    runtime.plugin_manager = SimpleNamespace(
        load_all=load_and_stop,
        terminate_all=noop,
    )
    runtime.plugin_publication_lock = PluginPublicationLock(home)
    runtime._plugin_publication_locked = False
    runtime.tools = SimpleNamespace(get_tool=lambda _name: None)
    runtime.loop = SimpleNamespace(shutdown_compaction=noop)
    runtime.event_bus = SimpleNamespace(aclose=noop)
    runtime.session_manager = SimpleNamespace(close=lambda: None)

    with pytest.raises(RuntimeError, match="inspect lock proof"):
        await runtime.inspect_modules()
    await runtime.stop()

    after_shutdown = PluginPublicationLock(home)
    after_shutdown.acquire()
    after_shutdown.release()


def test_trusted_batch_command_rejects_active_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN", "turn:owner")
    monkeypatch.setattr(
        main.sys,
        "argv",
        [
            "main.py",
            "plugin-install-trusted-batch",
            "--workspace",
            str(tmp_path / "workspace"),
            "--batch",
            str(tmp_path / "batch.json"),
            "--confirm-trusted",
        ],
    )

    with pytest.raises(SystemExit, match="不能由 active turn 调用"):
        main._run_lightweight_command()


def test_trusted_batch_command_rejects_shared_home_used_by_other_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "shared-plugin-home"
    owner = PluginPublicationLock(home)
    owner.acquire()
    monkeypatch.delenv("AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN", raising=False)
    monkeypatch.setattr(
        main.sys,
        "argv",
        [
            "main.py",
            "plugin-install-trusted-batch",
            "--workspace",
            str(tmp_path / "idle-workspace"),
            "--plugins-home",
            str(home),
            "--batch",
            str(tmp_path / "missing.json"),
            "--confirm-trusted",
        ],
    )

    try:
        with pytest.raises(SystemExit, match="发布或消费 owner"):
            main._run_lightweight_command()
    finally:
        owner.release()

    workspace_owner = WorkspaceInstanceLock(tmp_path / "idle-workspace")
    workspace_owner.acquire()
    workspace_owner.release()


def test_trusted_batch_reports_completed_plugins_before_v2_failure(
    tmp_path: Path,
) -> None:
    accepted = _create_plugin_repository(tmp_path / "citation", "citation")
    rejected = _create_plugin_repository(
        tmp_path / "legacy",
        "legacy",
        api_version=2,
    )
    batch_path = tmp_path / "trusted.json"
    batch_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plugins": [
                    {
                        "source": str(repository),
                        "marketplace": "lab",
                        "ref": _git_output(repository, "rev-parse", "HEAD"),
                    }
                    for repository in (accepted, rejected)
                ],
            }
        ),
        encoding="utf-8",
    )
    home = tmp_path / "plugins-home"

    with pytest.raises(
        RuntimeError,
        match=r"index=1 completed=\['citation@lab'\].*api_version = 3",
    ):
        install_trusted_plugin_batch(
            workspace=tmp_path / "workspace",
            batch_path=batch_path,
            plugins_home=home,
        )

    accepted_pointers = read_pointers(home / "cache" / "lab" / "citation")
    assert accepted_pointers is not None
    assert accepted_pointers.stable == accepted_pointers.latest
    assert not (home / "cache" / "lab" / "legacy" / ".pointers.json").exists()


def test_trusted_batch_command_prints_machine_readable_receipt(tmp_path: Path) -> None:
    repository = _create_plugin_repository(tmp_path / "citation", "citation")
    batch_path = tmp_path / "trusted.json"
    revision = _git_output(repository, "rev-parse", "HEAD")
    batch_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plugins": [
                    {
                        "source": str(repository),
                        "marketplace": "lab",
                        "ref": revision,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(Path(main.__file__)),
            "plugin-install-trusted-batch",
            "--workspace",
            str(tmp_path / "workspace"),
            "--plugins-home",
            str(tmp_path / "plugins-home"),
            "--batch",
            str(batch_path),
            "--confirm-trusted",
            "--json",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key != "AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN"
        },
        check=False,
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["programmaticValidation"] == "bypassed_by_operator_trust"
    assert receipt["plugins"][0]["sourceRevision"] == revision


def _create_plugin_repository(
    path: Path,
    name: str,
    *,
    api_version: int = 3,
) -> Path:
    path.mkdir(parents=True)
    (path / "plugin.py").write_text(
        f"api_version = {api_version}\nname = {name!r}\nversion = '1.0.0'\n",
        encoding="utf-8",
    )
    (path / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        f"name = {name!r}\n"
        "version = '1.0.0'\n"
        f"api_version = {api_version}\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    for args in (
        ("init",),
        ("config", "user.name", "test"),
        ("config", "user.email", "test@example.com"),
        ("add", "."),
        ("commit", "-m", "init"),
    ):
        result = subprocess.run(
            ["git", *args],
            cwd=path,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
        assert result.returncode == 0, result.stderr
    return path


def _git_output(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()

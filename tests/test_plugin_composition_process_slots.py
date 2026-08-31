from __future__ import annotations

from pathlib import Path
import sys

import pytest

from agent.plugin_composition import (
    MANAGED_PROCESSES,
    CompositionRoot,
    ManagedProcessDefinition,
    PluginRuntime,
)
from agent.plugin_composition.process_slots import (
    PluginManagedProcesses,
    _freeze_plugin_managed_processes,
)
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import RuntimeSnapshot
from bus.event_bus import EventBus


def _plugin_dir(root: Path, name: str = "calendar") -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "api.py").write_text(
        "import http.server\n"
        "import os\n"
        "class Handler(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200)\n"
        "        self.end_headers()\n"
        "    def log_message(self, format, *args):\n"
        "        pass\n"
        "server = http.server.ThreadingHTTPServer(\n"
        "    ('127.0.0.1', int(os.environ['PORT'])),\n"
        "    Handler,\n"
        ")\n"
        "server.serve_forever()\n",
        encoding="utf-8",
    )
    return plugin_dir


def _runtime(plugin_dir: Path) -> PluginRuntime:
    return PluginRuntime(
        plugin_id=plugin_dir.name,
        generation_id="test-generation",
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        workspace=plugin_dir / "workspace",
        config=None,
    )


def _definition(*, port: int = 18000) -> ManagedProcessDefinition:
    return ManagedProcessDefinition(
        name="calendar_api",
        command=("python", "api.py"),
        env={"MODE": "calendar"},
        port_env="PORT",
        formal_port=port,
        readiness_path="/health",
        startup_timeout_seconds=15.0,
    )


@pytest.mark.asyncio
async def test_process_registry_freezes_health_identity_and_cleanup(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("process-registry")
    processes = PluginManagedProcesses(root.instance_token)
    _ = await root.context.provide(MANAGED_PROCESSES, processes)
    plugin_dir = _plugin_dir(tmp_path)

    async def apply(ctx) -> None:
        await ctx.require(MANAGED_PROCESSES).register(ctx, _definition())

    fiber = await root.mount(
        apply,
        name="calendar",
        inject=(MANAGED_PROCESSES,),
        runtime=_runtime(plugin_dir),
    )
    registry = _freeze_plugin_managed_processes(
        processes,
        root.instance_token,
    )
    binding = registry["calendar_api"]
    assert binding.descriptor.owner == "calendar"
    assert binding.definition.env["MODE"] == "calendar"
    assert binding.is_live()
    assert not hasattr(processes, "freeze")
    incident = binding.incident_reporter(
        "process_readiness_failed",
        "calendar_api did not become ready",
    )
    assert incident.owner == "calendar"
    assert root.recent_incidents() == (incident,)

    await fiber.dispose()
    assert not binding.is_live()
    assert _freeze_plugin_managed_processes(
        processes,
        root.instance_token,
    ) is registry
    await root.dispose()


@pytest.mark.asyncio
async def test_process_registry_identity_ignores_root_and_runtime_path(
    tmp_path: Path,
) -> None:
    identities: list[str] = []
    for suffix in ("candidate", "formal"):
        root = CompositionRoot(f"process-{suffix}")
        processes = PluginManagedProcesses(root.instance_token)
        _ = await root.context.provide(MANAGED_PROCESSES, processes)
        plugin_dir = _plugin_dir(tmp_path / suffix)

        async def apply(ctx) -> None:
            await ctx.require(MANAGED_PROCESSES).register(ctx, _definition())

        _ = await root.mount(
            apply,
            name="calendar",
            inject=(MANAGED_PROCESSES,),
            runtime=_runtime(plugin_dir),
        )
        identities.append(
            _freeze_plugin_managed_processes(
                processes,
                root.instance_token,
            ).identity
        )
        await root.dispose()
    assert identities[0] == identities[1]


@pytest.mark.asyncio
async def test_process_registry_rejects_invalid_endpoint_and_artifact_escape(
    tmp_path: Path,
) -> None:
    plugin_dir = _plugin_dir(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (plugin_dir / "outside-link").symlink_to(outside, target_is_directory=True)
    definitions = (
        ManagedProcessDefinition(
            name="bad_port",
            command=("python", "api.py"),
            formal_port=0,
        ),
        ManagedProcessDefinition(
            name="bad_ready",
            command=("python", "api.py"),
            formal_port=18000,
            readiness_path="https://example.com/health",
        ),
        ManagedProcessDefinition(
            name="escaped",
            command=("python", "api.py"),
            cwd="outside-link",
            formal_port=18000,
        ),
    )
    for index, definition in enumerate(definitions):
        root = CompositionRoot(f"process-invalid-{index}")
        processes = PluginManagedProcesses(root.instance_token)
        _ = await root.context.provide(MANAGED_PROCESSES, processes)

        async def apply(ctx, definition=definition) -> None:
            await ctx.require(MANAGED_PROCESSES).register(ctx, definition)

        _ = await root.mount(
            apply,
            name="calendar",
            inject=(MANAGED_PROCESSES,),
            runtime=_runtime(plugin_dir),
        )
        assert not root.receipt().ready
        registry = _freeze_plugin_managed_processes(
            processes,
            root.instance_token,
        )
        assert len(registry) == 0
        await root.dispose()


@pytest.mark.asyncio
async def test_process_registry_rejects_context_from_another_root(
    tmp_path: Path,
) -> None:
    root_a = CompositionRoot("process-root-a")
    root_b = CompositionRoot("process-root-b")
    processes_a = PluginManagedProcesses(root_a.instance_token)
    processes_b = PluginManagedProcesses(root_b.instance_token)
    _ = await root_a.context.provide(MANAGED_PROCESSES, processes_a)
    _ = await root_b.context.provide(MANAGED_PROCESSES, processes_b)
    plugin_dir = _plugin_dir(tmp_path)

    async def apply(ctx) -> None:
        await processes_a.register(ctx, _definition())

    _ = await root_b.mount(
        apply,
        name="calendar",
        inject=(MANAGED_PROCESSES,),
        runtime=_runtime(plugin_dir),
    )

    assert any(
        "插件 managed process Service 不属于当前 Root" in (fiber.error or "")
        for fiber in root_b.receipt().fibers
    )
    assert root_a.receipt().health == ()
    assert root_b.receipt().health == ()
    assert root_a.receipt().effects == (
        "root:service:core.managed_processes",
    )
    assert root_b.receipt().effects == (
        "root:service:core.managed_processes",
    )
    assert (
        len(
            _freeze_plugin_managed_processes(
                processes_a,
                root_a.instance_token,
            )
        )
        == 0
    )
    assert (
        len(
            _freeze_plugin_managed_processes(
                processes_b,
                root_b.instance_token,
            )
        )
        == 0
    )

    await root_b.dispose()
    await root_a.dispose()


def _manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


def _source(version: str) -> str:
    return (
        "from agent.plugin_composition import (\n"
        "    MANAGED_PROCESSES, ManagedProcessDefinition,\n"
        ")\n"
        "api_version = 3\n"
        "name = 'calendar'\n"
        f"version = '{version}'\n"
        "inject = (MANAGED_PROCESSES,)\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(MANAGED_PROCESSES).register(\n"
        "        ctx, ManagedProcessDefinition(\n"
        "            name='calendar_api', command=('python', 'api.py'),\n"
        f"            env={{'VERSION': '{version}'}}, formal_port=18000,\n"
        "            readiness_path='/health',\n"
        "        ),\n"
        "    )\n"
    )


def _write_plugin(tmp_path: Path, version: str) -> Path:
    plugin_dir = _plugin_dir(tmp_path / "plugins")
    _write_plugin_version(plugin_dir, version)
    requirements = plugin_dir / "requirements.txt"
    requirements.write_text("", encoding="utf-8")
    interpreter = plugin_dir / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        f"#!/bin/sh\nexec {sys.executable} \"$@\"\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    return plugin_dir


def _write_plugin_version(plugin_dir: Path, version: str) -> None:
    (plugin_dir / "plugin.py").write_text(_source(version), encoding="utf-8")
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        'name = "calendar"\n'
        f'version = "{version}"\n'
        "api_version = 3\n"
        'entrypoint = "plugin.py"\n\n'
        "[[python]]\n"
        'requirements = "requirements.txt"\n\n'
        "[[processes]]\n"
        'name = "calendar_api"\n'
        'command = ["python", "api.py"]\n'
        f'env = {{VERSION = "{version}"}}\n'
        'port_env = "PORT"\n'
        "formal_port = 18000\n"
        'readiness_path = "/health"\n',
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_manager_keeps_process_registry_private_and_rebuilds_formal(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(tmp_path, "1")
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None and stable.managed_process_registry is not None
    stable_registry = stable.managed_process_registry
    assert stable_registry["calendar_api"].definition.env["VERSION"] == "1"

    _write_plugin_version(plugin_dir, "2")
    candidate = await manager.prepare_candidate("calendar")
    assert candidate is not None and candidate.runtime_snapshot is not None
    candidate_registry = candidate.runtime_snapshot.managed_process_registry
    assert candidate_registry is not None
    assert candidate_registry["calendar_api"].definition.env["VERSION"] == "2"
    assert manager.current_snapshot is stable
    assert manager.current_snapshot.managed_process_registry is stable_registry

    result = await manager.publish_prepared("calendar")
    assert result["publication_state"] == "committed"
    current = manager.current_snapshot
    assert current is not None and current.managed_process_registry is not None
    assert current.managed_process_registry is not candidate_registry
    assert current.managed_process_registry["calendar_api"].definition.env[
        "VERSION"
    ] == "2"
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_manager_rejects_process_registry_from_another_root(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(tmp_path, "1")
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None and stable.managed_process_registry is not None
    stable_registry = stable.managed_process_registry

    _write_plugin_version(plugin_dir, "2")
    candidate = await manager.prepare_candidate("calendar")
    assert candidate is not None and candidate.runtime_snapshot is not None

    def replace_registry(snapshot: RuntimeSnapshot) -> None:
        snapshot.managed_process_registry = stable_registry
        snapshot.managed_process_registry_identity = stable_registry.identity

    async def release_validation(_snapshot: RuntimeSnapshot) -> None:
        return None

    manager.bind_dashboard_preparer(
        replace_registry,
        validation_releaser=release_validation,
    )
    with pytest.raises(RuntimeError, match="managed process registry"):
        await manager.publish_prepared("calendar")
    assert manager.current_snapshot is stable

    await manager.discard_prepared("calendar")
    await manager.terminate_all()

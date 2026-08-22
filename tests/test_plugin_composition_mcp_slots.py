from __future__ import annotations

import asyncio
import os
import sys
import socket
import shutil
from pathlib import Path

import pytest

from agent.plugin_composition import (
    MCP_SERVERS,
    CompositionError,
    CompositionRoot,
    EndpointEnv,
    McpServerDefinition,
    PluginRuntime,
)
from agent.plugin_composition.channels import (
    ChannelCapability,
    ChannelDeliveryReceipt,
    ChannelFactoryContext,
    ChannelReady,
    CoreChannelDefinition,
    DeliveryStatus,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    StopReceipt,
)
from agent.control.models import TurnRequest
from agent.control.ports import ControlExecutionResult
from agent.control.runtime import ConversationRuntime
from agent.plugin_composition.mcp_slots import (
    PluginMcpServers,
    _freeze_plugin_mcp_servers,
)
from agent.plugins.manager import PluginManager
from agent.plugins.artifacts import ArtifactPointer, read_pointers, write_pointers
from agent.plugins.manifest import write_plugin_manifest
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from bus.events import InboundMessage
from session.store import SessionStore
from utils.process_group import OwnedProcessGroup


def _runtime(plugin_dir: Path) -> PluginRuntime:
    return PluginRuntime(
        plugin_id=plugin_dir.name,
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        workspace=plugin_dir / "workspace",
        config=None,
    )


def _definition(*, candidate_backend: str = "recording") -> McpServerDefinition:
    return McpServerDefinition(
        name="calendar",
        command=("python", "mcp.py"),
        cwd=".",
        env={"MODE": "stdio"},
        required_tools=("get_events",),
        candidate_read_only_tools=("get_events",),
        endpoint_env=(EndpointEnv("PORT", "calendar_api"),),
        candidate_env={"CALENDAR_BACKEND": candidate_backend},
    )


def _plugin_dir(root: Path, name: str = "calendar") -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "mcp.py").write_text("print('probe')\n", encoding="utf-8")
    return plugin_dir


@pytest.mark.asyncio
async def test_mcp_registry_freezes_descriptor_health_and_cleanup(
    tmp_path: Path,
) -> None:
    plugin_dir = _plugin_dir(tmp_path)
    root = CompositionRoot("mcp-registry")
    servers = PluginMcpServers(root.instance_token)
    _ = await root.context.provide(MCP_SERVERS, servers)

    async def apply(ctx) -> None:
        await ctx.require(MCP_SERVERS).register(ctx, _definition())

    fiber = await root.mount(
        apply,
        name="calendar",
        inject=(MCP_SERVERS,),
        runtime=_runtime(plugin_dir),
    )
    registry = _freeze_plugin_mcp_servers(servers, root.instance_token)
    binding = registry["calendar"]
    assert binding.descriptor.owner == "calendar"
    assert binding.descriptor.endpoint_env == (
        EndpointEnv("PORT", "calendar_api"),
    )
    assert binding.health.healthy
    assert binding.is_live()
    assert registry.identity == registry.catalog_digest
    incident = binding.incident_reporter(
        "mcp_handshake_failed",
        "calendar initialize failed",
    )
    assert incident.owner == "calendar"
    assert root.recent_incidents() == (incident,)

    await fiber.dispose()
    assert _freeze_plugin_mcp_servers(servers, root.instance_token) is registry
    assert not binding.is_live()
    assert root.receipt().effects == ("root:service:core.mcp_servers",)
    await root.dispose()


@pytest.mark.asyncio
async def test_mcp_registry_rejects_duplicate_frozen_and_reserved_env(
    tmp_path: Path,
) -> None:
    plugin_dir = _plugin_dir(tmp_path)
    root = CompositionRoot("mcp-invalid")
    servers = PluginMcpServers(root.instance_token)
    _ = await root.context.provide(MCP_SERVERS, servers)
    captured = None

    async def apply(ctx) -> None:
        nonlocal captured
        captured = ctx
        await ctx.require(MCP_SERVERS).register(ctx, _definition())

    _ = await root.mount(
        apply,
        name="calendar",
        inject=(MCP_SERVERS,),
        runtime=_runtime(plugin_dir),
    )
    _ = _freeze_plugin_mcp_servers(servers, root.instance_token)
    assert captured is not None
    with pytest.raises(CompositionError, match="已冻结"):
        await servers.register(captured, _definition())
    await root.dispose()

    root = CompositionRoot("mcp-reserved")
    servers = PluginMcpServers(root.instance_token)
    _ = await root.context.provide(MCP_SERVERS, servers)

    async def reserved(ctx) -> None:
        definition = McpServerDefinition(
            name="calendar",
            command=("python",),
            env={"AKASHIC_WORKSPACE": "/tmp/escape"},
        )
        await ctx.require(MCP_SERVERS).register(ctx, definition)

    _ = await root.mount(
        reserved,
        name="calendar",
        inject=(MCP_SERVERS,),
        runtime=_runtime(plugin_dir),
    )
    assert not root.receipt().ready
    assert any(
        "env 无效" in (fiber.error or "") for fiber in root.receipt().fibers
    )
    assert len(_freeze_plugin_mcp_servers(servers, root.instance_token)) == 0
    await root.dispose()


@pytest.mark.asyncio
async def test_mcp_registry_identity_ignores_runtime_root(tmp_path: Path) -> None:
    identities: list[str] = []
    for suffix in ("candidate", "formal"):
        plugin_dir = _plugin_dir(tmp_path / suffix)
        root = CompositionRoot(f"mcp-{suffix}")
        servers = PluginMcpServers(root.instance_token)
        _ = await root.context.provide(MCP_SERVERS, servers)

        async def apply(ctx) -> None:
            await ctx.require(MCP_SERVERS).register(ctx, _definition())

        _ = await root.mount(
            apply,
            name="calendar",
            inject=(MCP_SERVERS,),
            runtime=_runtime(plugin_dir),
        )
        identities.append(
            _freeze_plugin_mcp_servers(servers, root.instance_token).identity
        )
        await root.dispose()
    assert identities[0] == identities[1]


@pytest.mark.asyncio
async def test_mcp_registry_rejects_missing_command_and_escaped_cwd(
    tmp_path: Path,
) -> None:
    plugin_dir = _plugin_dir(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (plugin_dir / "outside-link").symlink_to(outside, target_is_directory=True)
    definitions = (
        McpServerDefinition(name="missing", command=("python", "missing.py")),
        McpServerDefinition(name="escaped", command=("python",), cwd="outside-link"),
    )
    for index, definition in enumerate(definitions):
        root = CompositionRoot(f"mcp-path-{index}")
        servers = PluginMcpServers(root.instance_token)
        _ = await root.context.provide(MCP_SERVERS, servers)

        async def apply(ctx, definition=definition) -> None:
            await ctx.require(MCP_SERVERS).register(ctx, definition)

        _ = await root.mount(
            apply,
            name="calendar",
            inject=(MCP_SERVERS,),
            runtime=_runtime(plugin_dir),
        )
        assert not root.receipt().ready
        assert (
            len(_freeze_plugin_mcp_servers(servers, root.instance_token)) == 0
        )
        await root.dispose()


@pytest.mark.asyncio
async def test_plugins_cannot_freeze_shared_mcp_declarations_early(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("mcp-core-freeze-owner")
    servers = PluginMcpServers(root.instance_token)
    _ = await root.context.provide(MCP_SERVERS, servers)

    definitions = (
        _definition(),
        McpServerDefinition(name="contacts", command=("python", "mcp.py")),
    )
    for name, definition in zip(("calendar", "contacts"), definitions, strict=True):
        plugin_dir = _plugin_dir(tmp_path, name)

        async def apply(ctx, definition=definition, name=name) -> None:
            service = ctx.require(MCP_SERVERS)
            if name == "calendar":
                with pytest.raises(AttributeError):
                    _ = getattr(service, "freeze")
            await service.register(ctx, definition)

        fiber = await root.mount(
            apply,
            name=name,
            inject=(MCP_SERVERS,),
            runtime=_runtime(plugin_dir),
        )
        assert fiber.state.value == "active"

    frozen = _freeze_plugin_mcp_servers(servers, root.instance_token)
    assert tuple(frozen) == ("calendar", "contacts")
    await root.dispose()


@pytest.mark.asyncio
async def test_mcp_registry_rejects_context_from_another_root(
    tmp_path: Path,
) -> None:
    root_a = CompositionRoot("mcp-root-a")
    root_b = CompositionRoot("mcp-root-b")
    servers_a = PluginMcpServers(root_a.instance_token)
    servers_b = PluginMcpServers(root_b.instance_token)
    _ = await root_a.context.provide(MCP_SERVERS, servers_a)
    _ = await root_b.context.provide(MCP_SERVERS, servers_b)
    plugin_dir = _plugin_dir(tmp_path)

    async def apply(ctx) -> None:
        await servers_a.register(ctx, _definition())

    _ = await root_b.mount(
        apply,
        name="calendar",
        inject=(MCP_SERVERS,),
        runtime=_runtime(plugin_dir),
    )

    assert any(
        "插件 MCP 声明 Service 不属于当前 Root" in (fiber.error or "")
        for fiber in root_b.receipt().fibers
    )
    assert root_a.receipt().health == ()
    assert root_b.receipt().health == ()
    assert root_a.receipt().effects == ("root:service:core.mcp_servers",)
    assert root_b.receipt().effects == ("root:service:core.mcp_servers",)
    assert len(_freeze_plugin_mcp_servers(servers_a, root_a.instance_token)) == 0
    assert len(_freeze_plugin_mcp_servers(servers_b, root_b.instance_token)) == 0

    await root_b.dispose()
    await root_a.dispose()


def _manager(
    tmp_path: Path,
    *,
    tool_registry: ToolRegistry | None = None,
) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=tool_registry,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


class _CoreChannelAdapter:
    def __init__(self, binding_token: str) -> None:
        self._binding_token = binding_token

    async def start(self) -> ChannelReady:
        return ChannelReady(self._binding_token)

    async def deliver(
        self,
        request: ProviderDeliveryRequest,
    ) -> ProviderDeliveryReceipt:
        return ProviderDeliveryReceipt(request.delivery_id, DeliveryStatus.DELIVERED)

    async def stop(self) -> StopReceipt:
        return StopReceipt(self._binding_token, resources_closed=True)


def _core_channel_definition() -> CoreChannelDefinition:
    def factory(context: ChannelFactoryContext) -> _CoreChannelAdapter:
        return _CoreChannelAdapter(context.binding_token)

    return CoreChannelDefinition(
        name="web",
        capabilities=frozenset({ChannelCapability.OUTBOUND}),
        factory=factory,
        inbound_identity=None,
        source_revision="core-native-v3",
        config_revision="core-native-v3",
        generation_id="core-native-v3",
    )


def _port_live(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _plugin_source(version: str, *, python_command: str = "python") -> str:
    return (
        "from agent.plugin_composition import (\n"
        "    MANAGED_PROCESSES, MCP_SERVERS, EndpointEnv,\n"
        "    ManagedProcessDefinition, McpServerDefinition,\n"
        ")\n"
        "api_version = 3\n"
        "name = 'calendar'\n"
        f"version = '{version}'\n"
        "inject = (MCP_SERVERS, MANAGED_PROCESSES)\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(MANAGED_PROCESSES).register(\n"
        "        ctx, ManagedProcessDefinition(\n"
        f"            name='calendar_api', command=({python_command!r}, 'api.py'),\n"
        "            formal_port=18000, readiness_path='/health',\n"
        "        ),\n"
        "    )\n"
        "    await ctx.require(MCP_SERVERS).register(\n"
        "        ctx, McpServerDefinition(\n"
        f"            name='calendar', command=({python_command!r}, 'mcp.py'),\n"
        "            required_tools=('get_events',),\n"
        "            candidate_read_only_tools=('get_events',),\n"
        "            endpoint_env=(EndpointEnv('PORT', 'calendar_api'),),\n"
        f"            candidate_env={{'VERSION': '{version}'}},\n"
        "        ),\n"
        "    )\n"
    )


def _write_manager_plugin(tmp_path: Path, version: str) -> Path:
    plugin_dir = _plugin_dir(tmp_path / "plugins")
    (plugin_dir / "plugin.py").write_text(
        _plugin_source(version, python_command=sys.executable),
        encoding="utf-8",
    )
    (plugin_dir / "api.py").write_text(
        "import os\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'ready')\n"
        "    def log_message(self, *_args): pass\n"
        "HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    (plugin_dir / "mcp.py").write_text(
        "import json, os, sys\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw); method = msg.get('method')\n"
        "    if method == 'initialize': result = {'protocolVersion': '2025-11-25'}\n"
        "    elif method == 'tools/list': result = {'tools': [{'name': 'get_events', "
        "'description': 'read events', 'inputSchema': {'type': 'object'}}]}\n"
        "    elif method == 'tools/call': result = {'content': [{'type': 'text', "
        "'text': '|'.join((os.environ.get('VERSION', 'formal'), "
        "os.environ['PORT'], os.environ['AKA_PLUGIN_DATA_DIR']))}]}\n"
        "    else: continue\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    return plugin_dir


def _write_static_manager_plugin(tmp_path: Path, version: str) -> Path:
    plugin_dir = _plugin_dir(tmp_path / "plugins")
    (plugin_dir / "entry.py").write_text(
        _plugin_source(version),
        encoding="utf-8",
    )
    (plugin_dir / "api.py").write_text(
        "import os\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'ready')\n"
        "    def log_message(self, *_args): pass\n"
        "HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    (plugin_dir / "mcp.py").write_text(
        "import json, os, sys\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw); method = msg.get('method')\n"
        "    if method == 'initialize': result = {'protocolVersion': '2025-11-25'}\n"
        "    elif method == 'tools/list': result = {'tools': [{'name': 'get_events', "
        "'description': 'read events', 'inputSchema': {'type': 'object'}}]}\n"
        "    elif method == 'tools/call': result = {'content': [{'type': 'text', "
        "'text': '|'.join((os.environ.get('VERSION', 'formal'), "
        "os.environ['PORT'], os.environ['AKA_PLUGIN_DATA_DIR']))}]}\n"
        "    else: continue\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    (plugin_dir / "requirements.txt").write_text("", encoding="utf-8")
    interpreter = plugin_dir / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        f"#!/bin/sh\nexec {sys.executable} \"$@\"\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = \"calendar\"\n"
        f"version = \"{version}\"\n"
        "api_version = 3\n"
        "entrypoint = \"entry.py\"\n\n"
        "[[python]]\n"
        "requirements = \"requirements.txt\"\n\n"
        "[[mcp]]\n"
        "name = \"calendar\"\n"
        "command = [\"python\", \"mcp.py\"]\n"
        "required_tools = [\"get_events\"]\n"
        "candidate_read_only_tools = [\"get_events\"]\n"
        "endpoint_env = [{env = \"PORT\", process = \"calendar_api\"}]\n"
        f"candidate_env = {{VERSION = \"{version}\"}}\n\n"
        "[[process]]\n"
        "name = \"calendar_api\"\n"
        "command = [\"python\", \"api.py\"]\n"
        "port_env = \"PORT\"\n"
        "formal_port = 18000\n"
        "readiness_path = \"/health\"\n",
        encoding="utf-8",
    )
    return plugin_dir


def _upgrade_static_manager_plugin(plugin_dir: Path, version: str) -> None:
    (plugin_dir / "entry.py").write_text(
        _plugin_source(version),
        encoding="utf-8",
    )
    manifest = plugin_dir / "akashic.plugin.toml"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        text.replace('version = "1"', f'version = "{version}"').replace(
            'VERSION = "1"',
            f'VERSION = "{version}"',
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_static_manifest_is_admission_source_and_reconciles_mcp_root(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )

    await manager.load_all()
    generation = manager._active_generations["calendar"]  # pyright: ignore[reportPrivateUsage]
    assert generation.entrypoint == "entry.py"
    assert generation.static_manifest is not None
    assert generation.static_manifest.mcp_servers[0].name == "calendar"
    assert generation.static_manifest.managed_processes[0].name == "calendar_api"
    runtime_commands = dict(generation.static_runtime_commands)
    assert runtime_commands["mcp:calendar"][0] == str(
        plugin_dir / ".venv" / "bin" / "python"
    )
    assert runtime_commands["process:calendar_api"][0] == str(
        plugin_dir / ".venv" / "bin" / "python"
    )
    assert manager.current_snapshot is not None
    assert manager.current_snapshot.mcp_server_registry is not None
    assert manager.current_snapshot.managed_process_registry is not None
    runtime = manager._composition_generation_host.get(  # pyright: ignore[reportPrivateUsage]
        generation.generation_id
    )
    assert runtime is not None and runtime.processes is not None
    endpoint = runtime.processes.endpoint("calendar_api")
    assert endpoint.port == 18000 and _port_live(endpoint.port)
    tool_registry = manager.current_snapshot.tool_registry
    assert tool_registry is not None
    tool = tool_registry.get_tool("mcp_calendar__get_events")
    assert tool is not None
    assert await tool.execute() == "|".join(
        ("formal", "18000", str(generation.data_dir))
    )

    (plugin_dir / "entry.py").write_text(
        _plugin_source("2"),
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        (plugin_dir / "akashic.plugin.toml")
        .read_text(encoding="utf-8")
        .replace('version = "1"', 'version = "2"')
        .replace('VERSION = "1"', 'VERSION = "2"'),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("calendar")
    assert candidate is not None
    assert candidate.entrypoint == "entry.py"
    assert candidate.static_manifest is not None
    await manager.discard_prepared("calendar")
    await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_unrelated_mcp_reload_keeps_builtin_runtimes_root_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep unchanged builtin runtimes isolated while an MCP plugin reloads."""

    # 1. Start Scheduler, Subagent, and a real MCP plugin in one stable Root.
    calendar_dir = _write_static_manager_plugin(tmp_path, "1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SessionStore(workspace / "sessions.db")
    executions: list[TurnRequest] = []

    async def execute(request: TurnRequest) -> ControlExecutionResult:
        executions.append(request)
        return ControlExecutionResult(response=f"child:{request.input}")

    async def publish(_message: InboundMessage) -> None:
        raise AssertionError("synchronous fixture must not publish a continuation")

    async def deliver(_message: object) -> ChannelDeliveryReceipt:
        raise AssertionError("empty scheduler must not deliver")

    conversation = ConversationRuntime(store, execute)
    builtin_root = Path(__file__).resolve().parents[1] / "plugins"
    manager = PluginManager(
        plugin_dirs=[
            tmp_path / "plugins",
            builtin_root / "scheduler",
            builtin_root / "subagent",
        ],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(validate_semantic_schema=False),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_conversation_runtime(
        conversation,
        programmatic_session_creator=store.create_session,
        programmatic_session_reader=store.get_session_meta,
    )
    manager.bind_continuation_publisher(publish)
    manager.bind_delivery_sender(deliver)
    await manager.load_all()
    lifecycle = asyncio.create_task(manager.run_runtime_services())
    old_snapshot = manager.current_snapshot
    assert old_snapshot is not None and old_snapshot.composition_root is not None
    old_lease = manager.snapshot_store.lease()
    stop_entered = asyncio.Event()
    release_stop = asyncio.Event()

    async def execute_tool(snapshot, lease, name, arguments, turn_id):
        assert snapshot.tool_registry is not None
        token = bind_runtime_snapshot(lease)
        snapshot.tool_registry.set_context(turn_id=turn_id)
        try:
            return await snapshot.tool_registry.execute(
                name,
                arguments,
                raise_errors=True,
            )
        finally:
            reset_runtime_snapshot(token)

    try:
        for _ in range(200):
            if (
                old_snapshot.composition_root.instance_token
                in manager._runtime_started_roots  # pyright: ignore[reportPrivateUsage]
            ):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("old Root runtime did not start")

        old_spawn = await execute_tool(
            old_snapshot,
            old_lease,
            "spawn",
            {"task": "old-root"},
            "parent:old",
        )
        assert "child:old-root" in old_spawn

        # 2. Prepare MCP v2 while a Turn holds the old Root, then publish it.
        _upgrade_static_manager_plugin(calendar_dir, "2")
        candidate = await manager.prepare_candidate("calendar")
        assert candidate is not None and candidate.runtime_snapshot is not None
        assert old_snapshot.lease_count == 1
        await old_lease.release()

        original_stop = manager._stop_runtime_snapshot  # pyright: ignore[reportPrivateUsage]

        async def gated_stop(snapshot) -> None:
            if snapshot is old_snapshot:
                stop_entered.set()
                await release_stop.wait()
            await original_stop(snapshot)

        monkeypatch.setattr(manager, "_stop_runtime_snapshot", gated_stop)
        result = await manager.publish_prepared("calendar")
        assert result["publication_state"] == "committed"
        await asyncio.wait_for(stop_entered.wait(), timeout=5)
        new_snapshot = manager.current_snapshot
        assert new_snapshot is not None and new_snapshot is not old_snapshot
        assert old_snapshot.lease_count == 0
        assert old_snapshot.composition_root is not None
        assert new_snapshot.composition_root is not None

        for plugin_id in ("scheduler", "subagent"):
            assert (
                old_snapshot.generations[plugin_id].instance.module
                is new_snapshot.generations[plugin_id].instance.module
            )
        assert old_snapshot.plugin_tool_catalog is not None
        assert new_snapshot.plugin_tool_catalog is not None
        for tool_name in ("list_schedules", "spawn", "spawn_manage"):
            old_binding = old_snapshot.plugin_tool_catalog[tool_name]
            new_binding = new_snapshot.plugin_tool_catalog[tool_name]
            assert old_binding.handler is not None
            assert new_binding.handler is not None
            assert old_binding.handler is not new_binding.handler
            assert old_binding.is_live()
            assert new_binding.is_live()

        # 3. Drain the old Root; only the new Root may admit later child Turns.
        release_stop.set()
        for _ in range(500):
            if not old_snapshot.plugin_tool_catalog["spawn"].is_live():
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("old Root did not drain")

        new_lease = manager.snapshot_store.lease()
        try:
            new_spawn = await execute_tool(
                new_snapshot,
                new_lease,
                "spawn",
                {"task": "new-root"},
                "parent:new",
            )
            new_schedules = await execute_tool(
                new_snapshot,
                new_lease,
                "list_schedules",
                {},
                "parent:new-list",
            )
        finally:
            await new_lease.release()
        assert "child:new-root" in new_spawn
        assert new_schedules == "当前没有待执行的定时任务"
        assert [request.input for request in executions] == ["old-root", "new-root"]
    finally:
        release_stop.set()
        if old_snapshot.lease_count:
            await old_lease.release()
        lifecycle.cancel()
        _ = await asyncio.gather(lifecycle, return_exceptions=True)
        await manager.terminate_all()
        await conversation.shutdown()
        store.close()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_core_channel_rebind_preserves_live_mcp_tools(tmp_path: Path) -> None:
    _ = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )

    try:
        await manager.load_all()
        await manager.bind_core_channel_definitions((_core_channel_definition(),))

        snapshot = manager.current_snapshot
        assert snapshot is not None and snapshot.tool_registry is not None
        tool = snapshot.tool_registry.get_tool("mcp_calendar__get_events")
        assert tool is not None
        assert await tool.execute() == "|".join(
            (
                "formal",
                "18000",
                str(tmp_path / "workspace/plugin-data/calendar-builtin"),
            )
        )
    finally:
        await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_builtin_direct_formal_failure_recovers_stable_runtime_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-builtin-recovery")
    plugin_dir = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    try:
        await manager.load_all()
        stable = manager.generation("calendar")
        stable_snapshot = manager.current_snapshot
        assert stable is not None and stable_snapshot is not None
        assert _port_live(18000)
        _upgrade_static_manager_plugin(plugin_dir, "2")
        candidate = await manager.prepare_candidate("calendar")
        assert candidate is not None and candidate.reload_tx_id is not None
        original_start = manager._composition_generation_host.start  # pyright: ignore[reportPrivateUsage]
        original_restore = manager._restore_replaced_composition_runtime  # pyright: ignore[reportPrivateUsage]
        formal_failed = False
        restore_failures = 0

        async def fail_candidate_formal_once(*args, **kwargs):
            nonlocal formal_failed
            if kwargs.get("mode") == "formal" and not formal_failed:
                assert manager.current_snapshot is stable_snapshot
                assert not stable_snapshot.accepting_leases
                provisional = manager.latest_snapshot
                assert provisional is not None and provisional is not stable_snapshot
                assert not provisional.accepting_leases
                formal_failed = True
                raise RuntimeError("builtin candidate formal start failed")
            return await original_start(*args, **kwargs)

        async def fail_stable_restore_twice(*args, **kwargs):
            nonlocal restore_failures
            if restore_failures < 2:
                restore_failures += 1
                raise RuntimeError("builtin stable restore failed")
            return await original_restore(*args, **kwargs)

        monkeypatch.setattr(
            manager._composition_generation_host,  # pyright: ignore[reportPrivateUsage]
            "start",
            fail_candidate_formal_once,
        )
        monkeypatch.setattr(
            manager,
            "_restore_replaced_composition_runtime",
            fail_stable_restore_twice,
        )

        with pytest.raises(RuntimeError, match="旧 stable runtime 恢复失败"):
            await manager.publish_prepared("calendar")

        record = manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        )
        assert record.phase == "degraded"
        assert record.recovery_target == "base"
        assert manager.current_snapshot is stable_snapshot
        assert manager.generation("calendar") is stable
        assert not _port_live(18000)

        recovered = await manager.retry_runtime_recovery("calendar")

        assert recovered["publication_state"] == "recovered"
        assert recovered["recovery_target"] == "base"
        assert manager.current_snapshot is stable_snapshot
        assert manager.generation("calendar") is stable
        assert _port_live(18000)
        assert manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        ).phase == "recovered"
    finally:
        await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_installed_runtime_candidate_isolated_and_commit_failure_restores_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-runtime-v1")
    monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    source_root = _write_static_manager_plugin(tmp_path / "source", "1")
    plugin_base = tmp_path / "home" / "cache" / "lab" / "calendar"
    stable_artifact = plugin_base / ".artifacts" / "1.0.0-stable"
    latest_artifact = plugin_base / ".artifacts" / "2.0.0-latest"
    shutil.copytree(source_root, stable_artifact)
    shutil.copytree(source_root, latest_artifact)
    latest_entry = latest_artifact / "entry.py"
    latest_entry.write_text(_plugin_source("2"), encoding="utf-8")
    latest_manifest = latest_artifact / "akashic.plugin.toml"
    latest_manifest.write_text(
        latest_manifest.read_text(encoding="utf-8")
        .replace('version = "1"', 'version = "2"')
        .replace('VERSION = "1"', 'VERSION = "2"'),
        encoding="utf-8",
    )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-stable")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-latest")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"calendar@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    try:
        await manager.load_all()
        stable = manager.generation("calendar@lab")
        stable_snapshot = manager.current_snapshot
        assert stable is not None and stable_snapshot is not None
        assert _port_live(18000)

        write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
        result = (await manager.reconcile_changed())[0]
        candidate = manager.ready_candidate
        assert result.get("publication_state") == "latest_ready", result
        assert candidate is not None and candidate.runtime_snapshot is not None
        assert candidate.reload_tx_id is not None
        journal_record = manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        )
        assert journal_record.runtime_owner_boot_id == "boot-runtime-v1"
        assert journal_record.base_artifact_pointer == stable_pointer.path
        assert journal_record.candidate_artifact_pointer == latest_pointer.path
        candidate_runtime = manager._composition_generation_host.get(  # pyright: ignore[reportPrivateUsage]
            candidate.generation_id
        )
        assert candidate_runtime is not None and candidate_runtime.processes is not None
        candidate_port = candidate_runtime.processes.endpoint("calendar_api").port
        assert candidate_port != 18000 and _port_live(candidate_port)
        candidate_tool = candidate.runtime_snapshot.tool_registry.get_tool(  # type: ignore[union-attr]
            "mcp_calendar__get_events"
        )
        assert candidate_tool is not None
        candidate_output = await candidate_tool.execute()
        assert candidate_output.startswith(f"2|{candidate_port}|")
        assert isinstance(candidate_output, str)
        assert "plugin-validation" in candidate_output
        assert manager.current_snapshot is stable_snapshot
        assert _port_live(18000)

        original_host_start = manager._composition_generation_host.start  # pyright: ignore[reportPrivateUsage]
        original_restore = manager._restore_replaced_composition_runtime  # pyright: ignore[reportPrivateUsage]
        formal_failed = False
        restore_failures = 0

        async def fail_candidate_formal_once(*args, **kwargs):
            nonlocal formal_failed
            if kwargs.get("mode") == "formal" and not formal_failed:
                assert manager.current_snapshot is stable_snapshot
                assert not stable_snapshot.accepting_leases
                provisional = manager.latest_snapshot
                assert provisional is not None and provisional is not stable_snapshot
                assert not provisional.accepting_leases
                formal_failed = True
                raise RuntimeError("candidate formal start failed")
            return await original_host_start(*args, **kwargs)

        async def fail_stable_restore_once(*args, **kwargs):
            nonlocal restore_failures
            if restore_failures < 2:
                restore_failures += 1
                raise RuntimeError("stable runtime restore failed")
            return await original_restore(*args, **kwargs)

        monkeypatch.setattr(
            manager._composition_generation_host,  # pyright: ignore[reportPrivateUsage]
            "start",
            fail_candidate_formal_once,
        )
        monkeypatch.setattr(
            manager,
            "_restore_replaced_composition_runtime",
            fail_stable_restore_once,
        )
        with pytest.raises(RuntimeError, match="candidate formalization 失败"):
            await manager.switch_ready("calendar@lab")

        degraded = manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        )
        assert degraded.phase == "degraded"
        assert degraded.recovery_target == "base"
        assert manager.ready_candidate is not None
        assert not _port_live(18000) and not _port_live(candidate_port)

        recovered = await manager.retry_runtime_recovery("calendar@lab")

        assert recovered["publication_state"] == "recovered"
        assert manager.current_snapshot is stable_snapshot
        assert manager.ready_candidate is None
        assert _port_live(18000)
        assert manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        ).phase == "recovered"

        monkeypatch.undo()
        write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
        result = (await manager.reconcile_changed())[0]
        candidate = manager.ready_candidate
        assert result.get("publication_state") == "latest_ready", result
        assert candidate is not None
        candidate_runtime = manager._composition_generation_host.get(  # pyright: ignore[reportPrivateUsage]
            candidate.generation_id
        )
        assert candidate_runtime is not None and candidate_runtime.processes is not None
        candidate_port = candidate_runtime.processes.endpoint("calendar_api").port

        def fail_owner_commit(*_args: object) -> None:
            raise RuntimeError("runtime owner commit failed")

        monkeypatch.setattr(
            manager,
            "_activate_published_generation",
            fail_owner_commit,
        )
        with pytest.raises(RuntimeError, match="runtime owner commit failed"):
            await manager.switch_ready("calendar@lab")

        assert manager.current_snapshot is stable_snapshot
        assert manager.generation("calendar@lab") is stable
        assert manager.ready_candidate is None
        assert _port_live(18000)
        assert not _port_live(candidate_port)
        stable_tool = stable_snapshot.tool_registry.get_tool(  # type: ignore[union-attr]
            "mcp_calendar__get_events"
        )
        assert stable_tool is not None
        assert await stable_tool.execute() == "|".join(
            ("formal", "18000", str(stable.data_dir))
        )

        monkeypatch.undo()
        write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
        retry_result = (await manager.reconcile_changed())[0]
        retry_candidate = manager.ready_candidate
        assert retry_result.get("publication_state") == "latest_ready", retry_result
        assert retry_candidate is not None
        retry_runtime = manager._composition_generation_host.get(  # pyright: ignore[reportPrivateUsage]
            retry_candidate.generation_id
        )
        assert retry_runtime is not None and retry_runtime.processes is not None
        retry_port = retry_runtime.processes.endpoint("calendar_api").port
        assert retry_port != 18000 and _port_live(retry_port)

        promoted = await manager.switch_ready("calendar@lab")

        current = manager.current_snapshot
        active = manager.generation("calendar@lab")
        assert promoted["publication_state"] == "promoted"
        assert current is not None and current is not stable_snapshot
        assert active is retry_candidate
        assert _port_live(18000) and not _port_live(retry_port)
        promoted_tool = current.tool_registry.get_tool(  # type: ignore[union-attr]
            "mcp_calendar__get_events"
        )
        assert promoted_tool is not None
        assert await promoted_tool.execute() == "|".join(
            ("formal", "18000", str(active.data_dir))
        )
    finally:
        await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_committed_watchdog_failure_is_journaled_and_restartable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-watchdog-failure")
    _ = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    try:
        await manager.load_all()
        generation = manager.generation("calendar")
        assert generation is not None
        process_host = manager._composition_generation_host._process_host  # pyright: ignore[reportPrivateUsage]
        process_host._recovery_backoff_seconds = ()  # pyright: ignore[reportPrivateUsage]
        owned = process_host._generations[generation.generation_id]  # pyright: ignore[reportPrivateUsage]
        process = owned.entries["calendar_api"].process
        assert process is not None
        process.kill()

        action = None
        for _ in range(100):
            actions = manager._reload_journal.pending_recovery()  # pyright: ignore[reportPrivateUsage]
            if actions:
                action = actions[0]
                break
            await asyncio.sleep(0.01)
        assert action is not None
        assert action.phase == "degraded"
        assert action.action == "retry_runtime_recovery"
        assert action.recovery_target == "candidate"
        assert action.runtime_owner_boot_id == "boot-watchdog-failure"
        assert manager._composition_generation_host.failure(  # pyright: ignore[reportPrivateUsage]
            generation.generation_id
        ) is not None

        recovered = await manager.retry_runtime_recovery("calendar")

        assert recovered["recovery_target"] == "candidate"
        assert manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            action.tx_id
        ).phase == "recovered"
        runtime = manager._composition_generation_host.get(  # pyright: ignore[reportPrivateUsage]
            generation.generation_id
        )
        assert runtime is not None and runtime.processes is not None
        assert _port_live(runtime.processes.endpoint("calendar_api").port)
    finally:
        await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_watchdog_failure_shutdown_drains_healthy_sibling_and_keeps_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-watchdog-shutdown")
    _ = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    terminated = False
    try:
        await manager.load_all()
        generation = manager.generation("calendar")
        assert generation is not None
        composition_host = manager._composition_generation_host  # pyright: ignore[reportPrivateUsage]
        process_host = composition_host._process_host  # pyright: ignore[reportPrivateUsage]
        process_host._recovery_backoff_seconds = ()  # pyright: ignore[reportPrivateUsage]
        owned = process_host._generations[generation.generation_id]  # pyright: ignore[reportPrivateUsage]
        process = owned.entries["calendar_api"].process
        assert process is not None
        process.kill()
        action = None
        for _ in range(100):
            actions = manager._reload_journal.pending_recovery()  # pyright: ignore[reportPrivateUsage]
            if actions:
                action = actions[0]
                break
            await asyncio.sleep(0.01)
        assert action is not None and action.recovery_target == "candidate"
        caplog.clear()

        await manager.terminate_all()
        terminated = True

        record = manager._reload_journal.get(action.tx_id)  # pyright: ignore[reportPrivateUsage]
        assert record.phase == "degraded"
        assert record.recovery_target == "candidate"
        assert composition_host._mcp_host.get(generation.generation_id) is None  # pyright: ignore[reportPrivateUsage]
        assert composition_host._mcp_host.tombstone(generation.generation_id) is None  # pyright: ignore[reportPrivateUsage]
        assert composition_host._process_host.get(generation.generation_id) is not None  # pyright: ignore[reportPrivateUsage]
        assert composition_host._process_host.tombstone(generation.generation_id) is not None  # pyright: ignore[reportPrivateUsage]

        recovered = await manager.retry_runtime_recovery("calendar")
        assert "composition-runtime" in str(recovered["retry_receipt"])
        assert manager._reload_journal.get(action.tx_id).phase == "recovered"  # pyright: ignore[reportPrivateUsage]
        assert composition_host.failure(generation.generation_id) is None
        assert not any(
            "observer 已失效" in record.message
            or "health callback failed" in record.message
            for record in caplog.records
        )
    finally:
        if not terminated:
            await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_watchdog_failure_joins_prepared_candidate_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-watchdog-prepared")
    plugin_dir = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    try:
        await manager.load_all()
        stable = manager.generation("calendar")
        stable_snapshot = manager.current_snapshot
        assert stable is not None and stable_snapshot is not None
        _upgrade_static_manager_plugin(plugin_dir, "2")
        candidate = await manager.prepare_candidate("calendar")
        assert candidate is not None and candidate.reload_tx_id is not None
        validation_root = candidate.validation_workspace
        assert validation_root is not None

        process_host = manager._composition_generation_host._process_host  # pyright: ignore[reportPrivateUsage]
        process_host._recovery_backoff_seconds = ()  # pyright: ignore[reportPrivateUsage]
        owned = process_host._generations[stable.generation_id]  # pyright: ignore[reportPrivateUsage]
        process = owned.entries["calendar_api"].process
        assert process is not None
        process.kill()

        action = None
        for _ in range(100):
            actions = manager._reload_journal.pending_recovery()  # pyright: ignore[reportPrivateUsage]
            if actions and actions[0].phase == "degraded":
                action = actions[0]
                break
            await asyncio.sleep(0.01)
        assert action is not None
        assert len(manager._reload_journal.pending_recovery()) == 1  # pyright: ignore[reportPrivateUsage]
        assert action.tx_id == candidate.reload_tx_id
        assert action.generation_id == candidate.generation_id
        assert action.base_generation_id == stable.generation_id
        assert action.recovery_target == "base"
        with pytest.raises(RuntimeError, match="撤销准入"):
            await manager.publish_prepared("calendar")

        recovered = await manager.retry_runtime_recovery("calendar")

        assert recovered["recovery_target"] == "base"
        assert manager.current_snapshot is stable_snapshot
        assert manager.generation("calendar") is stable
        assert candidate.scope.closed
        assert not validation_root.parent.exists()
        assert _port_live(18000)
        assert manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        ).phase == "recovered"
    finally:
        await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_watchdog_failure_revokes_ready_candidate_and_restores_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-watchdog-ready")
    source_root = _write_static_manager_plugin(tmp_path / "source", "1")
    plugin_base = tmp_path / "home" / "cache" / "lab" / "calendar"
    stable_artifact = plugin_base / ".artifacts" / "1.0.0-stable"
    latest_artifact = plugin_base / ".artifacts" / "2.0.0-latest"
    shutil.copytree(source_root, stable_artifact)
    shutil.copytree(source_root, latest_artifact)
    _upgrade_static_manager_plugin(latest_artifact, "2")
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-stable")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-latest")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"calendar@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    try:
        await manager.load_all()
        stable = manager.generation("calendar@lab")
        stable_snapshot = manager.current_snapshot
        assert stable is not None and stable_snapshot is not None
        write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
        result = (await manager.reconcile_changed())[0]
        ready = manager.ready_candidate
        assert result.get("publication_state") == "latest_ready"
        assert ready is not None and ready.reload_tx_id is not None
        candidate_snapshot = ready.runtime_snapshot
        assert candidate_snapshot is not None
        candidate_runtime = manager._composition_generation_host.get(  # pyright: ignore[reportPrivateUsage]
            ready.generation_id
        )
        assert candidate_runtime is not None and candidate_runtime.processes is not None
        candidate_port = candidate_runtime.processes.endpoint("calendar_api").port

        process_host = manager._composition_generation_host._process_host  # pyright: ignore[reportPrivateUsage]
        process_host._recovery_backoff_seconds = ()  # pyright: ignore[reportPrivateUsage]
        owned = process_host._generations[stable.generation_id]  # pyright: ignore[reportPrivateUsage]
        process = owned.entries["calendar_api"].process
        assert process is not None
        process.kill()
        action = None
        for _ in range(100):
            actions = manager._reload_journal.pending_recovery()  # pyright: ignore[reportPrivateUsage]
            if actions and actions[0].phase == "degraded":
                action = actions[0]
                break
            await asyncio.sleep(0.01)
        assert action is not None
        assert len(manager._reload_journal.pending_recovery()) == 1  # pyright: ignore[reportPrivateUsage]
        assert action.tx_id == ready.reload_tx_id
        assert action.recovery_target == "base"
        assert not candidate_snapshot.accepting_leases
        with pytest.raises(RuntimeError, match="撤销准入"):
            await manager.switch_ready("calendar@lab")
        pointers = read_pointers(plugin_base)
        assert pointers is not None
        assert pointers.stable == stable_pointer
        assert pointers.latest == latest_pointer
        manager._reload_journal.advance(  # pyright: ignore[reportPrivateUsage]
            action.tx_id,
            "degraded",
            resource="plugin-skill-projection",
            error="formal skill rollback incomplete",
            recovery_target="base",
        )

        recovered = await manager.retry_runtime_recovery("calendar@lab")

        assert recovered["recovery_target"] == "base"
        assert "stable-skill-projection-restored" in str(
            recovered["retry_receipt"]
        )
        assert manager.current_snapshot is stable_snapshot
        assert manager.generation("calendar@lab") is stable
        assert manager.ready_candidate is None
        assert ready.scope.closed
        assert _port_live(18000)
        assert not _port_live(candidate_port)
        pointers = read_pointers(plugin_base)
        assert pointers is not None
        assert pointers.stable == stable_pointer
        assert pointers.latest == stable_pointer
        assert manager._reload_journal.get(action.tx_id).phase == "recovered"  # pyright: ignore[reportPrivateUsage]
    finally:
        await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_runtime_recovery_finishes_after_resume_even_when_caller_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-recovery-cancel")
    _ = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    release = asyncio.Event()
    entered = asyncio.Event()
    try:
        await manager.load_all()
        generation = manager.generation("calendar")
        assert generation is not None
        process_host = manager._composition_generation_host._process_host  # pyright: ignore[reportPrivateUsage]
        process_host._recovery_backoff_seconds = ()  # pyright: ignore[reportPrivateUsage]
        owned = process_host._generations[generation.generation_id]  # pyright: ignore[reportPrivateUsage]
        process = owned.entries["calendar_api"].process
        assert process is not None
        process.kill()
        action = None
        for _ in range(100):
            actions = manager._reload_journal.pending_recovery()  # pyright: ignore[reportPrivateUsage]
            if actions:
                action = actions[0]
                break
            await asyncio.sleep(0.01)
        assert action is not None

        async def blocking_resumer() -> None:
            entered.set()
            await release.wait()

        manager._endpoint_resumer = blocking_resumer  # pyright: ignore[reportPrivateUsage]
        retry = asyncio.create_task(manager.retry_runtime_recovery("calendar"))
        await entered.wait()
        assert manager._reload_journal.get(action.tx_id).phase == "degraded"  # pyright: ignore[reportPrivateUsage]
        retry.cancel()
        await asyncio.sleep(0)
        assert not retry.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await retry

        assert manager._reload_journal.get(action.tx_id).phase == "recovered"  # pyright: ignore[reportPrivateUsage]
        assert manager.current_snapshot is not None
        assert manager.current_snapshot.accepting_leases
    finally:
        release.set()
        await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_runtime_recovery_finishes_host_retry_before_exposing_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-host-retry-cancel")
    _ = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    release = asyncio.Event()
    entered = asyncio.Event()
    try:
        await manager.load_all()
        generation = manager.generation("calendar")
        assert generation is not None
        composition_host = manager._composition_generation_host  # pyright: ignore[reportPrivateUsage]
        process_host = composition_host._process_host  # pyright: ignore[reportPrivateUsage]
        process_host._recovery_backoff_seconds = ()  # pyright: ignore[reportPrivateUsage]
        owned = process_host._generations[generation.generation_id]  # pyright: ignore[reportPrivateUsage]
        process = owned.entries["calendar_api"].process
        assert process is not None
        process.kill()
        action = None
        for _ in range(100):
            actions = manager._reload_journal.pending_recovery()  # pyright: ignore[reportPrivateUsage]
            if actions:
                action = actions[0]
                break
            await asyncio.sleep(0.01)
        assert action is not None
        original_retry = composition_host.retry_runtime_recovery

        async def blocking_retry(generation_id: str) -> str:
            entered.set()
            await release.wait()
            return await original_retry(generation_id)

        monkeypatch.setattr(composition_host, "retry_runtime_recovery", blocking_retry)
        retry = asyncio.create_task(manager.retry_runtime_recovery("calendar"))
        await entered.wait()
        retry.cancel()
        await asyncio.sleep(0)
        assert not retry.done()
        assert manager._reload_journal.get(action.tx_id).phase == "degraded"  # pyright: ignore[reportPrivateUsage]

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await retry

        assert manager._reload_journal.get(action.tx_id).phase == "recovered"  # pyright: ignore[reportPrivateUsage]
        runtime = composition_host.get(generation.generation_id)
        assert runtime is not None and runtime.processes is not None
        assert _port_live(runtime.processes.endpoint("calendar_api").port)
    finally:
        release.set()
        await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_candidate_start_cleanup_failure_keeps_durable_retry_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-candidate-start-failure")
    plugin_dir = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    original_terminate = OwnedProcessGroup.terminate

    async def fail_runtime_cleanup(
        _group: OwnedProcessGroup,
        *,
        timeout_s: float,
    ) -> None:
        _ = timeout_s
        raise RuntimeError("injected candidate start cleanup failure")

    try:
        await manager.load_all()
        stable_snapshot = manager.current_snapshot
        _upgrade_static_manager_plugin(plugin_dir, "2")
        candidate = await manager.prepare_candidate("calendar")
        assert candidate is not None and candidate.reload_tx_id is not None
        (plugin_dir / "mcp.py").write_text(
            "raise SystemExit(23)\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            OwnedProcessGroup,
            "terminate",
            fail_runtime_cleanup,
        )

        with pytest.raises(RuntimeError, match="runtime cleanup 未完成"):
            await manager.publish_prepared("calendar")

        record = manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        )
        assert record.phase == "cleanup_failed"
        assert record.recovery_action == "retry_generation_cleanup"
        assert record.recovery_target == "base"
        assert manager.current_snapshot is stable_snapshot
        assert manager._composition_generation_host.failure(  # pyright: ignore[reportPrivateUsage]
            candidate.generation_id
        ) is not None
        retained = manager._composition_generation_host.get(  # pyright: ignore[reportPrivateUsage]
            candidate.generation_id
        )
        assert retained is not None and retained.processes is not None
        retained_port = retained.processes.endpoint("calendar_api").port
        assert _port_live(retained_port)

        monkeypatch.setattr(
            OwnedProcessGroup,
            "terminate",
            original_terminate,
        )
        recovered = await manager.retry_runtime_recovery("calendar")

        assert recovered["recovery_target"] == "base"
        assert manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        ).phase == "aborted"
        assert not _port_live(retained_port)
    finally:
        monkeypatch.setattr(
            OwnedProcessGroup,
            "terminate",
            original_terminate,
        )
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_stable_boot_cleanup_failure_creates_durable_retry_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-stable-start-failure")
    plugin_dir = _write_static_manager_plugin(tmp_path, "1")
    (plugin_dir / "mcp.py").write_text(
        "raise SystemExit(24)\n",
        encoding="utf-8",
    )
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    original_terminate = OwnedProcessGroup.terminate

    async def fail_runtime_cleanup(
        _group: OwnedProcessGroup,
        *,
        timeout_s: float,
    ) -> None:
        _ = timeout_s
        raise RuntimeError("injected stable boot cleanup failure")

    monkeypatch.setattr(OwnedProcessGroup, "terminate", fail_runtime_cleanup)
    try:
        with pytest.raises(RuntimeError, match="cleanup"):
            await manager.load_all()

        actions = manager._reload_journal.pending_recovery()  # pyright: ignore[reportPrivateUsage]
        assert len(actions) == 1
        action = actions[0]
        assert action.plugin_id == "calendar"
        assert action.phase == "cleanup_failed"
        assert action.action == "retry_generation_cleanup"
        assert action.recovery_target == "base"
        assert action.runtime_owner_boot_id == "boot-stable-start-failure"
        retained = manager._composition_generation_host.get(  # pyright: ignore[reportPrivateUsage]
            action.generation_id
        )
        assert retained is not None and retained.processes is not None
        retained_port = retained.processes.endpoint("calendar_api").port
        assert _port_live(retained_port)

        monkeypatch.setattr(
            OwnedProcessGroup,
            "terminate",
            original_terminate,
        )
        recovered = await manager.retry_runtime_recovery("calendar")

        assert recovered["recovery_target"] == "base"
        assert manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            action.tx_id
        ).phase == "aborted"
        assert not _port_live(retained_port)
    finally:
        monkeypatch.setattr(
            OwnedProcessGroup,
            "terminate",
            original_terminate,
        )
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_cleanup_failure_requires_host_retry_before_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-cleanup-retry")
    monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    source_root = _write_static_manager_plugin(tmp_path / "source", "1")
    plugin_base = tmp_path / "home" / "cache" / "lab" / "calendar"
    stable_artifact = plugin_base / ".artifacts" / "1.0.0-stable"
    candidate_artifact = plugin_base / ".artifacts" / "2.0.0-latest"
    shutil.copytree(source_root, stable_artifact)
    shutil.copytree(source_root, candidate_artifact)
    (candidate_artifact / "entry.py").write_text(
        _plugin_source("2"),
        encoding="utf-8",
    )
    candidate_manifest = candidate_artifact / "akashic.plugin.toml"
    candidate_manifest.write_text(
        candidate_manifest.read_text(encoding="utf-8")
        .replace('version = "1"', 'version = "2"')
        .replace('VERSION = "1"', 'VERSION = "2"'),
        encoding="utf-8",
    )
    base_pointer = ArtifactPointer(".artifacts/1.0.0-stable")
    candidate_pointer = ArtifactPointer(".artifacts/2.0.0-latest")
    write_pointers(plugin_base, stable=base_pointer, latest=base_pointer)
    write_plugin_manifest(
        {"calendar@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    original_terminate = OwnedProcessGroup.terminate
    terminate_calls = 0

    async def fail_terminate_once(
        group: OwnedProcessGroup,
        *,
        timeout_s: float,
    ) -> None:
        nonlocal terminate_calls
        terminate_calls += 1
        if terminate_calls == 2:
            raise RuntimeError("injected process-group cleanup failure")
        await original_terminate(group, timeout_s=timeout_s)

    try:
        await manager.load_all()
        stable_snapshot = manager.current_snapshot
        assert stable_snapshot is not None and _port_live(18000)
        write_pointers(
            plugin_base,
            stable=base_pointer,
            latest=candidate_pointer,
        )
        result = (await manager.reconcile_changed())[0]
        candidate = manager.ready_candidate
        assert result.get("publication_state") == "latest_ready"
        assert candidate is not None and candidate.reload_tx_id is not None
        runtime = manager._composition_generation_host.get(  # pyright: ignore[reportPrivateUsage]
            candidate.generation_id
        )
        assert runtime is not None and runtime.processes is not None
        candidate_port = runtime.processes.endpoint("calendar_api").port
        monkeypatch.setattr(OwnedProcessGroup, "terminate", fail_terminate_once)

        with pytest.raises(RuntimeError, match="runtime cleanup 未完成"):
            await manager.drop_candidate("calendar@lab")

        record = manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        )
        assert record.phase == "cleanup_failed"
        assert record.recovery_target == "base"
        assert manager.ready_candidate is candidate
        assert manager.current_snapshot is stable_snapshot
        assert _port_live(18000)
        assert manager._composition_generation_host.failure(  # pyright: ignore[reportPrivateUsage]
            candidate.generation_id
        ) is not None

        recovered = await manager.retry_runtime_recovery("calendar@lab")

        assert recovered["publication_state"] == "recovered"
        assert recovered["recovery_target"] == "base"
        assert manager.ready_candidate is None
        assert manager.current_snapshot is stable_snapshot
        assert _port_live(18000) and not _port_live(candidate_port)
        assert manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        ).phase == "aborted"
    finally:
        monkeypatch.setattr(OwnedProcessGroup, "terminate", original_terminate)
        await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_installed_stable_boot_cleanup_failure_targets_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-installed-stable-failure")
    source_root = _write_static_manager_plugin(tmp_path / "source", "1")
    plugin_base = tmp_path / "home" / "cache" / "lab" / "calendar"
    stable_artifact = plugin_base / ".artifacts" / "1.0.0-stable"
    shutil.copytree(source_root, stable_artifact)
    (stable_artifact / "mcp.py").write_text(
        "raise SystemExit(24)\n",
        encoding="utf-8",
    )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-stable")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"calendar@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    original_terminate = OwnedProcessGroup.terminate

    async def fail_runtime_cleanup(
        _group: OwnedProcessGroup,
        *,
        timeout_s: float,
    ) -> None:
        _ = timeout_s
        raise RuntimeError("injected installed stable cleanup failure")

    monkeypatch.setattr(OwnedProcessGroup, "terminate", fail_runtime_cleanup)
    try:
        with pytest.raises(RuntimeError, match="cleanup"):
            await manager.load_all()
        action = manager._reload_journal.pending_recovery()[0]  # pyright: ignore[reportPrivateUsage]
        assert action.action == "retry_generation_cleanup"
        assert action.recovery_target == "base"
        assert action.base_artifact_pointer == stable_pointer.path
        assert action.candidate_artifact_pointer is None
        assert manager.current_snapshot is None

        monkeypatch.setattr(OwnedProcessGroup, "terminate", original_terminate)
        recovered = await manager.retry_runtime_recovery("calendar@lab")

        assert recovered["recovery_target"] == "base"
        assert recovered["generation_id"] is None
        assert recovered["snapshot_id"] is None
        assert manager._reload_journal.get(action.tx_id).phase == "aborted"  # pyright: ignore[reportPrivateUsage]
        pointers = read_pointers(plugin_base)
        assert pointers is not None
        assert pointers.stable == stable_pointer
        assert pointers.latest == stable_pointer
    finally:
        monkeypatch.setattr(OwnedProcessGroup, "terminate", original_terminate)
        await manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_supervised_boot_reconciles_degraded_runtime_to_exact_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-runtime-old")
    monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    source_root = _write_static_manager_plugin(tmp_path / "source", "1")
    plugin_base = tmp_path / "home" / "cache" / "lab" / "calendar"
    stable_artifact = plugin_base / ".artifacts" / "1.0.0-stable"
    latest_artifact = plugin_base / ".artifacts" / "2.0.0-latest"
    shutil.copytree(source_root, stable_artifact)
    shutil.copytree(source_root, latest_artifact)
    (latest_artifact / "entry.py").write_text(
        _plugin_source("2"),
        encoding="utf-8",
    )
    latest_manifest = latest_artifact / "akashic.plugin.toml"
    latest_manifest.write_text(
        latest_manifest.read_text(encoding="utf-8")
        .replace('version = "1"', 'version = "2"')
        .replace('VERSION = "1"', 'VERSION = "2"'),
        encoding="utf-8",
    )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-stable")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-latest")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"calendar@lab": True},
        plugins_home=tmp_path / "home",
    )
    old_manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    new_manager: PluginManager | None = None
    orphan: asyncio.subprocess.Process | None = None
    try:
        await old_manager.load_all()
        write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
        _ = await old_manager.reconcile_changed()
        candidate = old_manager.ready_candidate
        assert candidate is not None and candidate.reload_tx_id is not None

        original_start = old_manager._composition_generation_host.start  # pyright: ignore[reportPrivateUsage]
        formal_failed = False

        async def fail_formal_once(*args, **kwargs):
            nonlocal formal_failed
            if kwargs.get("mode") == "formal" and not formal_failed:
                formal_failed = True
                raise RuntimeError("formal start failed before pointer commit")
            return await original_start(*args, **kwargs)

        async def fail_restore(*_args, **_kwargs):
            raise RuntimeError("old runtime restore remains uncertain")

        monkeypatch.setattr(
            old_manager._composition_generation_host,  # pyright: ignore[reportPrivateUsage]
            "start",
            fail_formal_once,
        )
        monkeypatch.setattr(
            old_manager,
            "_restore_replaced_composition_runtime",
            fail_restore,
        )
        with pytest.raises(RuntimeError, match="candidate formalization 失败"):
            await old_manager.switch_ready("calendar@lab")
        action = old_manager._reload_journal.pending_recovery()[0]  # pyright: ignore[reportPrivateUsage]
        assert action.phase == "degraded"
        assert action.recovery_target == "base"
        assert action.runtime_owner_boot_id == "boot-runtime-old"

        orphan_env = dict(os.environ)
        orphan_env["AKASHIC_BOOT_ID"] = "boot-runtime-old"
        orphan = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            env=orphan_env,
            start_new_session=True,
        )
        assert orphan.returncode is None

        monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-runtime-new")
        new_manager = _manager(
            tmp_path,
            tool_registry=ToolRegistry(follow_runtime_snapshot=False),
        )
        await new_manager.load_all()

        pointers = read_pointers(plugin_base)
        assert pointers is not None
        assert pointers.stable == stable_pointer
        assert pointers.latest == stable_pointer
        assert await asyncio.wait_for(orphan.wait(), timeout=2) != 0
        stable = new_manager.generation("calendar@lab")
        snapshot = new_manager.current_snapshot
        assert stable is not None and snapshot is not None
        assert stable.plugin_dir == stable_artifact
        assert _port_live(18000)
        tool = snapshot.tool_registry.get_tool(  # type: ignore[union-attr]
            "mcp_calendar__get_events"
        )
        assert tool is not None
        assert await tool.execute() == "|".join(
            ("formal", "18000", str(stable.data_dir))
        )
        assert new_manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            action.tx_id
        ).phase == "recovered"
    finally:
        if orphan is not None and orphan.returncode is None:
            orphan.kill()
            _ = await orphan.wait()
        if new_manager is not None:
            await new_manager.terminate_all()
        await old_manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_supervised_boot_rebuilds_exact_candidate_after_pointer_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-candidate-old")
    monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    source_root = _write_static_manager_plugin(tmp_path / "source", "1")
    plugin_base = tmp_path / "home" / "cache" / "lab" / "calendar"
    stable_artifact = plugin_base / ".artifacts" / "1.0.0-stable"
    candidate_artifact = plugin_base / ".artifacts" / "2.0.0-latest"
    shutil.copytree(source_root, stable_artifact)
    shutil.copytree(source_root, candidate_artifact)
    (candidate_artifact / "entry.py").write_text(
        _plugin_source("2"),
        encoding="utf-8",
    )
    candidate_manifest = candidate_artifact / "akashic.plugin.toml"
    candidate_manifest.write_text(
        candidate_manifest.read_text(encoding="utf-8")
        .replace('version = "1"', 'version = "2"')
        .replace('VERSION = "1"', 'VERSION = "2"'),
        encoding="utf-8",
    )
    base_pointer = ArtifactPointer(".artifacts/1.0.0-stable")
    candidate_pointer = ArtifactPointer(".artifacts/2.0.0-latest")
    write_pointers(plugin_base, stable=base_pointer, latest=base_pointer)
    write_plugin_manifest(
        {"calendar@lab": True},
        plugins_home=tmp_path / "home",
    )
    old_manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    new_manager: PluginManager | None = None
    orphan: asyncio.subprocess.Process | None = None
    old_manager_closed = False
    try:
        await old_manager.load_all()
        write_pointers(
            plugin_base,
            stable=base_pointer,
            latest=candidate_pointer,
        )
        _ = await old_manager.reconcile_changed()
        candidate = old_manager.ready_candidate
        assert candidate is not None and candidate.reload_tx_id is not None
        old_manager._advance_reload(candidate, "promoting")  # pyright: ignore[reportPrivateUsage]
        write_pointers(
            plugin_base,
            stable=candidate_pointer,
            latest=candidate_pointer,
        )
        old_manager._advance_reload(  # pyright: ignore[reportPrivateUsage]
            candidate,
            "degraded",
            error="process crashed after pointer commit",
            resource=f"composition-runtime:{candidate.generation_id}",
            formal_effects=("candidate_pointer_committed",),
            recovery_action="retry_runtime_recovery",
            recovery_target="candidate",
        )

        # A real new boot cannot coexist with the old Gateway event loop.
        await old_manager.terminate_all()
        old_manager_closed = True
        orphan_env = dict(os.environ)
        orphan_env["AKASHIC_BOOT_ID"] = "boot-candidate-old"
        orphan = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            env=orphan_env,
            start_new_session=True,
        )
        assert orphan.returncode is None

        monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-candidate-new")
        new_manager = _manager(
            tmp_path,
            tool_registry=ToolRegistry(follow_runtime_snapshot=False),
        )
        await new_manager.load_all()

        pointers = read_pointers(plugin_base)
        assert pointers is not None
        assert pointers.stable == candidate_pointer
        assert pointers.latest == candidate_pointer
        stable = new_manager.generation("calendar@lab")
        snapshot = new_manager.current_snapshot
        assert stable is not None and snapshot is not None
        assert stable.plugin_dir == candidate_artifact
        assert stable.source_revision == candidate.source_revision
        tool = snapshot.tool_registry.get_tool(  # type: ignore[union-attr]
            "mcp_calendar__get_events"
        )
        assert tool is not None
        assert await tool.execute() == "|".join(
            ("formal", "18000", str(stable.data_dir))
        )
        assert new_manager._reload_journal.get(  # pyright: ignore[reportPrivateUsage]
            candidate.reload_tx_id
        ).phase == "recovered"
        assert await asyncio.wait_for(orphan.wait(), timeout=2) != 0
    finally:
        if orphan is not None and orphan.returncode is None:
            orphan.kill()
            _ = await orphan.wait()
        if new_manager is not None:
            await new_manager.terminate_all()
        if not old_manager_closed:
            await old_manager.terminate_all()
    assert not _port_live(18000)


@pytest.mark.parametrize(
    ("supervised", "current_boot", "recorded_boot", "message"),
    (
        (False, "boot-new", "boot-old", "supervised boot identity"),
        (True, "boot-same", "boot-same", "不同于当前进程"),
        (True, "boot-new", None, "旧 boot identity"),
    ),
)
@pytest.mark.asyncio
async def test_runtime_recovery_rejects_unowned_boot_cleanup_without_pointer_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supervised: bool,
    current_boot: str,
    recorded_boot: str | None,
    message: str,
) -> None:
    monkeypatch.setenv("AKASHIC_BOOT_ID", current_boot)
    if supervised:
        monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    else:
        monkeypatch.delenv("AKASHIC_SUPERVISED", raising=False)
    source_root = _write_static_manager_plugin(tmp_path / "source", "1")
    plugin_base = tmp_path / "home" / "cache" / "lab" / "calendar"
    stable_artifact = plugin_base / ".artifacts" / "1.0.0-stable"
    candidate_artifact = plugin_base / ".artifacts" / "2.0.0-latest"
    shutil.copytree(source_root, stable_artifact)
    shutil.copytree(source_root, candidate_artifact)
    base_pointer = ArtifactPointer(".artifacts/1.0.0-stable")
    candidate_pointer = ArtifactPointer(".artifacts/2.0.0-latest")
    write_pointers(
        plugin_base,
        stable=base_pointer,
        latest=candidate_pointer,
    )
    write_plugin_manifest(
        {"calendar@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = _manager(
        tmp_path,
        tool_registry=ToolRegistry(follow_runtime_snapshot=False),
    )
    tx_id = manager._reload_journal.begin(  # pyright: ignore[reportPrivateUsage]
        plugin_id="calendar@lab",
        base_snapshot_id="stable-v1",
        base_generation_id="calendar:stable:1",
        generation_id="calendar:candidate:2",
        source_revision="source-v2",
        config_revision="config-v2",
        base_artifact_pointer=base_pointer.path,
        candidate_artifact_pointer=candidate_pointer.path,
    )
    if recorded_boot is not None:
        manager._reload_journal.mark_runtime_owner(  # pyright: ignore[reportPrivateUsage]
            tx_id,
            recorded_boot,
        )
    manager._reload_journal.advance(  # pyright: ignore[reportPrivateUsage]
        tx_id,
        "degraded",
        resource="composition-runtime:calendar:candidate:2",
        error="runtime owner uncertain",
        recovery_target="base",
    )

    with pytest.raises(RuntimeError, match=message):
        await manager.load_all()

    pointers = read_pointers(plugin_base)
    assert pointers is not None
    assert pointers.stable == base_pointer
    assert pointers.latest == candidate_pointer
    assert manager.current_snapshot is None
    assert manager.generation("calendar@lab") is None
    assert not _port_live(18000)


@pytest.mark.asyncio
async def test_static_candidate_without_formal_data_leaves_no_formal_directory(
    tmp_path: Path,
) -> None:
    _ = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(tmp_path)
    formal_data = tmp_path / "workspace" / "plugin-data" / "calendar-builtin"

    candidate = await manager.prepare_candidate("calendar")

    assert candidate is not None
    assert candidate.validation_workspace is not None
    assert candidate.runtime_snapshot is not None
    candidate_root = candidate.runtime_snapshot.composition_root
    assert candidate_root is not None
    runtime = candidate_root.root_fiber.children[0].runtime
    assert runtime is not None and runtime.data_dir.is_dir()
    assert candidate.validation_data_inventory == ()
    assert not formal_data.exists()

    await manager.discard_prepared("calendar")
    assert not formal_data.exists()
    assert not candidate.validation_workspace.parent.exists()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_static_first_publication_rolls_back_new_formal_data_on_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(tmp_path)
    formal_data = tmp_path / "workspace" / "plugin-data" / "calendar-builtin"
    candidate = await manager.prepare_candidate("calendar")
    assert candidate is not None

    def fail_owner_commit(*_args: object) -> None:
        raise RuntimeError("owner commit failed")

    monkeypatch.setattr(
        manager,
        "_activate_published_generation",
        fail_owner_commit,
    )
    with pytest.raises(RuntimeError, match="owner commit failed"):
        await manager.publish_prepared("calendar")

    assert manager.current_snapshot is None
    assert not formal_data.exists()
    assert candidate.scope.closed is True

    monkeypatch.undo()
    replacement = await manager.prepare_candidate("calendar")
    assert replacement is not None
    result = await manager.publish_prepared("calendar")
    assert result["publication_state"] == "committed"
    assert formal_data.is_dir()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_static_manifest_declarations_do_not_activate_inactive_plugin(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_static_manager_plugin(tmp_path, "1")
    entrypoint = plugin_dir / "entry.py"
    entrypoint.write_text(
        entrypoint.read_text(encoding="utf-8")
        + "\ndef is_active(services):\n"
        + "    return False\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is not None
    assert snapshot.composition_active_plugin_ids == frozenset()
    assert snapshot.mcp_server_registry is not None
    assert len(snapshot.mcp_server_registry) == 0
    assert snapshot.managed_process_registry is not None
    assert len(snapshot.managed_process_registry) == 0
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ('required_tools = ["get_events"]', 'required_tools = ["other"]', "MCP 声明"),
        ("formal_port = 18000", "formal_port = 18001", "managed process 声明"),
    ),
)
async def test_static_manifest_runtime_drift_excludes_failed_stable_plugin(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    plugin_dir = _write_static_manager_plugin(tmp_path, "1")
    manifest = plugin_dir / "akashic.plugin.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.current_snapshot is None
    assert manager._active_generations == {}  # pyright: ignore[reportPrivateUsage]
    gate = manager.latest_gate("calendar")
    assert gate is not None and gate.status == "failed"
    assert message in str(gate.checks[-1].evidence)
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_manager_keeps_candidate_mcp_registry_private_until_publish(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None and stable.mcp_server_registry is not None
    stable_registry = stable.mcp_server_registry
    assert stable_registry["calendar"].definition.candidate_env["VERSION"] == "1"

    _upgrade_static_manager_plugin(plugin_dir, "2")
    candidate = await manager.prepare_candidate("calendar")
    assert candidate is not None and candidate.runtime_snapshot is not None
    candidate_registry = candidate.runtime_snapshot.mcp_server_registry
    assert candidate_registry is not None
    assert candidate_registry["calendar"].definition.candidate_env["VERSION"] == "2"
    assert manager.current_snapshot is stable
    assert manager.current_snapshot.mcp_server_registry is stable_registry

    result = await manager.publish_prepared("calendar")
    assert result["publication_state"] == "committed"
    current = manager.current_snapshot
    assert current is not None and current.mcp_server_registry is not None
    assert current.mcp_server_registry is not candidate_registry
    assert current.mcp_server_registry["calendar"].definition.candidate_env["VERSION"] == "2"
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_manager_rejects_mcp_registry_drift_before_publish(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_static_manager_plugin(tmp_path, "1")
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None and stable.mcp_server_registry is not None
    stable_registry = stable.mcp_server_registry

    _upgrade_static_manager_plugin(plugin_dir, "2")
    candidate = await manager.prepare_candidate("calendar")
    assert candidate is not None and candidate.runtime_snapshot is not None

    def replace_with_stable_registry(snapshot: RuntimeSnapshot) -> None:
        snapshot.mcp_server_registry = stable_registry
        snapshot.mcp_server_registry_identity = stable_registry.identity

    async def release_validation(_snapshot: RuntimeSnapshot) -> None:
        return None

    manager.bind_dashboard_preparer(
        replace_with_stable_registry,
        validation_releaser=release_validation,
    )
    with pytest.raises(RuntimeError, match="MCP registry"):
        await manager.publish_prepared("calendar")
    assert manager.current_snapshot is stable
    assert manager.current_snapshot.mcp_server_registry is stable_registry

    await manager.discard_prepared("calendar")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_manager_rejects_mcp_endpoint_without_same_owner_process(
    tmp_path: Path,
) -> None:
    plugin_dir = _plugin_dir(tmp_path / "plugins")
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import (\n"
        "    MCP_SERVERS, EndpointEnv, McpServerDefinition,\n"
        ")\n"
        "api_version = 3\n"
        "name = 'calendar'\n"
        "version = '1'\n"
        "inject = (MCP_SERVERS,)\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(MCP_SERVERS).register(\n"
        "        ctx, McpServerDefinition(\n"
        "            name='calendar', command=('python', 'mcp.py'),\n"
        "            endpoint_env=(EndpointEnv('PORT', 'calendar_api'),),\n"
        "        ),\n"
        "    )\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.generation("calendar") is None
    gate = manager.latest_gate("calendar")
    assert gate is not None and gate.status == "failed"
    assert "缺少同 owner managed process" in gate.failure_reason
    await manager.terminate_all()

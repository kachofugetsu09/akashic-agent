from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import pytest

from agent.plugin_composition import (
    MANAGED_PROCESSES,
    MCP_SERVERS,
    CompositionRoot,
    EndpointEnv,
    ManagedProcessDefinition,
    McpServerDefinition,
    PluginRuntime,
)
from agent.plugin_composition.mcp_slots import PluginMcpServers
from agent.plugin_composition.process_slots import PluginManagedProcesses
from agent.plugins.composition_generation_host import CompositionGenerationHost
from agent.plugins.generation import GateResult, PluginContributions, PluginGeneration
from agent.plugins.scope import PluginScope
from agent.plugins.snapshot import RuntimeSnapshotCompiler
from agent.tools.registry import ToolRegistry


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _port_live(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _write_http_server(path: Path) -> None:
    path.write_text(
        "import os\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'ready')\n"
        "    def log_message(self, *_args): pass\n"
        "HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()\n",
        encoding="utf-8",
    )


def _write_mcp_server(path: Path) -> None:
    path.write_text(
        "import json, os, sys\n"
        "for raw in sys.stdin:\n"
        "    msg = json.loads(raw); method = msg.get('method')\n"
        "    if method == 'initialize': result = {'protocolVersion': '2025-11-25'}\n"
        "    elif method == 'tools/list': result = {'tools': [{'name': 'read', "
        "'description': 'read env', 'inputSchema': {'type': 'object'}}]}\n"
        "    elif method == 'tools/call':\n"
        "        result = {'content': [{'type': 'text', 'text': '|'.join((\n"
        "            os.environ['ROLE'], os.environ['PORT'],\n"
        "            os.environ['AKA_PLUGIN_DATA_DIR'], os.environ['AKASHIC_WORKSPACE'],\n"
        "        ))}]}\n"
        "    else: continue\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_exact_root_candidate_materializes_process_mcp_and_tool_route(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "calendar"
    plugin_dir.mkdir()
    process_script = plugin_dir / "process.py"
    mcp_script = plugin_dir / "mcp.py"
    _write_http_server(process_script)
    _write_mcp_server(mcp_script)
    data_dir = tmp_path / "validation-data"
    workspace = tmp_path / "validation-workspace"
    data_dir.mkdir()
    workspace.mkdir()

    root = CompositionRoot("runtime-host")
    process_declarations = PluginManagedProcesses(root.instance_token)
    mcp_declarations = PluginMcpServers(root.instance_token)
    _ = await root.context.provide(MANAGED_PROCESSES, process_declarations)
    _ = await root.context.provide(MCP_SERVERS, mcp_declarations)
    formal_port = _free_port()

    async def apply(ctx) -> None:
        await ctx.require(MANAGED_PROCESSES).register(
            ctx,
            ManagedProcessDefinition(
                name="calendar_api",
                command=("python", "process.py"),
                cwd=".",
                formal_port=formal_port,
                readiness_path="/health",
            ),
        )
        await ctx.require(MCP_SERVERS).register(
            ctx,
            McpServerDefinition(
                name="calendar",
                command=("python", "mcp.py"),
                cwd=".",
                required_tools=("read",),
                candidate_read_only_tools=("read",),
                endpoint_env=(EndpointEnv("PORT", "calendar_api"),),
                candidate_env={"ROLE": "recording"},
            ),
        )

    _ = await root.mount(
        apply,
        name="calendar",
        inject=(MANAGED_PROCESSES, MCP_SERVERS),
        runtime=PluginRuntime(
            plugin_id="calendar",
            generation_id="test-generation",
            plugin_dir=plugin_dir,
            data_dir=data_dir,
            workspace=workspace,
            config=None,
        ),
    )
    generation = PluginGeneration(
        plugin_id="calendar",
        generation_id="calendar:test",
        module_path="plugins.calendar",
        source_revision="source",
        config_revision="config",
        plugin_dir=plugin_dir,
        data_dir=data_dir,
        config=None,
        instance=object(),
        scope=PluginScope("calendar"),
        contributions=PluginContributions(manifest={}),
        gate_result=GateResult(
            gate_id="gate",
            plugin_id="calendar",
            candidate_revision="source",
            status="passed",
            checks=(),
        ),
        static_runtime_commands=(
            ("mcp:calendar", (sys.executable, str(mcp_script))),
            ("process:calendar_api", (sys.executable, str(process_script))),
        ),
    )
    snapshot = RuntimeSnapshotCompiler().compile(
        {generation.plugin_id: generation},
        composition_root=root,
    )
    snapshot.tool_registry = ToolRegistry(follow_runtime_snapshot=False)
    host = CompositionGenerationHost()

    try:
        runtime = await host.start(
            generation,
            snapshot,
            mode="candidate",
        )
        assert runtime is not None and runtime.processes is not None
        endpoint = runtime.processes.endpoint("calendar_api")
        assert endpoint.port != formal_port
        assert _port_live(endpoint.port)
        assert root.receipt().ready

        registry = host.attach_tools(snapshot.tool_registry, runtime)
        assert registry is snapshot.tool_registry and registry is not None
        tool = registry.get_tool("mcp_calendar__read")
        assert tool is not None
        output = await tool.execute()
        assert output == "|".join(
            ("recording", str(endpoint.port), str(data_dir), str(workspace))
        )
    finally:
        await host.stop(generation.generation_id)
        await root.dispose()
        _ = await generation.scope.aclose()

    assert not _port_live(formal_port)
    assert not any(
        task.get_name().startswith(("mcp_generation_", "managed_process_"))
        for task in asyncio.all_tasks()
        if not task.done()
    )

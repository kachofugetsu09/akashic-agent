from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

import agent.mcp.client as client_module
import agent.plugins.mcp_generation_host as mcp_host_module
from agent.plugin_composition import (
    MCP_SERVERS,
    CompositionRoot,
    EndpointEnv,
    McpServerDefinition,
    PluginRuntime,
)
from agent.plugin_composition.mcp_slots import (
    McpServerRegistry,
    PluginMcpServers,
    _freeze_plugin_mcp_servers,
)
from agent.plugins.mcp_generation_host import (
    McpGeneration,
    McpGenerationHost,
    McpMaterializedCommand,
)
from utils.process_group import OwnedProcessGroup


def test_incident_error_text_has_no_blank_timeout_message() -> None:
    assert mcp_host_module._error_text(TimeoutError()) == "TimeoutError"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _runtime(plugin_dir: Path) -> PluginRuntime:
    return PluginRuntime(
        plugin_id=plugin_dir.name,
        generation_id="test-generation",
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        workspace=plugin_dir / "workspace",
        config=None,
    )


def _write_server(path: Path, *, mode: str = "normal") -> None:
    if mode == "always_exit":
        exit_code = "    if method == 'tools/list':\n        time.sleep(0.01); raise SystemExit(17)\n"
    elif mode == "first_exit":
        exit_code = (
            "    if method == 'tools/list' and epoch == 1:\n"
            "        time.sleep(0.01); raise SystemExit(17)\n"
        )
    elif mode == "hang_initialize":
        exit_code = "    if method == 'initialize':\n        time.sleep(30)\n"
    else:
        exit_code = ""
    catalog = (
        "            {'name': 'read_tool', 'description': 'changed', "
        "'inputSchema': {'type': 'object', 'properties': {'changed': {'type': 'boolean'}}}},\n"
        if mode == "catalog_drift"
        else "            {'name': 'read_tool', 'description': 'read', 'inputSchema': {'type': 'object'}},\n"
    )
    path.write_text(
        (
            "import json, os, sys, time\n"
            "from pathlib import Path\n"
            "counter_path = Path(os.environ.get('COUNTER', 'counter'))\n"
            "epoch = int(counter_path.read_text()) + 1 if counter_path.exists() else 1\n"
            "counter_path.write_text(str(epoch))\n"
            "print('server stderr epoch=' + str(epoch), file=sys.stderr, flush=True)\n"
            "for raw in sys.stdin:\n"
            "    msg = json.loads(raw); method = msg.get('method')\n"
            "    if method == 'initialize':\n"
            "        result = {'protocolVersion': '2025-11-25'}\n"
            "    elif method == 'tools/list':\n"
            "        result = {'tools': [\n"
            + catalog
            + "            {'name': 'write_tool', 'description': 'write', 'inputSchema': {'type': 'object'}},\n"
            + "        ]}\n"
            + "    elif method == 'tools/call':\n"
            + "        name = msg['params']['name']\n"
            + "        if name == 'write_tool':\n"
            + "            result = {'isError': True, 'content': [{'type': 'text', 'text': 'write denied'}]}\n"
            + "        else:\n"
            + "            result = {'content': [{'type': 'text', 'text': '|'.join((\n"
            + "                os.environ.get('ROLE', 'none'), os.environ.get('GEN', 'none'),\n"
            + "                os.environ.get('PORT', 'none'), str(epoch),\n"
            + "            ))}]}\n"
            + "    else:\n"
            + "        continue\n"
            + "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)\n"
            + exit_code
        ),
        encoding="utf-8",
    )


async def _registry(
    tmp_path: Path,
    script: Path,
    *,
    required_tools: tuple[str, ...] = ("read_tool",),
    candidate_tools: tuple[str, ...] = ("read_tool",),
    candidate_env: dict[str, str] | None = None,
) -> tuple[CompositionRoot, McpServerRegistry]:
    plugin_dir = tmp_path / "calendar"
    plugin_dir.mkdir(exist_ok=True)
    (plugin_dir / script.name).write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
    root = CompositionRoot("mcp-host-test")
    service = PluginMcpServers(root.instance_token)
    _ = await root.context.provide(MCP_SERVERS, service)
    projected_candidate_env = (
        {"ROLE": "candidate"} if candidate_env is None else candidate_env
    )
    definition = McpServerDefinition(
        name="calendar",
        command=("python", script.name),
        cwd=".",
        env={"DECLARED": "yes"},
        required_tools=required_tools,
        candidate_read_only_tools=candidate_tools,
        endpoint_env=(EndpointEnv("PORT", "calendar_api"),),
        candidate_env=projected_candidate_env,
    )

    async def apply(ctx) -> None:
        await ctx.require(MCP_SERVERS).register(ctx, definition)

    _ = await root.mount(
        apply,
        name="calendar",
        inject=(MCP_SERVERS,),
        runtime=_runtime(plugin_dir),
    )
    return root, _freeze_plugin_mcp_servers(service, root.instance_token)


def _command(script: Path, *, env: dict[str, str] | None = None) -> dict[str, McpMaterializedCommand]:
    return {
        "calendar": McpMaterializedCommand(
            command=(sys.executable, str(script)),
            cwd=str(script.parent),
            env={"COUNTER": str(script.parent / "counter"), **(env or {})},
        )
    }


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        await asyncio.sleep(0.02)


def _generation_is_healthy(generation: McpGeneration) -> bool:
    try:
        generation.assert_healthy()
    except RuntimeError:
        return False
    return True


@pytest.mark.asyncio
async def test_candidate_filters_tools_and_projects_controlled_env(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)
    host = McpGenerationHost()
    try:
        generation = await host.start_candidate(
            "candidate-a",
            registry,
            _command(script, env={"GEN": "candidate-a"}),
            endpoint_ports={"calendar_api": _free_port()},
        )
        assert generation.state == "ready"
        assert generation.server("calendar").tool_names == ("read_tool",)
        assert len(generation.logs("calendar").stderr) <= 8
        result = await generation.route("calendar").call("read_tool", {})
        role, generation_name, port, _epoch = result.output.split("|")
        assert result.status == "success"
        assert role == "candidate"
        assert generation_name == "candidate-a"
        assert port.isdecimal()
        with pytest.raises(PermissionError, match="allowlist"):
            await generation.route("calendar").call("write_tool", {})
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_formal_exposes_all_tools_and_does_not_apply_candidate_env(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)
    host = McpGenerationHost()
    try:
        generation = await host.start_formal(
            "formal-a",
            registry,
            _command(script, env={"GEN": "formal-a"}),
            endpoint_ports={"calendar_api": _free_port()},
        )
        assert generation.server("calendar").tool_names == ("read_tool", "write_tool")
        result = await generation.route("calendar").call("write_tool", {})
        assert result.status == "tool_error"
        read_result = await generation.route("calendar").call("read_tool", {})
        assert read_result.output.startswith("none|formal-a|")
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_child_env_scrubs_ambient_candidate_value(monkeypatch, tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)
    monkeypatch.setenv("ROLE", "candidate")
    host = McpGenerationHost()
    try:
        formal = await host.start_formal(
            "formal-ambient-env",
            registry,
            _command(script),
            endpoint_ports={"calendar_api": _free_port()},
        )
        formal_result = await formal.route("calendar").call("read_tool", {})
        assert formal_result.output.split("|", 1)[0] == "none"
        await host.stop_generation("formal-ambient-env")

        candidate = await host.start_candidate(
            "candidate-ambient-env",
            registry,
            _command(script),
            endpoint_ports={"calendar_api": _free_port()},
        )
        candidate_result = await candidate.route("calendar").call("read_tool", {})
        assert candidate_result.output.split("|", 1)[0] == "candidate"
        await host.stop_generation("candidate-ambient-env")
    finally:
        await host.close()
        await root.dispose()


def test_host_reload_does_not_mutate_shared_process_environment_builder() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib\n"
                "import agent.mcp.client as client\n"
                "import agent.plugins.mcp_generation_host as host\n"
                "original = client.owned_process_env\n"
                "importlib.reload(host)\n"
                "assert client.owned_process_env is original\n"
                "assert isinstance(client.owned_process_env({}), dict)\n"
            ),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


@pytest.mark.asyncio
async def test_formal_rejects_catalog_drift_from_candidate_identity(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)
    host = McpGenerationHost()
    try:
        candidate = await host.start_candidate(
            "candidate-catalog",
            registry,
            _command(script),
            endpoint_ports={"calendar_api": _free_port()},
        )
        expected_digest = candidate.catalog_digest("calendar")
        _write_server(script, mode="catalog_drift")
        with pytest.raises(RuntimeError, match="catalog drift"):
            await host.start_formal(
                "formal-catalog-drift",
                registry,
                _command(script),
                endpoint_ports={"calendar_api": _free_port()},
                expected_catalog_digests={"calendar": expected_digest},
            )
        assert host.get("formal-catalog-drift") is None
        assert host.tombstone("formal-catalog-drift") is None
        assert candidate.catalog_digest("calendar") == expected_digest
        await host.stop_generation("candidate-catalog")
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_materialized_candidate_env_is_rejected_for_candidate_and_formal(
    tmp_path: Path,
) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)
    host = McpGenerationHost()
    try:
        for generation_id, start in (
            ("candidate-base-env", host.start_candidate),
            ("formal-base-env", host.start_formal),
        ):
            with pytest.raises(ValueError, match="candidate-only"):
                await start(
                    generation_id,
                    registry,
                    _command(script, env={"ROLE": "candidate"}),
                    endpoint_ports={"calendar_api": _free_port()},
                )
            assert host.get(generation_id) is None
        assert not (script.parent / "counter").exists()
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_declaration_rejects_overlapping_candidate_env(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(
        tmp_path,
        script,
        candidate_env={"DECLARED": "candidate"},
    )
    host = McpGenerationHost()
    try:
        with pytest.raises(ValueError, match="不得重叠"):
            await host.start_formal(
                "formal-overlap",
                registry,
                _command(script),
                endpoint_ports={"calendar_api": _free_port()},
            )
        assert host.get("formal-overlap") is None
        assert not (script.parent / "counter").exists()
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_materialized_argv0_must_be_pinned_absolute_executable(
    tmp_path: Path,
) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)
    host = McpGenerationHost()
    try:
        with pytest.raises(ValueError, match=r"argv\[0\].*absolute executable"):
            await host.start_candidate(
                "candidate-path-hostile",
                registry,
                {
                    "calendar": McpMaterializedCommand(
                        command=("python", str(script)),
                        cwd=str(script.parent),
                        env={"COUNTER": str(script.parent / "counter")},
                    )
                },
                endpoint_ports={"calendar_api": _free_port()},
            )
        assert host.get("candidate-path-hostile") is None
        assert not (script.parent / "counter").exists()
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_readiness_rejects_missing_required_tool_and_cleans_process(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(
        tmp_path,
        script,
        required_tools=("missing",),
        candidate_tools=(),
    )
    host = McpGenerationHost()
    try:
        with pytest.raises(RuntimeError, match="required tool 缺失"):
            await host.start_candidate(
                "candidate-missing",
                registry,
                _command(script),
                endpoint_ports={"calendar_api": _free_port()},
            )
        assert host.get("candidate-missing") is None
        assert host.tombstone("candidate-missing") is None
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_start_cancellation_drains_client_and_restores_cancelled_error(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script, mode="hang_initialize")
    root, registry = await _registry(tmp_path, script)
    host = McpGenerationHost(readiness_timeout_seconds=30)
    task = asyncio.create_task(
        host.start_candidate(
            "candidate-cancel",
            registry,
            _command(script),
            endpoint_ports={"calendar_api": _free_port()},
        )
    )
    try:
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert host.get("candidate-cancel") is None
        assert host.tombstone("candidate-cancel") is None
        assert not [
            current
            for current in asyncio.all_tasks()
            if current is not asyncio.current_task()
            and current.get_name().startswith("mcp_generation_")
        ]
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_start_health_bridge_cancellation_drains_client(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)

    def cancel_start(
        generation_id: str,
        server_name: str,
        healthy: bool,
        reason: str,
    ) -> None:
        if reason == "starting":
            raise asyncio.CancelledError

    host = McpGenerationHost(on_health=cancel_start)
    try:
        with pytest.raises(asyncio.CancelledError):
            await host.start_candidate(
                "candidate-health-cancel",
                registry,
                _command(script),
                endpoint_ports={"calendar_api": _free_port()},
            )
        assert host.get("candidate-health-cancel") is None
        assert host.tombstone("candidate-health-cancel") is None
        assert not [
            current
            for current in asyncio.all_tasks()
            if current is not asyncio.current_task()
            and current.get_name().startswith("mcp_generation_")
        ]
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_client_epoch_recovery_is_fenced_and_bounded(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "server.py"
    _write_server(script, mode="first_exit")
    monkeypatch.setattr(client_module, "_RECOVERY_DELAYS", (0.01, 0.01, 0.01))
    root, registry = await _registry(tmp_path, script)
    host = McpGenerationHost()
    try:
        generation = await host.start_candidate(
            "candidate-recover",
            registry,
            _command(script),
            endpoint_ports={"calendar_api": _free_port()},
        )
        initial_epoch = generation.server("calendar").epoch
        await _wait_until(lambda: generation.server("calendar").epoch > initial_epoch)
        await _wait_until(lambda: _generation_is_healthy(generation))
        generation.assert_healthy()
        result = await generation.route("calendar").call("read_tool", {})
        assert result.status == "success"
        assert result.output.endswith("|2")
        await host.stop_generation("candidate-recover")
        assert host.get("candidate-recover") is None
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_recovery_health_bridge_cancellation_retains_degraded_tombstone(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "server.py"
    _write_server(script, mode="first_exit")
    monkeypatch.setattr(client_module, "_RECOVERY_DELAYS", (0.01, 0.01, 0.01))
    root, registry = await _registry(tmp_path, script)

    def cancel_epoch_incident(
        generation_id: str,
        server_name: str,
        kind: str,
        message: str,
    ) -> None:
        if kind == "process_epoch":
            raise asyncio.CancelledError

    host = McpGenerationHost(on_incident=cancel_epoch_incident)
    try:
        generation = await host.start_candidate(
            "candidate-recovery-health-cancel",
            registry,
            _command(script),
            endpoint_ports={"calendar_api": _free_port()},
        )
        await _wait_until(
            lambda: host.tombstone("candidate-recovery-health-cancel") is not None,
            timeout=5,
        )
        assert generation.state == "degraded"
        tombstone = host.tombstone("candidate-recovery-health-cancel")
        assert tombstone is not None
        assert tombstone.state == "degraded"
        assert tombstone.action == "retry_runtime_recovery"
        await host.retry_runtime_recovery("candidate-recovery-health-cancel")
        assert host.get("candidate-recovery-health-cancel") is None
        assert host.tombstone("candidate-recovery-health-cancel") is None
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_cleanup_failure_retains_tombstone_until_retry(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)
    host = McpGenerationHost()
    await host.start_candidate(
        "candidate-cleanup",
        registry,
        _command(script),
        endpoint_ports={"calendar_api": _free_port()},
    )
    original_terminate = OwnedProcessGroup.terminate
    calls = 0

    async def fail_once(self: OwnedProcessGroup, *, timeout_s: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected terminate failure")
        await original_terminate(self, timeout_s=timeout_s)

    monkeypatch.setattr(OwnedProcessGroup, "terminate", fail_once)
    try:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await host.stop_generation("candidate-cleanup")
        tombstone = host.tombstone("candidate-cleanup")
        assert tombstone is not None
        assert tombstone.action == "retry_generation_cleanup"
        assert host.get("candidate-cleanup") is not None
        await host.retry_generation_cleanup("candidate-cleanup")
        assert host.tombstone("candidate-cleanup") is None
        assert host.get("candidate-cleanup") is None
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_stopped_health_failure_is_diagnostic_not_cleanup_failure(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)

    def fail_stopped(
        generation_id: str,
        server_name: str,
        healthy: bool,
        reason: str,
    ) -> None:
        if reason == "stopped":
            raise RuntimeError("health sink disposed")

    host = McpGenerationHost(on_health=fail_stopped)
    try:
        await host.start_candidate(
            "candidate-observation-failure",
            registry,
            _command(script),
            endpoint_ports={"calendar_api": _free_port()},
        )
        await host.stop_generation("candidate-observation-failure")
        assert host.get("candidate-observation-failure") is None
        assert host.tombstone("candidate-observation-failure") is None
        diagnostics = host.diagnostics("candidate-observation-failure")
        assert len(diagnostics) == 1
        assert diagnostics[0].reason == "stopped"
        assert "health sink disposed" in diagnostics[0].error
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_degraded_recovery_tombstone_is_actionable(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "server.py"
    _write_server(script, mode="always_exit")
    monkeypatch.setattr(client_module, "_RECOVERY_DELAYS", (0.01, 0.01, 0.01))
    root, registry = await _registry(tmp_path, script)
    host = McpGenerationHost()
    try:
        _ = await host.start_candidate(
            "candidate-degraded",
            registry,
            _command(script),
            endpoint_ports={"calendar_api": _free_port()},
        )
        await _wait_until(
            lambda: host.tombstone("candidate-degraded") is not None,
            timeout=5,
        )
        tombstone = host.tombstone("candidate-degraded")
        assert tombstone is not None
        assert tombstone.state == "degraded"
        assert tombstone.action == "retry_runtime_recovery"
        await host.retry_runtime_recovery("candidate-degraded")
        assert host.get("candidate-degraded") is None
        assert host.tombstone("candidate-degraded") is None
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_same_server_name_isolated_across_generations(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)
    host = McpGenerationHost()
    try:
        first = await host.start_candidate(
            "generation-a",
            registry,
            _command(script, env={"GEN": "a"}),
            endpoint_ports={"calendar_api": _free_port()},
        )
        second = await host.start_candidate(
            "generation-b",
            registry,
            _command(script, env={"GEN": "b"}),
            endpoint_ports={"calendar_api": _free_port()},
        )
        assert (await first.route("calendar").call("read_tool", {})).output.startswith(
            "candidate|a|"
        )
        assert (await second.route("calendar").call("read_tool", {})).output.startswith(
            "candidate|b|"
        )
        await host.stop_generation("generation-a")
        with pytest.raises(RuntimeError, match="stale|不可调用"):
            await first.route("calendar").call("read_tool", {})
        with pytest.raises(RuntimeError, match="stale|不可用"):
            _ = first.state
        with pytest.raises(RuntimeError, match="stale|不可用"):
            first.assert_healthy()
        assert (await second.route("calendar").call("read_tool", {})).status == "success"
    finally:
        await host.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_health_bridge_failure_prevents_ready_publication(tmp_path: Path) -> None:
    script = tmp_path / "server.py"
    _write_server(script)
    root, registry = await _registry(tmp_path, script)

    def fail_ready(generation_id: str, server_name: str, healthy: bool, reason: str) -> None:
        if reason == "ready":
            raise RuntimeError("health bridge failed")

    host = McpGenerationHost(on_health=fail_ready)
    try:
        with pytest.raises(RuntimeError, match="health bridge failed"):
            await host.start_candidate(
                "candidate-health-failure",
                registry,
                _command(script),
                endpoint_ports={"calendar_api": _free_port()},
            )
        assert host.get("candidate-health-failure") is None
        assert host.tombstone("candidate-health-failure") is None
    finally:
        await host.close()
        await root.dispose()

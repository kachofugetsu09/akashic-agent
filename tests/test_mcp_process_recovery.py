from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

import agent.mcp.client as client_module
import agent.mcp.host as host_module
from agent.mcp.client import McpClient, McpToolInfo
from agent.mcp.host import McpGenerationHost
from agent.plugins.scope import PluginScope


def _write_restarting_server(path: Path) -> None:
    """Create a server whose first epoch exits and later epochs serve calls."""
    _ = path.write_text(
        "import json, os, time\n"
        "from pathlib import Path\n"
        "counter = Path(os.environ['COUNTER'])\n"
        "epoch = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(epoch))\n"
        "for raw in __import__('sys').stdin:\n"
        "    msg = json.loads(raw); method = msg.get('method')\n"
        "    if method == 'initialize':\n"
        "        result = {'protocolVersion': '2025-11-25'}\n"
        "    elif method == 'tools/list':\n"
        "        result = {'tools': [{'name': 'ping', 'description': 'stable', "
        "'inputSchema': {'type': 'object'}}]}\n"
        "    elif method == 'tools/call':\n"
        "        result = {'content': [{'type': 'text', 'text': f'epoch:{epoch}'}]}\n"
        "    else:\n"
        "        continue\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], "
        "'result': result}), flush=True)\n"
        "    if epoch == 1 and method == 'tools/list':\n"
        "        time.sleep(0.05); raise SystemExit(17)\n",
        encoding="utf-8",
    )


def _write_drifting_server(path: Path) -> None:
    """Create a server whose recovery epochs publish a changed description."""
    _ = path.write_text(
        "import json, os, time\n"
        "from pathlib import Path\n"
        "counter = Path(os.environ['COUNTER'])\n"
        "epoch = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(epoch))\n"
        "for raw in __import__('sys').stdin:\n"
        "    msg = json.loads(raw); method = msg.get('method')\n"
        "    if method == 'initialize':\n"
        "        result = {'protocolVersion': '2025-11-25'}\n"
        "    elif method == 'tools/list':\n"
        "        description = 'stable' if epoch == 1 else 'drifted'\n"
        "        result = {'tools': [{'name': 'ping', 'description': description, "
        "'inputSchema': {'type': 'object'}}]}\n"
        "    else:\n"
        "        continue\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], "
        "'result': result}), flush=True)\n"
        "    if epoch == 1 and method == 'tools/list':\n"
        "        time.sleep(0.05); raise SystemExit(17)\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(os.name == "nt", reason="process recovery exercise targets Linux")
@pytest.mark.asyncio
async def test_mcp_client_recovers_process_epoch_and_keeps_logical_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "server.py"
    counter = tmp_path / "counter"
    _write_restarting_server(script)
    monkeypatch.setattr(client_module, "_RECOVERY_DELAYS", (0.01, 0.01, 0.01))
    client = McpClient(
        "recovering",
        [sys.executable, str(script)],
        env={"COUNTER": str(counter)},
    )

    try:
        infos = await client.connect()
        assert infos == [
            McpToolInfo(
                name="ping",
                description="stable",
                input_schema={"type": "object"},
            )
        ]
        deadline = asyncio.get_running_loop().time() + 5
        while (not counter.exists() or counter.read_text() != "2") or not client.connected:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("MCP client did not recover")
            await asyncio.sleep(0.02)
        assert await client.call("ping", {}) == "epoch:2"
        client.assert_healthy()
    finally:
        await client.disconnect()


@pytest.mark.skipif(os.name == "nt", reason="process recovery exercise targets Linux")
@pytest.mark.asyncio
async def test_mcp_client_real_drift_epochs_are_cleaned_before_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = tmp_path / "server.py"
    counter = tmp_path / "counter"
    _write_drifting_server(script)
    monkeypatch.setattr(client_module, "_RECOVERY_DELAYS", (0.01, 0.01, 0.01))
    client = McpClient(
        "drifting",
        [sys.executable, str(script)],
        env={"COUNTER": str(counter)},
    )

    await client.connect()
    failure = await asyncio.wait_for(client.wait_fatal_failure(), timeout=5)
    assert "恢复次数耗尽" in str(failure)
    assert "工具契约发生漂移" in str(failure)
    assert counter.read_text() == "4"
    assert client._process is None
    assert client._process_group is None
    with pytest.raises(RuntimeError, match="恢复次数耗尽"):
        await client.connect()
    assert counter.read_text() == "4"
    await client.disconnect()


@pytest.mark.asyncio
async def test_mcp_client_exhausts_three_backoffs_on_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = McpClient("drift", ["server"])
    client._expected_tool_contract = client._tool_contract(
        [McpToolInfo("ping", "stable", {"type": "object"})]
    )
    client._failure_count = 1
    monkeypatch.setattr(client_module, "_RECOVERY_DELAYS", (0.0, 0.0, 0.0))

    async def drifted_connect() -> list[McpToolInfo]:
        raise RuntimeError("恢复后的工具契约发生漂移")

    monkeypatch.setattr(client, "_connect_impl", drifted_connect)
    client._recovery_task = asyncio.create_task(client._recover())

    failure = await asyncio.wait_for(client.wait_fatal_failure(), timeout=1)
    assert "恢复次数耗尽" in str(failure)
    assert "工具契约发生漂移" in str(failure)
    with pytest.raises(RuntimeError, match="恢复次数耗尽"):
        client.assert_healthy()
    with pytest.raises(RuntimeError, match="恢复次数耗尽"):
        await client.call("ping", {})


@pytest.mark.asyncio
async def test_mcp_client_call_gate_waits_for_recovery_and_disconnect_cancels_it() -> None:
    client = McpClient("waiting", ["server"])

    class ConnectedProcess:
        returncode = None

    async def finish_recovery() -> None:
        await asyncio.sleep(0.01)
        client._process = cast(Any, ConnectedProcess())

    client._recovery_task = asyncio.create_task(finish_recovery())
    await client._await_available()
    assert client.connected

    client._process = None
    entered = asyncio.Event()

    async def blocked_recovery() -> None:
        entered.set()
        await asyncio.Event().wait()

    client._recovering = True
    client._recovery_task = asyncio.create_task(blocked_recovery())
    _ = await entered.wait()
    waiting_call = asyncio.create_task(client._await_available())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="正在恢复"):
        await client.connect()
    await client.disconnect()
    with pytest.raises(ConnectionError, match="恢复被停止"):
        await waiting_call
    assert client._stopping is True
    assert client._recovery_task is None


@pytest.mark.asyncio
async def test_mcp_client_resets_crash_budget_only_after_stable_epoch() -> None:
    assert client_module._RECOVERY_DELAYS == (0.25, 1.0, 3.0)
    client = McpClient("stable", ["server"])

    class ExitedProcess:
        returncode = 17

    class ProcessGroup:
        group_id = 123

        async def terminate(self, *, timeout_s: float) -> None:
            assert timeout_s > 0

    process = cast(Any, ExitedProcess())
    group = cast(Any, ProcessGroup())
    client._process = process
    client._process_group = group
    client._failure_count = 3
    client._epoch_started_at = (
        asyncio.get_running_loop().time()
        - client_module._RECOVERY_STABLE_SECONDS
        - 1
    )

    recovered = asyncio.Event()

    async def record_recovery() -> None:
        recovered.set()

    client._recover = record_recovery  # type: ignore[method-assign]
    await client._watch_process_exit(process, group)
    _ = await recovered.wait()
    assert client._failure_count == 1


@pytest.mark.parametrize(
    "changed",
    [
        McpToolInfo("other", "stable", {"type": "object"}),
        McpToolInfo("ping", "changed", {"type": "object"}),
        McpToolInfo(
            "ping",
            "stable",
            {"type": "object", "properties": {"value": {"type": "string"}}},
        ),
    ],
)
def test_mcp_recovery_contract_covers_name_description_and_schema(
    changed: McpToolInfo,
) -> None:
    stable = McpClient._tool_contract(
        [McpToolInfo("ping", "stable", {"type": "object"})]
    )
    assert McpClient._tool_contract([changed]) != stable


class _HostClient:
    def __init__(self, name: str, **_: object) -> None:
        self.name = name
        self.tool_infos = [McpToolInfo("ping", "stable", {"type": "object"})]
        self.failure = asyncio.Event()
        self.error = RuntimeError(f"fatal:{name}")
        self.connected = True
        self._recovering = False
        self._recovery_task: asyncio.Task[None] | None = None

    async def connect(self) -> list[McpToolInfo]:
        return self.tool_infos

    async def disconnect(self) -> None:
        return None

    def assert_healthy(self) -> None:
        if self.failure.is_set():
            raise self.error

    async def wait_fatal_failure(self) -> RuntimeError:
        await self.failure.wait()
        return self.error


@pytest.mark.asyncio
async def test_mcp_candidate_recovering_or_disconnected_rejects_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_HostClient] = []

    def factory(name: str, **kwargs: object) -> _HostClient:
        client = _HostClient(name, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(host_module, "McpClient", factory)
    host = McpGenerationHost()
    scope = PluginScope("candidate-gate")
    await host.prepare(
        "candidate-gate",
        server_specs={"feed": {"command": ["server"]}},
        required_tools={"feed": ("ping",)},
        scope=scope,
    )

    clients[0]._recovering = True
    with pytest.raises(RuntimeError, match="正在恢复，不能晋升"):
        host.assert_healthy("candidate-gate")
    clients[0]._recovering = False
    clients[0].connected = False
    with pytest.raises(RuntimeError, match="当前无可用 process epoch"):
        host.assert_healthy("candidate-gate")

    await host.close("candidate-gate")
    await scope.aclose()


@pytest.mark.asyncio
async def test_mcp_host_failure_is_generation_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_HostClient] = []

    def factory(name: str, **kwargs: object) -> _HostClient:
        client = _HostClient(name, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(host_module, "McpClient", factory)
    host = McpGenerationHost()
    scopes = [PluginScope("active-a"), PluginScope("active-b")]
    specs: Mapping[str, Mapping[str, Any]] = {"feed": {"command": ["server"]}}
    for generation_id, scope in zip(("active-a", "active-b"), scopes, strict=True):
        await host.prepare(
            generation_id,
            server_specs=specs,
            required_tools={"feed": ("ping",)},
            scope=scope,
        )
        host.mark_active(generation_id)

    clients[0].failure.set()
    await asyncio.sleep(0)

    assert host.state("active-a") == "active"
    assert host.state("active-b") == "active"
    assert str(host.failure("active-a")) == "fatal:feed@active-a"
    assert host.failure("active-b") is None
    with pytest.raises(RuntimeError, match="fatal:feed@active-a"):
        host.assert_healthy("active-a")
    host.assert_healthy("active-b")

    for generation_id, scope in zip(("active-a", "active-b"), scopes, strict=True):
        await host.close(generation_id)
        _ = await scope.aclose()


@pytest.mark.asyncio
async def test_mcp_host_rejects_duplicate_generation_before_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_HostClient] = []

    def factory(name: str, **kwargs: object) -> _HostClient:
        client = _HostClient(name, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(host_module, "McpClient", factory)
    host = McpGenerationHost()
    scope = PluginScope("duplicate")
    specs: Mapping[str, Mapping[str, Any]] = {"feed": {"command": ["server"]}}
    await host.prepare(
        "same",
        server_specs=specs,
        required_tools={"feed": ("ping",)},
        scope=scope,
    )
    with pytest.raises(RuntimeError, match="generation 已存在"):
        await host.prepare(
            "same",
            server_specs=specs,
            required_tools={"feed": ("ping",)},
            scope=scope,
        )
    assert len(clients) == 1
    await host.close("same")
    _ = await scope.aclose()

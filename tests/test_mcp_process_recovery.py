from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest

import agent.mcp.client as client_module
from agent.mcp.client import McpClient, McpToolInfo


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

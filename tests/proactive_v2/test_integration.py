from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from agent.plugins.snapshot import (
    RuntimeSnapshotLease,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from proactive_v2.config import ProactiveConfig
from proactive_v2.loop import ProactiveLoop


class SnapshotMcpTool(Tool):
    name = "mcp_feed__get_proactive_events"
    description = "Fetch proactive events."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: object) -> str:
        return "[]"


def make_loop() -> ProactiveLoop:
    loop = object.__new__(ProactiveLoop)
    loop._cfg = ProactiveConfig()
    loop._sense = SimpleNamespace(target_session_key=lambda: "telegram:1")
    loop._proactive_kernel = SimpleNamespace(run_tick=AsyncMock(return_value=None))
    loop._runtime_snapshot_store = None
    loop._reload_lock = asyncio.Lock()
    return loop


@pytest.mark.asyncio
async def test_tick_calls_kernel() -> None:
    loop = make_loop()

    result = await loop._tick()

    loop._proactive_kernel.run_tick.assert_awaited_once_with("telegram:1")
    assert result is None


@pytest.mark.asyncio
async def test_tick_return_is_propagated() -> None:
    loop = make_loop()
    loop._proactive_kernel.run_tick = AsyncMock(return_value=42.0)

    assert await loop._tick() == 42.0


@pytest.mark.asyncio
async def test_kernel_route_stable_across_multiple_ticks() -> None:
    loop = make_loop()

    await loop._tick()
    await loop._tick()
    await loop._tick()

    assert loop._proactive_kernel.run_tick.await_count == 3


@pytest.mark.asyncio
async def test_start_failure_always_marks_loop_stopped() -> None:
    loop = make_loop()
    loop._running = False
    loop._stopped = asyncio.Event()
    loop._kernel_started = False
    loop._active_kernel_lease = None
    loop._cfg.default_channel = "cli"
    loop._cfg.default_chat_id = "test"
    loop._runtime_snapshot_store = object()

    async def fail_start() -> None:
        raise RuntimeError("start failed")

    async def stop_active() -> None:
        return None

    loop._start_current_snapshot = fail_start
    loop._stop_active_kernel = stop_active

    with pytest.raises(RuntimeError, match="start failed"):
        await loop.run()

    assert loop._stopped.is_set()


@pytest.mark.asyncio
async def test_mcp_runtime_keeps_snapshot_tools_in_gateway_child_task(
    tmp_path: Path,
) -> None:
    base_tools = ToolRegistry()
    snapshot_tools = ToolRegistry()
    snapshot_tools.register(
        SnapshotMcpTool(),
        source_type="mcp",
        source_name="feed",
    )
    snapshot = SimpleNamespace(tool_registry=snapshot_tools)
    lease = cast(
        RuntimeSnapshotLease,
        SimpleNamespace(active=True, snapshot=snapshot),
    )
    loop = object.__new__(ProactiveLoop)
    loop._sessions = SimpleNamespace(workspace=tmp_path)
    loop._shared_tools = base_tools
    loop._plugin_proactive_sources = []

    token = bind_runtime_snapshot(lease)
    try:
        runtime = loop._build_mcp_runtime()
    finally:
        reset_runtime_snapshot(token)

    result = await asyncio.create_task(
        runtime.pool.call("feed", "get_proactive_events", {})
    )
    assert result == []

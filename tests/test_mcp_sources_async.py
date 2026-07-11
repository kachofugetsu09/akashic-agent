from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from agent.mcp.client import McpClient, McpToolInfo
from agent.mcp.tool import McpToolWrapper
from agent.plugins.specs import ProactiveSourceSpec, RegisteredProactiveSource
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from proactive_v2 import mcp_sources


class FakePool:
    def __init__(self, responses: dict[tuple[str, str], object], failures: set[tuple[str, str]] | None = None) -> None:
        self._workspace = Path("unused")
        self.responses = responses
        self.failures = failures or set()
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.timeouts: list[float | None] = []

    async def call(self, server: str, tool_name: str, args: dict[str, Any], *, timeout: float | None = None) -> Any:
        self.calls.append((server, tool_name, dict(args)))
        self.timeouts.append(timeout)
        if (server, tool_name) in self.failures:
            raise RuntimeError(f"failed: {server}.{tool_name}")
        return self.responses[(server, tool_name)]


def source(plugin_id: str, source_id: str, channels: tuple, server: str, fetch: str, *, ack: str = "", poll: str = "") -> RegisteredProactiveSource:
    return RegisteredProactiveSource(
        plugin_id=plugin_id,
        spec=ProactiveSourceSpec(
            id=source_id,
            channels=channels,
            server=server,
            fetch_tool=fetch,
            ack_tool=ack,
            poll_tool=poll,
            poll_interval_seconds=10 if poll else 0,
        ),
    )


class McpJsonTool(Tool):
    name = "get_proactive_events"
    description = "test"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return '[{"kind":"content","event_id":"1"}]'


class SlowMcpTool(McpJsonTool):
    async def execute(self, **kwargs: Any) -> str:
        await asyncio.sleep(1)
        return "[]"


@pytest.mark.asyncio
async def test_shared_mcp_gateway_reuses_tool_registry(tmp_path: Path) -> None:
    tools = ToolRegistry()
    tools.register(McpJsonTool(), source_type="mcp", source_name="feed")
    gateway = mcp_sources.SharedMcpGateway(tmp_path, tools)
    assert await gateway.call("feed", "get_proactive_events", {}) == [
        {"kind": "content", "event_id": "1"}
    ]


@pytest.mark.asyncio
async def test_shared_mcp_gateway_applies_timeout(tmp_path: Path) -> None:
    tools = ToolRegistry()
    tools.register(SlowMcpTool(), source_type="mcp", source_name="feed")
    gateway = mcp_sources.SharedMcpGateway(tmp_path, tools)
    with pytest.raises(asyncio.TimeoutError):
        await gateway.call("feed", "get_proactive_events", {}, timeout=0.01)


@pytest.mark.asyncio
async def test_shared_mcp_gateway_forwards_timeout_to_mcp_client(
    tmp_path: Path,
) -> None:
    client = AsyncMock()
    client.name = "feed"
    client.call.return_value = "[]"
    tool = McpToolWrapper(
        cast(McpClient, client),
        McpToolInfo(
            name="get_proactive_events",
            description="test",
            input_schema={"type": "object", "properties": {}},
        ),
    )
    tools = ToolRegistry()
    tools.register(tool, source_type="mcp", source_name="feed")
    gateway = mcp_sources.SharedMcpGateway(tmp_path, tools)

    assert await gateway.call(
        "feed",
        "get_proactive_events",
        {},
        timeout=12.0,
    ) == []
    client.call.assert_awaited_once_with(
        "get_proactive_events",
        {},
        timeout=12.0,
    )


@pytest.mark.asyncio
async def test_fetch_source_once_and_routes_multiple_channels() -> None:
    pool = FakePool(
        {("mixed", "events"): [
            {"kind": "alert", "event_id": "a1"},
            {"kind": "content", "event_id": "c1"},
        ]}
    )
    sources = [source("demo@lab", "mixed", ("alert", "content"), "mixed", "events")]
    result = await mcp_sources.fetch_sources_async(cast(Any, pool), sources)
    assert len(pool.calls) == 1
    assert result["alert"][0]["ack_server"] == "demo@lab:mixed"
    assert result["content"][0]["ack_server"] == "demo@lab:mixed"


@pytest.mark.asyncio
async def test_context_accepts_single_dict() -> None:
    pool = FakePool({("ctx", "context"): {"available": True}})
    sources = [source("demo", "sleep", ("context",), "ctx", "context")]
    result = await mcp_sources.fetch_sources_async(cast(Any, pool), sources)
    assert result["context"] == [{"available": True, "_source": "demo:sleep"}]


@pytest.mark.asyncio
async def test_fetch_isolates_single_source_failure() -> None:
    pool = FakePool(
        {("ok", "events"): [{"kind": "content", "event_id": "1"}], ("bad", "events"): []},
        failures={("bad", "events")},
    )
    sources = [
        source("ok", "content", ("content",), "ok", "events"),
        source("bad", "content", ("content",), "bad", "events"),
    ]
    result = await mcp_sources.fetch_sources_async(cast(Any, pool), sources)
    assert [item["event_id"] for item in result["content"]] == ["1"]


@pytest.mark.asyncio
async def test_ack_targets_exact_source_and_passes_feedback() -> None:
    pool = FakePool({("feed", "ack"): {"ok": True}})
    sources = [source("feed@lab", "subscriptions", ("content",), "feed", "events", ack="ack")]
    await mcp_sources.acknowledge_async(
        cast(Any, pool),
        sources,
        "feed@lab:subscriptions",
        ["e1"],
        feedback="interesting",
    )
    assert pool.calls == [("feed", "ack", {"event_ids": ["e1"], "feedback": "interesting"})]


@pytest.mark.asyncio
async def test_poll_uses_timeout() -> None:
    pool = FakePool({("feed", "poll"): {"ok": True}})
    item = source("feed", "subscriptions", ("content",), "feed", "events", poll="poll")
    await mcp_sources.poll_source_async(cast(Any, pool), item)
    assert pool.timeouts == [mcp_sources._POLL_TOOL_TIMEOUT]

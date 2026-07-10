from __future__ import annotations
import asyncio
from typing import Any, cast

from pathlib import Path

import pytest

from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from proactive_v2 import mcp_sources


class _FakePool:
    def __init__(
        self,
        responses: dict[tuple[str, str], object],
        failures: set[tuple[str, str]] | None = None,
    ) -> None:
        self._workspace = Path("unused-workspace")
        self._responses = responses
        self._failures = failures or set()
        self.calls: list[tuple[str, str, dict]] = []
        self.timeouts: list[float | None] = []

    async def call(
        self,
        server: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        timeout: float | None = None,
    ):
        self.calls.append((server, tool_name, dict(args)))
        self.timeouts.append(timeout)
        if (server, tool_name) in self._failures:
            raise RuntimeError(f"failed: {server}.{tool_name}")
        return self._responses[(server, tool_name)]


class _McpJsonTool(Tool):
    name = "get_proactive_events"
    description = "test"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        _ = kwargs
        return '[{"kind":"content","event_id":"1"}]'


class _NamespacedMcpJsonTool(_McpJsonTool):
    name = "mcp_feed__get_proactive_events"


class _FailingMcpTool(_McpJsonTool):
    async def execute(self, **kwargs: Any) -> str:
        _ = kwargs
        raise RuntimeError("boom")


class _SlowMcpTool(_McpJsonTool):
    async def execute(self, **kwargs: Any) -> str:
        _ = kwargs
        await asyncio.sleep(1)
        return "[]"


@pytest.mark.asyncio
async def test_shared_mcp_gateway_reuses_tool_registry(tmp_path: Path):
    tools = ToolRegistry()
    tools.register(
        _McpJsonTool(),
        source_type="mcp",
        source_name="feed",
    )
    gateway = mcp_sources.SharedMcpGateway(tmp_path, tools)

    result = await gateway.call("feed", "get_proactive_events", {})

    assert result == [{"kind": "content", "event_id": "1"}]


@pytest.mark.asyncio
async def test_shared_mcp_gateway_resolves_namespaced_mcp_tool(tmp_path: Path):
    tools = ToolRegistry()
    tools.register(
        _NamespacedMcpJsonTool(),
        source_type="mcp",
        source_name="feed",
    )
    gateway = mcp_sources.SharedMcpGateway(tmp_path, tools)

    result = await gateway.call("feed", "get_proactive_events", {})

    assert result == [{"kind": "content", "event_id": "1"}]


@pytest.mark.asyncio
async def test_shared_mcp_gateway_raises_registry_execution_error(tmp_path: Path):
    tools = ToolRegistry()
    tools.register(_FailingMcpTool(), source_type="mcp", source_name="feed")
    gateway = mcp_sources.SharedMcpGateway(tmp_path, tools)

    with pytest.raises(RuntimeError, match="boom"):
        await gateway.call("feed", "get_proactive_events", {})


@pytest.mark.asyncio
async def test_shared_mcp_gateway_applies_timeout(tmp_path: Path):
    tools = ToolRegistry()
    tools.register(_SlowMcpTool(), source_type="mcp", source_name="feed")
    gateway = mcp_sources.SharedMcpGateway(tmp_path, tools)

    with pytest.raises(asyncio.TimeoutError):
        await gateway.call("feed", "get_proactive_events", {}, timeout=0.01)


@pytest.mark.asyncio
async def test_fetch_alert_events_async_filters_kind_and_sets_ack_server(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda _w=None: [
            {"channel": "alert", "server": "s1", "get_tool": "get_proactive_events"},
            {"channel": "context", "server": "ctx", "get_tool": "get_context"},
        ],
    )
    pool = _FakePool(
        {
            ("s1", "get_proactive_events"): [
                {"kind": "alert", "event_id": "a1"},
                {"kind": "content", "event_id": "c1"},
            ],
            ("ctx", "get_context"): {"available": True},
        }
    )

    result = await mcp_sources.fetch_alert_events_async(cast(Any, pool))

    assert result == [{"kind": "alert", "event_id": "a1", "ack_server": "s1"}]


@pytest.mark.asyncio
async def test_fetch_content_events_async_keeps_default_compat_channel_filter(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda _w=None: [
            {"channel": "", "server": "s1", "get_tool": "get_proactive_events"},
            {"channel": "alert", "server": "alert_only", "get_tool": "get_proactive_events"},
        ],
    )
    pool = _FakePool(
        {
            ("s1", "get_proactive_events"): [
                {"kind": "content", "event_id": "n1"},
                {"kind": "alert", "event_id": "a1"},
            ],
            ("alert_only", "get_proactive_events"): [{"kind": "content", "event_id": "x"}],
        }
    )

    result = await mcp_sources.fetch_content_events_async(cast(Any, pool))

    assert result == [{"kind": "content", "event_id": "n1", "ack_server": "s1"}]


@pytest.mark.asyncio
async def test_fetch_context_data_async_accepts_list(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda _w=None: [
            {"channel": "context", "server": "ctx1", "get_tool": "get_context"},
            {"channel": "context", "server": "ctx2", "get_tool": "get_context"},
        ],
    )
    pool = _FakePool(
        {
            ("ctx1", "get_context"): [{"available": True}],
            ("ctx2", "get_context"): [{"available": False}, "bad_item"],
        }
    )

    result = await mcp_sources.fetch_context_data_async(cast(Any, pool))

    assert result == [
        {"available": True, "_source": "ctx1"},
        {"available": False, "_source": "ctx2"},
    ]


@pytest.mark.asyncio
async def test_fetch_content_events_async_raises_when_source_failed(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda _w=None: [
            {"channel": "content", "server": "feed", "get_tool": "get_events"},
        ],
    )
    pool = _FakePool(
        {("feed", "get_events"): []},
        failures={("feed", "get_events")},
    )

    with pytest.raises(RuntimeError, match="feed"):
        await mcp_sources.fetch_content_events_async(cast(Any, pool))


@pytest.mark.asyncio
async def test_fetch_content_events_async_keeps_results_when_one_source_failed(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda _w=None: [
            {"channel": "content", "server": "ok", "get_tool": "get_events"},
            {"channel": "content", "server": "failed", "get_tool": "get_events"},
        ],
    )
    pool = _FakePool(
        {
            ("ok", "get_events"): [{"kind": "content", "event_id": "1"}],
            ("failed", "get_events"): [],
        },
        failures={("failed", "get_events")},
    )

    result = await mcp_sources.fetch_content_events_async(cast(Any, pool))

    assert result == [{"kind": "content", "event_id": "1", "ack_server": "ok"}]


@pytest.mark.asyncio
async def test_poll_content_feeds_async_raises_when_any_source_failed(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda _w=None: [
            {"channel": "content", "server": "s1", "poll_tool": "poll"},
            {"channel": "content", "server": "s2", "poll_tool": "poll"},
            {"channel": "alert", "server": "a1", "poll_tool": "poll"},
        ],
    )
    pool = _FakePool(
        {
            ("s1", "poll"): {"ok": True},
            ("s2", "poll"): {"ok": True},
            ("a1", "poll"): {"ok": True},
        },
        failures={("s2", "poll")},
    )

    with pytest.raises(RuntimeError) as exc:
        await mcp_sources.poll_content_feeds_async(cast(Any, pool))

    assert "s2" in str(exc.value)
    assert ("a1", "poll", {}) not in pool.calls
    assert pool.timeouts == [mcp_sources._POLL_TOOL_TIMEOUT, mcp_sources._POLL_TOOL_TIMEOUT]


@pytest.mark.asyncio
async def test_acknowledge_events_async_groups_by_ack_server(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda _w=None: [
            {"server": "fitbit", "ack_tool": "ack_events"},
            {"server": "feed", "ack_tool": "ack_events"},
        ],
    )
    pool = _FakePool(
        {
            ("fitbit", "ack_events"): {"ok": True},
            ("feed", "ack_events"): {"ok": True},
        }
    )

    events = [
        ("fitbit", "a1"),
        ("fitbit", "a2"),
        ("feed", "a3"),
        ("unknown", "x"),
    ]
    await mcp_sources.acknowledge_events_async(cast(Any, pool), events)

    assert ("fitbit", "ack_events", {"event_ids": ["a1", "a2"]}) in pool.calls
    assert ("feed", "ack_events", {"event_ids": ["a3"]}) in pool.calls


@pytest.mark.asyncio
async def test_acknowledge_events_async_raises_when_ack_failed(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda _w=None: [{"server": "feed", "ack_tool": "ack_events"}],
    )
    pool = _FakePool(
        {("feed", "ack_events"): {"ok": True}},
        failures={("feed", "ack_events")},
    )

    with pytest.raises(RuntimeError, match="feed"):
        await mcp_sources.acknowledge_events_async(cast(Any, pool), [("feed", "a1")])


@pytest.mark.asyncio
async def test_acknowledge_content_entries_async_passes_feedback(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda _w=None: [{"server": "feed", "ack_tool": "ack_content"}],
    )
    pool = _FakePool({("feed", "ack_content"): {"ok": True}})

    entries = [
        ("mcp:feed:evt-1", "fallback-1"),
        ("mcp:feed", "evt-2"),
        ("rss:other", "skip"),
    ]
    await mcp_sources.acknowledge_content_entries_async(cast(Any, pool), entries, feedback="interesting")

    assert (
        "feed",
        "ack_content",
        {"event_ids": ["evt-1", "evt-2"], "feedback": "interesting"},
    ) in pool.calls


@pytest.mark.asyncio
async def test_acknowledge_content_entries_async_raises_when_ack_failed(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda _w=None: [{"server": "feed", "ack_tool": "ack_content"}],
    )
    pool = _FakePool(
        {("feed", "ack_content"): {"ok": True}},
        failures={("feed", "ack_content")},
    )

    with pytest.raises(RuntimeError, match="feed"):
        await mcp_sources.acknowledge_content_entries_async(
            cast(Any, pool),
            [("mcp:feed", "a1")],
            feedback="interesting",
        )

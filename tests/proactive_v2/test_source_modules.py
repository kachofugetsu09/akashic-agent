from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from proactive_v2.config import ProactiveConfig
from proactive_v2.modules_source import McpGatewaySource, McpRuntimeModule


@pytest.mark.asyncio
async def test_mcp_runtime_module_manages_pool_lifecycle(monkeypatch):
    poll = AsyncMock()
    monkeypatch.setattr(
        "proactive_v2.modules_source.mcp_sources.poll_content_feeds_async",
        poll,
    )
    module = McpRuntimeModule(workspace=None, cfg=ProactiveConfig())
    module.pool.connect_all = AsyncMock(return_value=None)  # type: ignore[method-assign]
    module.pool.disconnect_all = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await module.start()
    await module.stop()

    module.pool.connect_all.assert_awaited_once()
    poll.assert_awaited_once_with(module.pool)
    module.pool.disconnect_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_mcp_gateway_source_builds_gateway_deps(monkeypatch):
    async def fake_alerts(pool):
        return [{"event_id": "alert-1"}]

    async def fake_content(pool):
        return [{"event_id": "c1"}, {"event_id": "c2"}]

    async def fake_context(pool):
        return [{"kind": "sleep"}]

    monkeypatch.setattr(
        "proactive_v2.modules_source.mcp_sources.fetch_alert_events_async",
        fake_alerts,
    )
    monkeypatch.setattr(
        "proactive_v2.modules_source.mcp_sources.fetch_content_events_async",
        fake_content,
    )
    monkeypatch.setattr(
        "proactive_v2.modules_source.mcp_sources.fetch_context_data_async",
        fake_context,
    )

    source = McpGatewaySource(object(), content_limit=5)  # type: ignore[arg-type]
    deps = source.build_gateway_deps(web_fetch_tool=None, max_chars=123)

    assert await deps.alert_fn() == [{"event_id": "alert-1"}]
    assert await deps.feed_fn(limit=1) == [{"event_id": "c1"}]
    assert await deps.context_fn() == [{"kind": "sleep"}]
    assert deps.content_limit == 5
    assert deps.max_chars == 123


@pytest.mark.asyncio
async def test_mcp_gateway_source_ack_routes(monkeypatch):
    content_ack = AsyncMock()
    alert_ack = AsyncMock()
    monkeypatch.setattr(
        "proactive_v2.modules_source.mcp_sources.acknowledge_content_entries_async",
        content_ack,
    )
    monkeypatch.setattr(
        "proactive_v2.modules_source.mcp_sources.acknowledge_events_async",
        alert_ack,
    )

    pool = object()
    source = McpGatewaySource(pool, content_limit=5)  # type: ignore[arg-type]

    await source.ack_fn("feed-mcp:item-1", 720)
    await source.alert_ack_fn("calendar:alert-1")

    content_ack.assert_awaited_once_with(
        pool,
        [("mcp:feed-mcp", "item-1")],
        ttl_hours=720,
    )
    assert alert_ack.await_args.args[1] == [("calendar", "alert-1")]

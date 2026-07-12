from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.plugins.specs import ProactiveSourceSpec, RegisteredProactiveSource
from plugins.default_proactive.source import McpGatewaySource
from plugins.default_proactive.source_poll import DefaultSourcePollModule
from plugins.default_proactive.plugin import DefaultModuleFactory
from plugins.default_proactive.runtime import ProactiveFlowRuntime


def registered(*, poll: bool = False) -> RegisteredProactiveSource:
    return RegisteredProactiveSource(
        plugin_id="feed@lab",
        spec=ProactiveSourceSpec(
            id="subscriptions",
            channels=("content",),
            server="feed",
            fetch_tool="events",
            ack_tool="ack",
            poll_tool="poll" if poll else "",
            poll_interval_seconds=3600 if poll else 0,
        ),
    )


def test_default_lifecycle_owns_source_poll_module() -> None:
    poll_module = object()
    runtime = object.__new__(ProactiveFlowRuntime)
    runtime._source_poll_module = poll_module

    modules = DefaultModuleFactory()(runtime)

    assert modules[0] is poll_module


@pytest.mark.asyncio
async def test_mcp_runtime_module_manages_poll_lifecycle(monkeypatch) -> None:
    poll = AsyncMock()
    monkeypatch.setattr("plugins.default_proactive.source_poll.mcp_sources.poll_source_async", poll)
    gateway = object()
    source = registered(poll=True)
    module = DefaultSourcePollModule(gateway=gateway, sources=[source])  # type: ignore[arg-type]

    await module.start()
    await module.stop()

    poll.assert_awaited_once_with(gateway, source)


@pytest.mark.asyncio
async def test_mcp_gateway_source_fetches_one_snapshot(monkeypatch) -> None:
    fetch = AsyncMock(
        return_value={
            "alert": [{"event_id": "alert-1"}],
            "content": [{"event_id": "c1"}, {"event_id": "c2"}],
            "context": [{"kind": "sleep"}],
        }
    )
    monkeypatch.setattr("plugins.default_proactive.source.mcp_sources.fetch_sources_async", fetch)
    pool = object()
    source = McpGatewaySource(pool, [registered()], content_limit=5)  # type: ignore[arg-type]
    deps = source.build_gateway_deps(web_fetch_tool=None, max_chars=123)
    deps.begin_fn()

    assert await deps.alert_fn() == [{"event_id": "alert-1"}]
    assert await deps.feed_fn(limit=1) == [{"event_id": "c1"}]
    assert await deps.context_fn() == [{"kind": "sleep"}]
    fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_mcp_gateway_source_ack_routes_exact_source(monkeypatch) -> None:
    acknowledge = AsyncMock()
    monkeypatch.setattr("plugins.default_proactive.source.mcp_sources.acknowledge_async", acknowledge)
    pool = object()
    sources = [registered()]
    source = McpGatewaySource(pool, sources, content_limit=5)  # type: ignore[arg-type]

    await source.ack_fn("feed@lab:subscriptions:item:1", "not_interesting")
    await source.alert_ack_fn("feed@lab:subscriptions:alert-1")

    assert acknowledge.await_args_list[0].args == (
        pool,
        sources,
        "feed@lab:subscriptions",
        ["item:1"],
    )
    assert acknowledge.await_args_list[0].kwargs == {"feedback": "not_interesting"}
    assert acknowledge.await_args_list[1].args[-2:] == (
        "feed@lab:subscriptions",
        ["alert-1"],
    )

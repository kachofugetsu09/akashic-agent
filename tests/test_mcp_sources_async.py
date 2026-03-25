from __future__ import annotations

import pytest

from proactive_v2 import mcp_sources


class _FakePool:
    def __init__(
        self,
        responses: dict[tuple[str, str], object],
        failures: set[tuple[str, str]] | None = None,
    ) -> None:
        self._responses = responses
        self._failures = failures or set()
        self.calls: list[tuple[str, str, dict]] = []

    async def call(self, server: str, tool_name: str, args: dict):
        self.calls.append((server, tool_name, dict(args)))
        if (server, tool_name) in self._failures:
            raise RuntimeError(f"failed: {server}.{tool_name}")
        return self._responses[(server, tool_name)]


@pytest.mark.asyncio
async def test_fetch_alert_events_async_filters_kind_and_sets_ack_server(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda: [
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

    result = await mcp_sources.fetch_alert_events_async(pool)

    assert result == [{"kind": "alert", "event_id": "a1", "ack_server": "s1"}]


@pytest.mark.asyncio
async def test_fetch_content_events_async_keeps_default_compat_channel_filter(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda: [
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

    result = await mcp_sources.fetch_content_events_async(pool)

    assert result == [{"kind": "content", "event_id": "n1", "ack_server": "s1"}]


@pytest.mark.asyncio
async def test_fetch_context_data_async_accepts_dict_and_list(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda: [
            {"channel": "context", "server": "ctx1", "get_tool": "get_context"},
            {"channel": "context", "server": "ctx2", "get_tool": "get_context"},
        ],
    )
    pool = _FakePool(
        {
            ("ctx1", "get_context"): {"available": True},
            ("ctx2", "get_context"): [{"available": False}, "bad_item"],
        }
    )

    result = await mcp_sources.fetch_context_data_async(pool)

    assert result == [
        {"available": True, "_source": "ctx1"},
        {"available": False, "_source": "ctx2"},
    ]


@pytest.mark.asyncio
async def test_poll_content_feeds_async_raises_when_any_source_failed(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda: [
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
        await mcp_sources.poll_content_feeds_async(pool)

    assert "s2" in str(exc.value)
    assert ("a1", "poll", {}) not in pool.calls


@pytest.mark.asyncio
async def test_acknowledge_events_async_groups_by_ack_server(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda: [
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

    class _Evt:
        def __init__(self, ack_server, ack_id, source_name=""):
            self._ack_server = ack_server
            self.ack_id = ack_id
            self.source_name = source_name

    events = [
        _Evt("fitbit", "a1"),
        _Evt("fitbit", "a2"),
        _Evt("", "a3", source_name="feed"),
        _Evt("unknown", "x"),
    ]
    await mcp_sources.acknowledge_events_async(pool, events)

    assert ("fitbit", "ack_events", {"event_ids": ["a1", "a2"]}) in pool.calls
    assert ("feed", "ack_events", {"event_ids": ["a3"]}) in pool.calls


@pytest.mark.asyncio
async def test_acknowledge_content_entries_async_passes_ttl_hours(monkeypatch):
    monkeypatch.setattr(
        mcp_sources,
        "_load_sources",
        lambda: [{"server": "feed", "ack_tool": "ack_content"}],
    )
    pool = _FakePool({("feed", "ack_content"): {"ok": True}})

    entries = [
        ("mcp:feed:evt-1", "fallback-1"),
        ("mcp:feed", "evt-2"),
        ("rss:other", "skip"),
    ]
    await mcp_sources.acknowledge_content_entries_async(pool, entries, ttl_hours=24)

    assert (
        "feed",
        "ack_content",
        {"event_ids": ["evt-1", "evt-2"], "ttl_hours": 24},
    ) in pool.calls

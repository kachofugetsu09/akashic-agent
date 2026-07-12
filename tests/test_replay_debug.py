import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest

from agent.tools.message_push import MessagePushTool
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from docker.debug.plugins.replay_debug.plugin import CaptureChannel, ReplayDebugPlugin
from docker.debug.plugins.replay_debug.replay_mcp import (
    acknowledge_replay_events,
    fetch_replay_events,
)
from docker.debug.replay_controller import (
    ReplayLayout,
    append_event,
    initialize,
    normalize_event,
    status,
)
from bootstrap.tools import _resolve_plugin_dirs
from infra.channels.contract import ChannelContext
from proactive_v2.mcp_sources import SharedMcpGateway


class _ReplayFetchTool(Tool):
    name = "mcp_replay-debug__fetch_replay_events"
    description = "replay"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return fetch_replay_events()


@pytest.mark.asyncio
async def test_capture_channel_records_replay_time(tmp_path, monkeypatch) -> None:
    clock_path = tmp_path / "clock.json"
    outbox_path = tmp_path / "outbox.jsonl"
    layout = ReplayLayout(
        tmp_path,
        tmp_path,
        clock_path,
        tmp_path / "events.jsonl",
        outbox_path,
    )
    initialize(layout, datetime(2026, 3, 4, 5, 6, tzinfo=UTC))
    monkeypatch.setenv("AKASHIC_REPLAY_CLOCK_FILE", str(clock_path))
    push = MessagePushTool()
    channel = CaptureChannel(outbox_path)

    await channel.start(cast(ChannelContext, SimpleNamespace(push_tool=push)))
    result = await push.execute(channel="replay", chat_id="user", message="hello")
    await channel.stop()

    assert result == "文本已发送"
    report = status(layout)
    assert report["outbox_messages"] == 1
    assert report["latest_outbound"]["message"] == "hello"
    assert report["latest_outbound"]["captured_at"] == "2026-03-04T05:06:00+00:00"


def test_replay_controller_tracks_available_events(tmp_path) -> None:
    layout = ReplayLayout(
        tmp_path,
        tmp_path,
        tmp_path / "clock.json",
        tmp_path / "events.jsonl",
        tmp_path / "outbox.jsonl",
    )
    initialize(layout, datetime(2026, 3, 4, 5, tzinfo=UTC))
    append_event(
        layout,
        {
            "event_id": "old",
            "source_id": "feed",
            "published_at": "2026-03-04T04:00:00Z",
            "preprocess_score": 0.8,
        },
    )
    append_event(
        layout,
        {
            "event_id": "future",
            "source_id": "feed",
            "published_at": "2026-03-04T06:00:00Z",
        },
    )

    report = status(layout)
    assert report["available_events"] == 1
    assert report["future_events"] == 1
    saved = list(layout.events_path.read_text(encoding="utf-8").splitlines())
    assert '"preprocess_score": 0.8' in saved[0]


def test_replay_event_requires_timezone() -> None:
    with pytest.raises(ValueError, match="时区"):
        normalize_event(
            {
                "event_id": "bad",
                "source_id": "feed",
                "published_at": "2026-03-04T05:00:00",
            }
        )


def test_debug_plugin_dir_can_be_added_from_env(tmp_path, monkeypatch) -> None:
    extra = tmp_path / "debug-plugins"
    monkeypatch.setenv("AKASHIC_EXTRA_PLUGIN_DIRS", str(extra))

    assert _resolve_plugin_dirs(tmp_path)[-1] == extra


def test_replay_mcp_only_returns_available_unacked_events(tmp_path, monkeypatch) -> None:
    layout = ReplayLayout(
        tmp_path,
        tmp_path,
        tmp_path / "clock.json",
        tmp_path / "events.jsonl",
        tmp_path / "outbox.jsonl",
    )
    initialize(layout, datetime(2026, 3, 4, 5, tzinfo=UTC))
    for event_id, kind, available_at in (
        ("alert-now", "alert", "2026-03-04T04:00:00Z"),
        ("content-now", "content", "2026-03-04T05:00:00Z"),
        ("context-future", "context", "2026-03-04T06:00:00Z"),
    ):
        append_event(
            layout,
            {
                "event_id": event_id,
                "kind": kind,
                "source_id": "replay",
                "title": event_id,
                "published_at": available_at,
                "available_at": available_at,
            },
        )
    monkeypatch.setenv("AKASHIC_REPLAY_CLOCK_FILE", str(layout.clock_path))
    monkeypatch.setenv("AKASHIC_REPLAY_EVENTS_FILE", str(layout.events_path))

    assert [event["event_id"] for event in json.loads(fetch_replay_events())] == [
        "alert-now",
        "content-now",
    ]

    result = json.loads(
        acknowledge_replay_events(["content-now"], feedback="interesting")
    )

    assert result == {"acked": 1}
    assert [event["event_id"] for event in json.loads(fetch_replay_events())] == ["alert-now"]
    ack_payload = json.loads((layout.replay_root / "acks.json").read_text(encoding="utf-8"))
    assert set(ack_payload["acked"]["content-now"]) == {"acked_at"}


def test_replay_debug_plugin_declares_three_channel_mcp_source(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AKASHIC_REPLAY_CLOCK_FILE", str(tmp_path / "clock.json"))
    monkeypatch.setenv("AKASHIC_REPLAY_EVENTS_FILE", str(tmp_path / "events.jsonl"))
    plugin = ReplayDebugPlugin()
    server = plugin.mcp_servers()[0]
    source = plugin.proactive_sources()[0]

    assert server.name == "replay-debug"
    assert server.command == ("python", "replay_mcp.py")
    assert source.channels == ("alert", "content", "context")
    assert source.fetch_tool == "fetch_replay_events"
    assert source.ack_tool == "acknowledge_replay_events"
    assert source.fetch_page_size == 50


def test_replay_debug_plugin_is_inert_without_replay_profile(monkeypatch) -> None:
    monkeypatch.delenv("AKASHIC_REPLAY_CLOCK_FILE", raising=False)
    monkeypatch.delenv("AKASHIC_REPLAY_EVENTS_FILE", raising=False)
    monkeypatch.delenv("AKASHIC_REPLAY_OUTBOX_FILE", raising=False)
    plugin = ReplayDebugPlugin()

    assert plugin.mcp_servers() == []
    assert plugin.proactive_sources() == []
    assert plugin.channels() == []


@pytest.mark.asyncio
async def test_replay_fetch_json_round_trips_through_shared_gateway(
    tmp_path, monkeypatch
) -> None:
    layout = ReplayLayout(
        tmp_path,
        tmp_path,
        tmp_path / "clock.json",
        tmp_path / "events.jsonl",
        tmp_path / "outbox.jsonl",
    )
    initialize(layout, datetime(2026, 3, 4, 5, tzinfo=UTC))
    for event_id in ("one", "two"):
        append_event(
            layout,
            {
                "event_id": event_id,
                "kind": "content",
                "source_id": "replay",
                "published_at": "2026-03-04T04:00:00Z",
            },
        )
    monkeypatch.setenv("AKASHIC_REPLAY_CLOCK_FILE", str(layout.clock_path))
    monkeypatch.setenv("AKASHIC_REPLAY_EVENTS_FILE", str(layout.events_path))
    registry = ToolRegistry()
    registry.register(
        _ReplayFetchTool(),
        source_type="mcp",
        source_name="replay-debug",
    )

    result = await SharedMcpGateway(tmp_path, registry).call(
        "replay-debug", "fetch_replay_events", {}
    )

    assert [event["event_id"] for event in result] == ["one", "two"]

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from bus.event_bus import EventBus
from infra.channels.base import AttachmentStore
from infra.channels.contract import ChannelContext
from plugins.qqbot.plugin import QQBotConfigModel, QQBotPlugin


class _Bus:
    def __init__(self) -> None:
        self.outbound = []

    def subscribe_outbound(self, channel: str, callback: object) -> None:
        self.outbound.append((channel, callback))


class _PushTool:
    def __init__(self) -> None:
        self.registrations = []

    def register_channel(self, name: str, **kwargs: object) -> None:
        self.registrations.append((name, sorted(kwargs)))


def test_qqbot_plugin_returns_no_channel_without_credentials():
    plugin = QQBotPlugin()
    plugin.context = cast(Any, SimpleNamespace(config=QQBotConfigModel()))

    assert plugin.channels() == []


@pytest.mark.asyncio
async def test_qqbot_plugin_channel_registers_runtime_hooks():
    plugin = QQBotPlugin()
    plugin.context = cast(Any, SimpleNamespace(config=QQBotConfigModel(
        app_id="app",
        client_secret="secret",
        allow_from=["user-1"],
    )))
    channel = plugin.channels()[0]

    async def _no_gateway_loop() -> None:
        return None

    channel._gateway_loop = _no_gateway_loop
    bus = _Bus()
    push_tool = _PushTool()
    await channel.start(
        ChannelContext(
            bus=cast(Any, bus),
            session_manager=cast(Any, SimpleNamespace()),
            event_bus=EventBus(),
            push_tool=cast(Any, push_tool),
            attachment_store=AttachmentStore(),
            http_resources=cast(Any, SimpleNamespace()),
            interrupt_controller=None,
            bot_commands=[],
            log=logging.getLogger("test.qqbot"),
        )
    )
    await channel.stop()

    assert bus.outbound[0][0] == "qqbot"
    assert push_tool.registrations == [("qqbot", ["stream_text", "text"])]

from __future__ import annotations

import logging
from collections.abc import Callable

from bootstrap.channel_host import ChannelHost
from bus.event_bus import EventBus
from bus.queue import MessageBus
from core.net.http import SharedHttpResources
from infra.channels.base import AttachmentStore
from infra.channels.contract import Channel, ChannelContext
from pathlib import Path


async def start_channels(
    *,
    bus: MessageBus,
    workspace: Path,
    http_resources: SharedHttpResources,
    event_bus: EventBus,
    command_catalog_provider: Callable[
        [], tuple[tuple[str, str], ...]
    ] | None = None,
    extra_channels: list[Channel] | None = None,
) -> ChannelHost:
    attachment_store = AttachmentStore(workspace / "uploads")

    def _ctx_factory(channel: Channel) -> ChannelContext:
        return ChannelContext(
            bus=bus,
            event_bus=event_bus,
            attachment_store=attachment_store,
            http_resources=http_resources,
            log=logging.getLogger(f"channels.{channel.name}"),
            command_catalog_provider=command_catalog_provider,
        )

    host = ChannelHost(_ctx_factory)

    for channel in extra_channels or []:
        host.add(channel)

    return host

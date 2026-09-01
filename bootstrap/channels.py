from __future__ import annotations

import logging
from collections.abc import Callable

from agent.config_models import Config
from agent.looping.interrupt import InterruptController
from agent.tools.message_push import MessagePushTool
from bootstrap.channel_host import ChannelHost
from bus.event_bus import EventBus
from bus.queue import MessageBus
from core.net.http import SharedHttpResources
from infra.channels.base import AttachmentStore
from infra.channels.contract import Channel, ChannelContext
from session.manager import SessionManager


async def start_channels(
    config: Config,
    *,
    bus: MessageBus,
    session_manager: SessionManager,
    push_tool: MessagePushTool,
    http_resources: SharedHttpResources,
    event_bus: EventBus,
    command_catalog_provider: Callable[
        [], tuple[tuple[str, str], ...]
    ] | None = None,
    interrupt_controller: InterruptController | None = None,
    extra_channels: list[Channel] | None = None,
) -> ChannelHost:
    attachment_store = AttachmentStore(session_manager.workspace / "uploads")

    def _ctx_factory(channel: Channel) -> ChannelContext:
        return ChannelContext(
            bus=bus,
            session_manager=session_manager,
            event_bus=event_bus,
            push_tool=push_tool,
            attachment_store=attachment_store,
            http_resources=http_resources,
            interrupt_controller=interrupt_controller,
            log=logging.getLogger(f"channels.{channel.name}"),
            command_catalog_provider=command_catalog_provider,
        )

    host = ChannelHost(_ctx_factory)

    if config.channels.telegram and config.channels.telegram.token:
        from infra.channels.telegram_channel import TelegramChannel

        tg = config.channels.telegram
        host.add(TelegramChannel(
            token=tg.token,
            bus=bus,
            session_manager=session_manager,
            allow_from=tg.allow_from,
            command_catalog_provider=command_catalog_provider,
            event_bus=event_bus,
            interrupt_controller=interrupt_controller,
            channel_name=tg.channel_name,
        ))

    if config.channels.qq and config.channels.qq.bot_uin:
        from infra.channels.qq_channel import QQChannel

        qq = config.channels.qq
        host.add(QQChannel(
            bot_uin=qq.bot_uin,
            bus=bus,
            session_manager=session_manager,
            allow_from=qq.allow_from,
            groups=qq.groups,
            websocket_open_timeout_seconds=qq.websocket_open_timeout_seconds,
            http_requester=http_resources.external_default,
            event_bus=event_bus,
            interrupt_controller=interrupt_controller,
        ))

    for channel in extra_channels or []:
        host.add(channel)

    return host

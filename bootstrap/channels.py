from __future__ import annotations

import logging

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
    bot_commands: list[tuple[str, str]] | None = None,
    interrupt_controller: InterruptController | None = None,
    plugin_channels: list[Channel] | None = None,
) -> tuple[object, ChannelHost]:
    from infra.channels.ipc_server import IPCServerChannel

    ipc = IPCServerChannel(
        bus,
        config.channels.socket,
        default_session_key=config.channels.cli_session_key,
    )
    await ipc.start()
    print(f"Agent 已启动  |  CLI 连接地址: {config.channels.socket}")

    attachment_store = AttachmentStore()

    def _ctx_factory(channel: Channel) -> ChannelContext:
        return ChannelContext(
            bus=bus,
            session_manager=session_manager,
            event_bus=event_bus,
            push_tool=push_tool,
            attachment_store=attachment_store,
            http_resources=http_resources,
            interrupt_controller=interrupt_controller,
            bot_commands=bot_commands or [],
            log=logging.getLogger(f"channels.{channel.name}"),
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
            bot_commands=bot_commands,
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

    for channel in plugin_channels or []:
        host.add(channel)

    return ipc, host

from __future__ import annotations

from typing import Any

from agent.config_models import Config
from agent.looping.interrupt import InterruptController
from agent.tools.message_push import MessagePushTool
from bus.event_bus import EventBus
from bus.queue import MessageBus
from core.net.http import SharedHttpResources
from session.manager import SessionManager


async def start_channels(
    config: Config,
    *,
    bus: MessageBus,
    session_manager: SessionManager,
    push_tool: MessagePushTool,
    http_resources: SharedHttpResources,
    event_bus: EventBus,
    interrupt_controller: InterruptController | None = None,
) -> tuple[Any, Any, Any]:
    from infra.channels.ipc_server import IPCServerChannel

    ipc = IPCServerChannel(bus, config.channels.socket)
    await ipc.start()
    print(f"Agent 已启动  |  CLI 连接地址: {config.channels.socket}")

    tg_channel = None
    if config.channels.telegram and config.channels.telegram.token:
        from infra.channels.telegram_channel import TelegramChannel

        tg = config.channels.telegram
        tg_channel = TelegramChannel(
            token=tg.token,
            bus=bus,
            session_manager=session_manager,
            allow_from=tg.allow_from,
            event_bus=event_bus,
            interrupt_controller=interrupt_controller,
        )
        await tg_channel.start()
        push_tool.register_channel(
            "telegram",
            text=tg_channel.send,
            stream_text=tg_channel.send_stream,
            file=tg_channel.send_file,
            image=tg_channel.send_image,
        )
        print("Telegram Bot 已启动")

    qq_channel = None
    if config.channels.qq and config.channels.qq.bot_uin:
        from infra.channels.qq_channel import QQChannel

        qq = config.channels.qq
        qq_channel = QQChannel(
            bot_uin=qq.bot_uin,
            bus=bus,
            session_manager=session_manager,
            allow_from=qq.allow_from,
            groups=qq.groups,
            http_requester=http_resources.external_default,
            interrupt_controller=interrupt_controller,
        )
        await qq_channel.start()
        push_tool.register_channel(
            "qq",
            text=qq_channel.send,
            file=qq_channel.send_file,
            image=qq_channel.send_image,
        )
        print(f"QQ Bot 已启动  |  QQ 号: {qq.bot_uin}")

    return ipc, tg_channel, qq_channel

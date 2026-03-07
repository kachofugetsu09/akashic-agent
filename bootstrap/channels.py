from __future__ import annotations

from typing import Any

from agent.config_models import Config
from agent.tools.message_push import MessagePushTool
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
) -> tuple[Any, Any, Any]:
    from channels.ipc_server import IPCServerChannel

    ipc = IPCServerChannel(bus, config.channels.socket)
    await ipc.start()
    print(f"Agent 已启动  |  CLI 连接地址: {config.channels.socket}")

    tg_channel = None
    if config.channels.telegram:
        from channels.telegram_channel import TelegramChannel

        tg = config.channels.telegram
        tg_channel = TelegramChannel(
            token=tg.token,
            bus=bus,
            session_manager=session_manager,
            allow_from=tg.allow_from,
        )
        await tg_channel.start()
        push_tool.register_channel(
            "telegram",
            text=tg_channel.send,
            file=tg_channel.send_file,
            image=tg_channel.send_image,
        )
        print("Telegram Bot 已启动")

    qq_channel = None
    if config.channels.qq:
        from channels.qq_channel import QQChannel

        qq = config.channels.qq
        qq_channel = QQChannel(
            bot_uin=qq.bot_uin,
            bus=bus,
            session_manager=session_manager,
            allow_from=qq.allow_from,
            groups=qq.groups,
            http_requester=http_resources.external_default,
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

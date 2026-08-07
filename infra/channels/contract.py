from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from agent.looping.interrupt import InterruptController
from agent.tools.message_push import MessagePushTool
from bus.event_bus import EventBus
from bus.queue import MessageBus
from core.net.http import SharedHttpResources
from infra.channels.base import AttachmentStore
from session.manager import SessionManager


class Channel(Protocol):
    """Own one ingress resource and acknowledge acquisition/release by returning."""

    name: str

    # 返回只表示入口 ownership 已取得且 channel ready；失败必须抛错。
    async def start(self, ctx: ChannelContext) -> None: ...

    # 返回只表示新 ingress 已停止、在途工作已收束且 ownership 已释放。
    async def stop(self) -> None: ...


@dataclass
class ChannelContext:
    bus: MessageBus
    session_manager: SessionManager
    event_bus: EventBus
    push_tool: MessagePushTool
    attachment_store: AttachmentStore
    http_resources: SharedHttpResources
    interrupt_controller: InterruptController | None
    mobile_bot_commands: list[tuple[str, str]]
    log: logging.Logger

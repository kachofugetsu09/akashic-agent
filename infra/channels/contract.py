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
    name: str

    async def start(self, ctx: ChannelContext) -> None: ...
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
    bot_commands: list[tuple[str, str]]
    log: logging.Logger

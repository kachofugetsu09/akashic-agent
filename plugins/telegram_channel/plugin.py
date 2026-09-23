from __future__ import annotations

from agent.plugin_composition import (
    CHANNELS,
    ChannelCapability,
    ChannelDefinition,
    Context,
    InboundIdentity,
    ServiceKey,
)

from functools import partial
from agent.plugin_composition.credentials import CREDENTIALS
from agent.plugin_composition.channels import CHANNEL_INPUT, RawInbound
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugin_composition.messages import MessageReader
from typing import Protocol


class SourceState(Protocol):
    def needs_reply(self, reader: MessageReader, source: str) -> bool: ...


SOURCE_STATE = ServiceKey[SourceState]("source.session.v1")

from .channel import TelegramChannelAdapter, build_telegram_channel
from .config import TelegramChannelConfig

api_version = 3
name = "telegram_channel"
version = "3.0.0"
desc = "Telegram inbound and outbound v3 channel adapter"
author = "Akashic"
inject = (CHANNELS, CHANNEL_INPUT, CREDENTIALS, MESSAGE_CATALOG, SOURCE_STATE)
Config = TelegramChannelConfig


async def apply(ctx: Context) -> None:
    """Register Telegram only when its ordinary plugin config enables it."""
    config = Config.model_validate(ctx.config)

    if not config.enabled:
        return
    async def interrupt(raw: RawInbound) -> bool:
        """先通过来源提交 pause 并等待旧工作，再由原 binding 发送 ack。"""
        session_id = f"telegram:{raw.message.chat_id}"
        reader = ctx.require(MESSAGE_CATALOG).reader(session_id)
        if reader.head(source="conversation") < 0:
            return False
        pending = ctx.require(SOURCE_STATE).needs_reply(reader, "conversation")
        await ctx.require(CHANNEL_INPUT)(session_id, raw.message_id, raw.message)
        return pending

    await ctx.require(CHANNELS).register(
        ctx,
        ChannelDefinition(
            name="telegram",
            capabilities=frozenset(
                {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND, ChannelCapability.CONTROL}
            ),
            factory=partial(build_telegram_channel, create_client=partial(ctx.require(CREDENTIALS).create, ctx)),
            config=ctx.config,
            interrupt=interrupt,
            inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
        ),
    )


__all__ = [
    "Config",
    "TelegramChannelAdapter",
    "api_version",
    "apply",
    "author",
    "build_telegram_channel",
    "desc",
    "inject",
    "name",
    "version",
]

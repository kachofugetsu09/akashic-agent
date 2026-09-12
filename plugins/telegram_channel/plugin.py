from __future__ import annotations

from agent.plugin_composition import (
    CHANNELS,
    ChannelCapability,
    ChannelDefinition,
    Context,
    InboundIdentity,
)

from .channel import TelegramChannelAdapter, build_telegram_channel
from .config import TelegramChannelConfig

api_version = 3
name = "telegram_channel"
version = "3.0.0"
desc = "Telegram inbound and outbound v3 channel adapter"
author = "Akashic"
inject = (CHANNELS,)
Config = TelegramChannelConfig


async def apply(ctx: Context, config: TelegramChannelConfig) -> None:
    """Register Telegram only when its ordinary plugin config enables it."""

    if not isinstance(config, TelegramChannelConfig):
        raise TypeError("Telegram channel config 必须通过 Config 校验")
    if not config.enabled:
        return
    await ctx.require(CHANNELS).register(
        ctx,
        ChannelDefinition(
            name="telegram",
            capabilities=frozenset(
                {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND, ChannelCapability.CONTROL}
            ),
            factory_export="build_telegram_channel",
            inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
            credential_paths=("token",),
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

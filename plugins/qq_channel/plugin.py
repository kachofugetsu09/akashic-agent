from __future__ import annotations

from agent.plugin_composition import (
    CHANNELS,
    ChannelCapability,
    ChannelDefinition,
    Context,
    InboundIdentity,
)

from agent.plugin_composition.channels import CHANNEL_INPUT

from .channel import QQChannelAdapter, build_qq_channel
from .config import QQChannelConfig

api_version = 3
name = "qq_channel"
version = "3.0.0"
desc = "NapCat OneBot QQ inbound and outbound v3 channel adapter"
author = "Akashic"
inject = (CHANNELS, CHANNEL_INPUT)
Config = QQChannelConfig


async def apply(ctx: Context) -> None:
    """Register the legacy QQ protocol only when its plugin config enables it."""
    config = Config.model_validate(ctx.config)

    if not config.enabled:
        return
    await ctx.require(CHANNELS).register(
        ctx,
        ChannelDefinition(
            name="qq",
            capabilities=frozenset({ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}),
            factory=build_qq_channel,
            config=ctx.config,
            inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
        ),
    )


__all__ = [
    "Config",
    "QQChannelAdapter",
    "api_version",
    "apply",
    "author",
    "build_qq_channel",
    "desc",
    "inject",
    "name",
    "version",
]

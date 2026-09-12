from __future__ import annotations

from agent.plugin_composition import (
    CHANNELS,
    ChannelCapability,
    ChannelDefinition,
    Context,
    InboundIdentity,
)

from .channel import QQChannelAdapter, build_qq_channel
from .config import QQChannelConfig

api_version = 3
name = "qq_channel"
version = "3.0.0"
desc = "NapCat OneBot QQ inbound and outbound v3 channel adapter"
author = "Akashic"
inject = (CHANNELS,)
Config = QQChannelConfig


async def apply(ctx: Context, config: QQChannelConfig) -> None:
    """Register the legacy QQ protocol only when its plugin config enables it."""

    if not isinstance(config, QQChannelConfig):
        raise TypeError("QQ channel config 必须通过 Config 校验")
    if not config.enabled:
        return
    await ctx.require(CHANNELS).register(
        ctx,
        ChannelDefinition(
            name="qq",
            capabilities=frozenset({ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}),
            factory_export="build_qq_channel",
            inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
            credential_paths=(),
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

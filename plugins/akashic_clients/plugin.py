from __future__ import annotations

from agent.plugin_composition import (
    CHANNELS,
    ChannelCapability,
    ChannelDefinition,
    Context,
    InboundIdentity,
)

from .capabilities import CLIENT_CAPABILITIES
from .channel import build_akashic_channel_factory
from agent.plugin_composition.channels import CHANNEL_INPUT
from .config import AkashicClientsConfig

api_version = 3
name = "akashic_clients"
version = "1.0.0"
desc = "Web and Mobile Akashic client channel"
author = "Akashic"
# Every dependency is a separate composition capability.  In particular,
# there is no client-wide service bus for Core to assemble.
inject = (CHANNELS, CHANNEL_INPUT, *CLIENT_CAPABILITIES)
Config = AkashicClientsConfig


async def apply(ctx: Context) -> None:
    """Register one ordinary channel over independent host capabilities."""
    config = Config.model_validate(ctx.config)

    if not config.enabled:
        return

    await ctx.require(CHANNELS).register(
        ctx,
        ChannelDefinition(
            name="akashic",
            capabilities=frozenset(
                {
                    ChannelCapability.INBOUND,
                    ChannelCapability.DURABLE_INBOUND,
                    ChannelCapability.OUTBOUND,
                    ChannelCapability.TURN_STREAM,
                }
            ),
            factory=build_akashic_channel_factory(config, ctx.runtime.workspace),
            inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
        ),
    )


__all__ = [
    "Config",
    "api_version",
    "apply",
    "author",
    "desc",
    "inject",
    "name",
    "version",
]

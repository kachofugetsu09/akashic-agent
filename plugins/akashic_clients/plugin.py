from __future__ import annotations

from agent.plugin_composition import (
    CHANNELS,
    ChannelCapability,
    ChannelDefinition,
    Context,
    InboundIdentity,
)

from .capabilities import CLIENT_CAPABILITIES
from .channel import build_akashic_channel, register_generation, unregister_generation
from .config import AkashicClientsConfig

api_version = 3
name = "akashic_clients"
version = "1.0.0"
desc = "Web and Mobile Akashic client channel"
author = "Akashic"
# Every dependency is a separate composition capability.  In particular,
# there is no client-wide service bus for Core to assemble.
inject = (CHANNELS, *CLIENT_CAPABILITIES)
Config = AkashicClientsConfig


async def apply(ctx: Context, config: AkashicClientsConfig) -> None:
    """Register one ordinary channel over independent host capabilities."""

    if not isinstance(config, AkashicClientsConfig):
        raise TypeError("akashic_clients config 必须通过 Config 校验")
    if not config.enabled:
        return

    # Context.generation_id identifies the whole composition Root.  The
    # channel host resolves factories with the plugin generation identity, so
    # bind this state to the exact runtime generation owned by this plugin.
    generation_id = ctx.runtime.generation_id
    register_generation(generation_id, config, ctx.runtime.workspace)

    async def cleanup() -> None:
        unregister_generation(generation_id)

    _ = await ctx.effect(lambda: cleanup, label="akashic-clients-generation")
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
            factory_export="build_akashic_channel",
            inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
            credential_paths=(),
        ),
    )


__all__ = [
    "Config",
    "api_version",
    "apply",
    "author",
    "build_akashic_channel",
    "desc",
    "inject",
    "name",
    "version",
]

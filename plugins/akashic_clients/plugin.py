from __future__ import annotations

from agent.plugin_composition import (
    CHANNELS,
    ChannelCapability,
    ChannelDefinition,
    Context,
    InboundIdentity,
)

from .channel import build_akashic_channel, register_generation, unregister_generation
from .config import AkashicClientsConfig
from .services import CLIENT_SERVICES

api_version = 3
name = "akashic_clients"
version = "1.0.0"
desc = "Web and Mobile Akashic client channel"
author = "Akashic"
inject = (CHANNELS, CLIENT_SERVICES)
Config = AkashicClientsConfig


async def apply(ctx: Context, config: AkashicClientsConfig) -> None:
    """Register one ordinary channel over the host-provided client services."""

    if not isinstance(config, AkashicClientsConfig):
        raise TypeError("akashic_clients config 必须通过 Config 校验")
    if not config.enabled:
        return

    services = ctx.require(CLIENT_SERVICES)
    register_generation(ctx.generation_id, config, services)

    async def cleanup() -> None:
        unregister_generation(ctx.generation_id)

    _ = await ctx.effect(lambda: cleanup, label="akashic-clients-generation")
    await ctx.require(CHANNELS).register(
        ctx,
        ChannelDefinition(
            name="akashic",
            capabilities=frozenset(
                {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}
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

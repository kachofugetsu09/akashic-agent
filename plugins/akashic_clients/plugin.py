from __future__ import annotations

from agent.plugin_composition import Context
from plugins.channels.contract import CHANNELS, ChannelCapability, ChannelDefinition, InboundIdentity

from .capabilities import CLIENT_CAPABILITIES, INSPECTION_RPC_KEYS, REPLY_STATUS, MCP_DETAIL, MCP_SERVERS
from plugins.models.contract import MODEL_CALL_STATS
from .channel import build_akashic_channel_factory
from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT
from .config import AkashicClientsConfig
from .navigation import NavigationPreferences
from plugins.ledger.contract import OWNER_STATE

api_version = 3
name = "akashic_clients"
version = "1.0.0"
desc = "Web Akashic client channel"
author = "Akashic"
# Every dependency is a separate composition capability.  In particular,
# there is no client-wide service bus for Core to assemble.
inject = (CHANNELS, CHANNEL_INPUT, OWNER_STATE, *CLIENT_CAPABILITIES)
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
            factory=build_akashic_channel_factory(
                ctx, config, ctx.runtime.workspace,
                NavigationPreferences(lambda: ctx.require(OWNER_STATE).open(ctx)),
            ),
            inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
            optional_services=frozenset((*INSPECTION_RPC_KEYS, MODEL_CALL_STATS, REPLY_STATUS, MCP_DETAIL, MCP_SERVERS)),
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

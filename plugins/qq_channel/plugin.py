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
function_inject = (CHANNELS, CHANNEL_INPUT)
Config = QQChannelConfig


async def run(ctx: Context) -> None:
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


from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG

inject = (PLUGIN_CONFIG, RUNTIME_CATALOG)


async def apply(ctx: Context) -> None:
    """设置入口常驻，业务依赖只影响功能分支。"""
    from .settings import mount
    function = await ctx.inject(function_inject, run, name="function")
    await mount(ctx, Config, function)

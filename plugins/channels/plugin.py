"""显式选择的 Channel provider。"""
from agent.plugin_composition import Context
from agent.plugin_composition.host import HOST_INFO
from agent.plugin_composition.channels import CHANNELS
from agent.plugin_composition.channel_io import (
    INPUT_CUSTODY, CHANNEL_IDENTITY, CHANNEL_ATTACHMENT_IMPORT, CHANNEL_ATTACHMENT_READ,
)
from .provider import PluginChannels

api_version = 3
name = "channels"
version = "1.0.0"
desc = "连接、接纳、原 binding 发送与持久输入恢复"
inject = (HOST_INFO, INPUT_CUSTODY, CHANNEL_IDENTITY, CHANNEL_ATTACHMENT_IMPORT, CHANNEL_ATTACHMENT_READ)


async def apply(ctx: Context) -> None:
    """目录和连接生命周期由普通 provider 实例拥有。"""
    channels = PluginChannels(ctx)
    await ctx.provide(CHANNELS, channels)

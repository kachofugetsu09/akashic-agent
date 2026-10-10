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
    from plugins.onboarding.contract import ONBOARDING, Ability, PreviewLine
    async def contribute(child: Context):
        await child.require(ONBOARDING).group(child, "channels", "聊天渠道", Ability(
            pitch="在聊天软件里找到它",
            benefit="在手机上直接给 Akashic 发消息，不用打开网页；主动消息也会发到这里。",
            preview=(PreviewLine("你 · Telegram", "帮我记一下，周五下午三点看牙。"),
                     PreviewLine("Akashic", "记下了：周五 15:00 看牙。前一天晚上提醒你？")),
        ))
    await ctx.inject((ONBOARDING,), contribute, name="onboarding")

    """目录和连接生命周期由普通 provider 实例拥有。"""
    channels = PluginChannels(ctx)
    await ctx.provide(CHANNELS, channels)

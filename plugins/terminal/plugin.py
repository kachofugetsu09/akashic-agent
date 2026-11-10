"""用户交互终端：会话里的 shell 由执行主机拥有，本插件按会话命名并提供浏览器入口。"""
from agent.plugin_composition import Context
from plugins.host_execution.contract import TERMINALS
from plugins.ui.contract import UI

from . import dashboard
from .sessions import SESSION_TERMINALS, SessionTerminals

api_version = 3
name = "terminal"
version = "1.0.0"
desc = "工作台里用户自己敲命令的终端；PTY 归执行主机，本插件只按会话命名"
inject = (TERMINALS,)


async def apply(ctx: Context) -> None:
    await ctx.provide(SESSION_TERMINALS, SessionTerminals(ctx))

    # UI 缺席时只是没有浏览器入口，终端服务照常可用。
    async def register_ui(child: Context) -> None:
        _ = await child.require(UI).register(
            child, web="web_module.js", dashboard=lambda: dashboard,
            requires=("conversation.tools.v2",),
            contract_digests={
                "conversation.tools.v2": "dcbf5db629d4fa6fed4019ad91663f9af94c5b9a02f9e67c94b0745e40f19049",
            },
        )

    _ = await ctx.inject((UI, SESSION_TERMINALS), register_ui, name="terminal-ui")

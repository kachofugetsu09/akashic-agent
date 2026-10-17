"""工作台“网站”标签：打开 show_site 发布的快照；快照与隔离入口归 sites 插件。"""
from agent.plugin_composition import Context
from plugins.ui.contract import UI

api_version = 3
name = "sites_tab"
version = "1.0.0"
desc = "在对话工作台的“网站”标签里打开 Agent 发布的网页，切换手机、平板与自适应宽度"
author = "Akashic Core"
inject = (UI,)


async def apply(ctx: Context) -> None:
    # 带 Web 模块的插件，其 Dashboard 路由都要求 Web 身份头，iframe 导航带不上；
    # 所以隔离入口 /sites/frame.html 留在没有 Web 模块的 sites 插件里，本插件只提供标签。
    _ = await ctx.require(UI).register(
        ctx, web="web_module.js",
        requires=("conversation.tools.v2",),
        contract_digests={
            "conversation.tools.v2": "dcbf5db629d4fa6fed4019ad91663f9af94c5b9a02f9e67c94b0745e40f19049",
        },
    )

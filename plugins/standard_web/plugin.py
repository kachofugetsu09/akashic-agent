from agent.plugin_composition import Context, ServiceKey
from plugins.tools.plugin import TOOLS, ToolView

from .web import register_web

api_version = 3
name = "standard_web"
version = "1.0.0"
desc = "提供普通 Web 搜索与读取工具"
inject = (TOOLS,)

STANDARD_WEB_TOOLS = ServiceKey[ToolView]("standard-web.tools.v1")


async def apply(ctx: Context, config: object) -> None:
    _ = config
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, description=desc)
    refs = await register_web(ctx)
    _ = await ctx.provide(STANDARD_WEB_TOOLS, catalog.view(*refs))

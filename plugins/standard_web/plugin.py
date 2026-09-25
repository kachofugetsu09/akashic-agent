from agent.plugin_composition import Context

from ._tool_boundary import TOOLS
from .web import register_web

api_version = 3
name = "standard_web"
version = "1.0.0"
desc = "提供普通 Web 搜索与读取工具"
inject = (TOOLS,)



async def apply(ctx: Context) -> None:
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)
    await register_web(ctx)

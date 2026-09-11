from agent.plugin_contracts.plugin_tools import STANDARD_WEB_TOOLS  # noqa: F401  (再导出)
from agent.plugin_composition import Context, ServiceKey
from agent.plugin_contracts.tools import TOOLS, ToolView

from .web import register_web

api_version = 3
name = "standard_web"
version = "1.0.0"
desc = "提供普通 Web 搜索与读取工具"
inject = (TOOLS,)



async def apply(ctx: Context, config: object) -> None:
    _ = config
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)
    refs = await register_web(ctx)
    _ = await ctx.provide(STANDARD_WEB_TOOLS, catalog.view(*refs))

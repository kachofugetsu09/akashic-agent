from agent.plugin_composition import Context

from ._tool_boundary import TOOLS
from .contract import WEB_PROVIDERS
from .web import Providers, WebConfig, register_web

api_version = 3
name = "standard_web"
version = "1.1.0"
desc = "提供 Web 搜索与读取工具；端点由各提供方插件负责"
inject = (TOOLS,)



async def apply(ctx: Context) -> None:
    providers = Providers(ctx, WebConfig.model_validate(ctx.config))
    _ = await ctx.provide(WEB_PROVIDERS, providers)
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)
    _ = await register_web(ctx, providers)

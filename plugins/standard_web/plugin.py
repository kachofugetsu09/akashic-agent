from importlib import import_module

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from plugins.ui.contract import Configuration, ModelSettings, mount_settings

from ._tool_boundary import TOOLS
from .contract import WEB_PROVIDERS
from .web import Providers, WebConfig, register_web

api_version = 3
name = "standard_web"
version = "1.2.0"
desc = "提供 Web 搜索与读取工具；端点由各提供方插件负责"
inject = (TOOLS, PLUGIN_CONFIG)

SETTINGS = ServiceKey[Configuration]("standard_web.settings.v1")
FIELDS = ("search_provider", "fetch_provider", "default_results", "default_fetch_timeout")


async def apply(ctx: Context) -> None:
    config = WebConfig.model_validate(ctx.config)
    providers = Providers(ctx, config)
    _ = await ctx.provide(WEB_PROVIDERS, providers)
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)
    _ = await register_web(ctx, providers, config)

    # 选项随已启用的提供方变化；所选提供方不可用时直接说明原因。
    async def status() -> dict[str, object]:
        problems = [problem for kind in ("search", "fetch") if isinstance(problem := providers.select(kind), str)]
        return {"ready": not problems, "reason": "；".join(problems),
                "choices": {"search_provider": providers.choices("search"), "fetch_provider": providers.choices("fetch")}}

    settings = ModelSettings(ctx, WebConfig, FIELDS, status=status)
    await mount_settings(ctx, SETTINGS, settings, lambda: import_module(".dashboard", __package__))

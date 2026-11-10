"""Exa 搜索提供方：端点、API key 与搜索选项归本插件，工具形状归 standard_web。"""
from agent.plugin_composition import CREDENTIALS, Context
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.standard_web.contract import WEB_PROVIDERS

from .search import ExaConfig, ExaSearch

api_version = 3
name = "web_search_exa"
version = "1.0.0"
desc = "经 Exa 搜索互联网；可配置自建或代理端点与 API key"
inject = (PLUGIN_CONFIG, RUNTIME_CATALOG)


async def run(ctx: Context) -> None:
    config = ExaConfig.model_validate(ctx.config)

    # 每次请求时读取凭据，不在插件内常驻明文 key。
    async def read_key() -> str | None:
        if config.api_key is None:
            return None
        async with ctx.require(CREDENTIALS).open(ctx, {"api_key": config.api_key}) as client:
            return client.credential(config.api_key)

    _ = await ctx.require(WEB_PROVIDERS).register_search(ctx, ExaSearch(config, read_key), label="Exa")


async def apply(ctx: Context) -> None:
    """设置入口常驻；standard_web 未启用时只有提供方分支等待。"""
    from .settings import mount
    function = await ctx.inject((WEB_PROVIDERS, CREDENTIALS), run, name="function")
    await mount(ctx, function)

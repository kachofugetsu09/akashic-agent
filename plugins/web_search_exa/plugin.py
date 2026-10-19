"""Exa 搜索提供方：端点、API key 与搜索选项归本插件，工具形状归 standard_web。"""
from agent.plugin_composition import CREDENTIALS, Context
from plugins.standard_web.contract import WEB_PROVIDERS

from .search import ExaConfig, ExaSearch

api_version = 3
name = "web_search_exa"
version = "1.0.0"
desc = "经 Exa 搜索互联网；可配置自建或代理端点与 API key"
inject = (WEB_PROVIDERS, CREDENTIALS)


async def apply(ctx: Context) -> None:
    config = ExaConfig.model_validate(ctx.config)

    # 每次请求时读取凭据，轮换 key 不需要重启插件。
    async def read_key() -> str | None:
        if config.api_key is None:
            return None
        async with ctx.require(CREDENTIALS).open(ctx, {"api_key": config.api_key}) as client:
            return client.credential(config.api_key)

    _ = await ctx.require(WEB_PROVIDERS).register_search(ctx, ExaSearch(config, read_key))

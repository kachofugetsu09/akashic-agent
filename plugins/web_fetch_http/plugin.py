"""HTTP 读取提供方：直接抓取网页正文；工具形状归 standard_web，上限、请求头与代理归本插件设置。"""
from importlib import import_module

import httpx

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from core.net.http import HttpRequester, RequestBudget, RetryPolicy
from plugins.standard_web.contract import WEB_PROVIDERS, FetchRequest
from plugins.ui.contract import Configuration, ModelSettings, mount_settings

from .fetch import FetchConfig, HttpFetch

api_version = 3
name = "web_fetch_http"
version = "1.1.0"
desc = "直接通过 HTTP(S) 读取网页正文；可配置大小上限、请求头与代理"
inject = (WEB_PROVIDERS, PLUGIN_CONFIG)

SETTINGS = ServiceKey[Configuration]("web_fetch_http.settings.v1")
FIELDS = ("max_response_mb", "max_text_chars", "proxy", "user_agent", "accept_language")


class ScopedFetch:
    """每次读取自建客户端并在结束时关闭，不跨调用持有连接。"""

    def __init__(self, config: FetchConfig | None = None):
        self._config = config or FetchConfig()

    async def fetch(self, request: FetchRequest) -> dict[str, object]:
        proxy = str(self._config.proxy) if self._config.proxy else None
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=20, max_keepalive_connections=10), proxy=proxy) as client:
            requester = HttpRequester(client, RetryPolicy(), 30.0, RequestBudget(total_timeout_s=45.0))
            return await HttpFetch(requester, self._config).fetch(request)


async def apply(ctx: Context) -> None:
    config = FetchConfig.model_validate(ctx.config)
    _ = await ctx.require(WEB_PROVIDERS).register_fetch(ctx, ScopedFetch(config), label="直接 HTTP")
    await mount_settings(ctx, SETTINGS, ModelSettings(ctx, FetchConfig, FIELDS), lambda: import_module(".dashboard", __package__))

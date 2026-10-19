"""HTTP 读取提供方：直接抓取网页正文；工具形状归 standard_web。"""
import httpx

from agent.plugin_composition import Context
from core.net.http import HttpRequester, RequestBudget, RetryPolicy
from plugins.standard_web.contract import WEB_PROVIDERS, FetchRequest

from .fetch import HttpFetch

api_version = 3
name = "web_fetch_http"
version = "1.0.0"
desc = "直接通过 HTTP(S) 读取网页正文"
inject = (WEB_PROVIDERS,)


class ScopedFetch:
    """每次读取自建客户端并在结束时关闭，不跨调用持有连接。"""

    async def fetch(self, request: FetchRequest) -> dict[str, object]:
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)) as client:
            requester = HttpRequester(client, RetryPolicy(), 30.0, RequestBudget(total_timeout_s=45.0))
            return await HttpFetch(requester).fetch(request)


async def apply(ctx: Context) -> None:
    _ = await ctx.require(WEB_PROVIDERS).register_fetch(ctx, ScopedFetch())

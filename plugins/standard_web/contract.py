"""Web 搜索与读取的提供方合同：standard_web 拥有工具与选择，提供方插件各自拥有端点、凭据和协议。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey

FetchFormat = Literal["text", "markdown", "html"]


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    num_results: int


@dataclass(frozen=True, slots=True)
class FetchRequest:
    url: str
    format: FetchFormat
    timeout_s: int


class WebSearchProvider(Protocol):
    """返回 JSON 对象；失败用 {"error": 原因} 表达，不抛出业务失败。"""

    async def search(self, request: SearchRequest) -> dict[str, object]: ...


class WebFetchProvider(Protocol):
    """返回 JSON 对象；失败用 {"error": 原因} 表达，不抛出业务失败。"""

    async def fetch(self, request: FetchRequest) -> dict[str, object]: ...


class WebProviders(Protocol):
    """提供方以自己的插件身份登记；同一插件每类最多一个，注销随 Effect。"""

    async def register_search(self, ctx: Context, provider: WebSearchProvider) -> Effect: ...

    async def register_fetch(self, ctx: Context, provider: WebFetchProvider) -> Effect: ...


WEB_PROVIDERS = ServiceKey[WebProviders]("standard_web.providers.v1")

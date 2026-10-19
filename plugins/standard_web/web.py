from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
import json
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from agent.plugin_composition import Context, Effect
from plugins.tools.contract import (
    normalize_tool_parameters,
    validate_tool_parameters,
)
from plugins.ledger.contract import ContentPart
from plugins.ledger.contract import json_value

from ._tool_boundary import CallSource, TOOLS, ToolRef, ToolResultValue
from .contract import (
    WEB_PROVIDERS, FetchFormat, FetchRequest, SearchRequest, WebFetchProvider, WebSearchProvider,
)

Kind = Literal["search", "fetch"]
_KIND_LABEL = {"search": "搜索", "fetch": "读取"}
_DEFAULT_RESULTS = 8
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120


class WebConfig(BaseModel):
    """未配置时只有一个可用提供方就用它；多个时要求明确选择。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    search_provider: str | None = None
    fetch_provider: str | None = None


class Providers:
    """按插件身份保存已启用的提供方；调用期间持有提供方的运行作用域。"""

    def __init__(self, ctx: Context, config: WebConfig):
        self._ctx = ctx
        self._config = config
        self._entries: dict[tuple[Kind, str], tuple[Context, WebSearchProvider | WebFetchProvider]] = {}

    async def register_search(self, ctx: Context, provider: WebSearchProvider) -> Effect:
        return await self._register(ctx, "search", provider)

    async def register_fetch(self, ctx: Context, provider: WebFetchProvider) -> Effect:
        return await self._register(ctx, "fetch", provider)

    # 身份取自贡献方 Context，不接受调用方指定其他插件。
    async def _register(self, ctx: Context, kind: Kind, provider: WebSearchProvider | WebFetchProvider) -> Effect:
        if ctx.root_instance_token is not self._ctx.root_instance_token or ctx.require(WEB_PROVIDERS) is not self:
            raise ValueError("Web 提供方必须登记到本组合的 standard_web")
        key = (kind, ctx.runtime.plugin_id)

        def setup() -> Callable[[], None]:
            if key in self._entries:
                raise ValueError(f"Web {_KIND_LABEL[kind]}提供方重复登记: {key[1]}")
            self._entries[key] = (ctx, provider)

            def cleanup() -> None:
                del self._entries[key]

            return cleanup

        return await ctx.effect(setup, label=f"web-{kind}:{key[1]}")

    # 1. 配置了就只用它；2. 未配置时恰好一个才自动选择，否则说明如何选。
    def select(self, kind: Kind) -> tuple[Context, WebSearchProvider | WebFetchProvider] | str:
        wanted = self._config.search_provider if kind == "search" else self._config.fetch_provider
        entries = {plugin: entry for (item, plugin), entry in self._entries.items() if item == kind}
        label = _KIND_LABEL[kind]
        if wanted:
            found = [entry for plugin, entry in entries.items() if wanted in (plugin, plugin.split("@")[0])]
            return found[0] if found else f"配置的 Web {label}提供方没有启用: {wanted}"
        if not entries:
            return f"没有启用的 Web {label}提供方，请启用一个提供方插件"
        if len(entries) > 1:
            names = "、".join(sorted(plugin.split("@")[0] for plugin in entries))
            return f"有多个 Web {label}提供方（{names}），请在 standard_web 配置 {kind}_provider 选择一个"
        return next(iter(entries.values()))

    async def search(self, request: SearchRequest) -> dict[str, object]:
        selected = self.select("search")
        if isinstance(selected, str):
            return {"error": selected, "query": request.query}
        owner, provider = selected
        async with owner.runtime_scope():
            return await cast(WebSearchProvider, provider).search(request)

    async def fetch(self, request: FetchRequest) -> dict[str, object]:
        selected = self.select("fetch")
        if isinstance(selected, str):
            return {"error": selected, "url": request.url}
        owner, provider = selected
        async with owner.runtime_scope():
            return await cast(WebFetchProvider, provider).fetch(request)


SEARCH_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "搜索关键词"},
        "num_results": {
            "type": "integer", "minimum": 1, "maximum": 20,
            "description": f"返回结果数量，默认 {_DEFAULT_RESULTS}，最大 20",
        },
    },
    "required": ["query"],
}

FETCH_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "要抓取的完整 URL，必须以 http:// 或 https:// 开头"},
        "format": {
            "type": "string", "enum": ["text", "markdown", "html"],
            "description": "返回格式：text 纯文本 / markdown 转换后的 Markdown / html 原始 HTML。默认 markdown",
        },
        "timeout": {
            "type": "integer", "minimum": 1, "maximum": _MAX_TIMEOUT,
            "description": f"超时秒数，默认 {_DEFAULT_TIMEOUT}，最大 {_MAX_TIMEOUT}",
        },
    },
    "required": ["url"],
}


class WebTool:
    idempotent = False

    def __init__(self, parameters: dict[str, object], run: Callable[[dict[str, Any]], Awaitable[dict[str, object]]]):
        self._parameters = parameters
        self._run = run

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        raw = cast(dict[str, Any], json_value(arguments))
        errors = validate_tool_parameters(raw, schema=normalize_tool_parameters(self._parameters))
        if errors:
            return "; ".join(errors)
        return raw

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        value = await self._run(cast(dict[str, Any], json_value(arguments)))
        if not isinstance(value, dict):
            raise TypeError("Web 提供方结果必须是 JSON 对象")
        if "error" in value and not isinstance(value["error"], str):
            raise TypeError("Web 提供方 error 必须是字符串")
        text = json.dumps(json_value(value), ensure_ascii=False)
        return ToolResultValue("error" if "error" in value else "success", (ContentPart("text", text),))

    async def query(self, key: str) -> ToolResultValue | None:
        return None


async def register_web(ctx: Context, providers: Providers) -> tuple[ToolRef, ...]:
    """工具只负责参数与结果形状；实际请求交给当前选中的提供方。"""
    async def search(arguments: dict[str, Any]) -> dict[str, object]:
        count = min(int(arguments.get("num_results", _DEFAULT_RESULTS)), 20)
        return await providers.search(SearchRequest(arguments["query"], count))

    async def fetch(arguments: dict[str, Any]) -> dict[str, object]:
        timeout = min(int(arguments.get("timeout", _DEFAULT_TIMEOUT)), _MAX_TIMEOUT)
        fmt = cast(FetchFormat, arguments.get("format", "markdown"))
        return await providers.fetch(FetchRequest(arguments["url"], fmt, timeout))

    tools = (
        ("web_search", "用关键词搜索互联网，返回最新的搜索结果（标题 + 摘要 + URL）。"
         "适合查询时效性信息：新闻、产品发布、价格、人物动态等。拿到 URL 后可用 web_fetch 获取完整内容。",
         SEARCH_PARAMETERS, search),
        ("web_fetch", "抓取指定 URL 的内容并返回。支持 text（纯文本）、markdown（转换后的 Markdown，默认）、"
         "html（原始 HTML）三种格式。仅支持 HTTP/HTTPS。",
         FETCH_PARAMETERS, fetch),
    )
    refs: list[ToolRef] = []
    for name, description, parameters, run in tools:
        @asynccontextmanager
        async def open_tool(_state: Mapping[str, object], parameters: dict[str, object] = parameters,
                            run: Callable[[dict[str, Any]], Awaitable[dict[str, object]]] = run) -> AsyncGenerator[WebTool]:
            yield WebTool(parameters, run)

        refs.append(cast(ToolRef, await ctx.require(TOOLS).register(
            ctx, name=name, description=description,
            parameters=normalize_tool_parameters(parameters), open=open_tool, parallel=True,
        )))
    return tuple(refs)

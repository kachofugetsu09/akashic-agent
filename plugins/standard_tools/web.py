from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
import json
from typing import Any, cast

import httpx

from agent.plugin_composition import Context
from agent.tools.base import normalize_tool_parameters
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from core.net.http import HttpRequester, RequestBudget, RetryPolicy
from plugins.tools.api import CallSource, Result
from plugins.tools.plugin import TOOLS
from session.message import ContentPart
from session.message_codec import json_value

from .files import prepare_arguments


class WebTool:
    idempotent = False

    def __init__(self, backend: WebFetchTool | WebSearchTool):
        self._backend = backend

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        return prepare_arguments(self._backend, arguments)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        text = await self._backend.execute(**cast(dict[str, Any], json_value(arguments)))
        value: object = json.loads(text)
        if not isinstance(value, dict):
            raise TypeError("Web 后端结果必须是 JSON 对象")
        if "error" in value and not isinstance(value["error"], str):
            raise TypeError("Web 后端 error 必须是字符串")
        return Result("error" if "error" in value else "success", (ContentPart("text", text),))

    async def query(self, key: str) -> Result | None:
        return None


async def register_web(ctx: Context) -> None:
    """复用 HTTP 预算与重试实现，每次资源 scope 自行关闭客户端。"""
    @asynccontextmanager
    async def open_fetch(_state: Mapping[str, object]) -> AsyncGenerator[WebTool]:
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)) as client:
            requester = HttpRequester(client, RetryPolicy(), 30.0, RequestBudget(total_timeout_s=45.0))
            yield WebTool(WebFetchTool(requester))

    @asynccontextmanager
    async def open_search(_state: Mapping[str, object]) -> AsyncGenerator[WebTool]:
        yield WebTool(WebSearchTool())

    for backend, open_tool in ((WebFetchTool, open_fetch), (WebSearchTool, open_search)):
        _ = await ctx.require(TOOLS).register(
            ctx, name=backend.name, description=backend.description,
            parameters=normalize_tool_parameters(backend.parameters), open=open_tool,
            risk="read-only", always_on=True,
        )

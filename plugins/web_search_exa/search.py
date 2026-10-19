"""Exa 搜索：经 Exa MCP 端点调用 web_search_exa；端点与可选 API key 由本插件配置。"""

import json
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from agent.plugin_composition import CredentialRef
from core.net.http import http_error_detail
from plugins.standard_web.contract import SearchRequest

DEFAULT_ENDPOINT = "https://mcp.exa.ai/mcp"


class ExaConfig(BaseModel):
    """endpoint 可换成自建或代理的 Exa MCP；api_key 为空时使用公开额度。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: HttpUrl = HttpUrl(DEFAULT_ENDPOINT)
    api_key: CredentialRef | None = None
    search_type: Literal["auto", "fast", "deep"] = "auto"
    livecrawl: Literal["fallback", "preferred"] = "fallback"
    timeout_seconds: float = Field(default=25.0, gt=0, le=120, allow_inf_nan=False)

    # key 只能存为凭据；端点里带账号或查询参数会把秘密写进普通配置。
    @field_validator("endpoint")
    @classmethod
    def check_endpoint(cls, value: HttpUrl) -> HttpUrl:
        if value.username or value.password or value.query or value.fragment:
            raise ValueError("endpoint 必须是不含账号、查询参数的 HTTP(S) 地址；API key 请填在 API key 字段")
        return value


class ExaSearch:
    """一次搜索一个 JSON-RPC 请求；API key 只在请求时读取，不进入结果或日志。"""

    def __init__(self, config: ExaConfig, read_key: Callable[[], Awaitable[str | None]]):
        self._config = config
        self._read_key = read_key

    async def search(self, request: SearchRequest) -> dict[str, object]:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": {
                    "query": request.query,
                    "numResults": request.num_results,
                    "livecrawl": self._config.livecrawl,
                    "type": self._config.search_type,
                },
            },
        }
        # 1. Exa MCP 以 exaApiKey 查询参数接收 key；没有 key 时走公开端点额度。
        key = await self._read_key()
        params = {"exaApiKey": key} if key else None
        try:
            async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                response = await client.post(
                    str(self._config.endpoint),
                    params=params,
                    json=payload,
                    headers={
                        "accept": "application/json, text/event-stream",
                        "content-type": "application/json",
                    },
                )
                _ = response.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            reason = http_error_detail(e, secrets=() if key is None else (key,))
            return {"error": f"搜索失败：{reason}（{self._config.endpoint.host}）", "query": request.query}

        # 2. JSON 与 SSE 都必须给出本次调用的明确结果；坏响应不是空搜索。
        try:
            return {"query": request.query, **_read_response(response)}
        except (ValueError, TypeError) as error:
            return {"error": f"搜索响应无效：{error}", "query": request.query}


def _read_response(response: httpx.Response) -> dict[str, object]:
    """验证实际 MCP reply，区分传输错误、工具失败和明确的空结果。"""
    # 1. SSE 可带进度通知；只有 id=1 的完整 reply 才能结束本次调用。
    media_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
    if media_type == "application/json":
        messages = (response.text,)
    elif media_type == "text/event-stream":
        messages = _events(response.text)
    else:
        raise ValueError(f"不支持的 Content-Type: {media_type}")
    for raw in messages:
        parsed: object = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("缺少 JSON-RPC 对象")
        message = cast(dict[str, object], parsed)
        if message.get("jsonrpc") != "2.0":
            raise ValueError("缺少 JSON-RPC 2.0 标记")
        if "id" not in message and isinstance(message.get("method"), str) and not ({"result", "error"} & message.keys()):
            continue
        if type(message.get("id")) is not int or message["id"] != 1:
            raise ValueError("响应不属于本次搜索请求")
        if ("error" in message) == ("result" in message):
            raise ValueError("响应必须只包含 result 或 error")
        if "error" in message:
            error = message["error"]
            if not isinstance(error, dict) or not isinstance(cast(dict[str, object], error).get("message"), str):
                raise ValueError("JSON-RPC error 缺少 message")
            return {"error": error["message"]}

        # 2. 保留全部文本和结构化结果；不忽略 isError 或用缺字段构造成功。
        result = message["result"]
        if not isinstance(result, dict):
            raise ValueError("工具 result 必须是对象")
        result = cast(dict[str, object], result)
        failed = result.get("isError", False)
        if type(failed) is not bool:
            raise TypeError("工具 isError 必须是 bool")
        content = result.get("content")
        if not isinstance(content, list):
            raise ValueError("工具 result 缺少 content 数组")
        texts: list[str] = []
        for block in cast(list[object], content):
            if not isinstance(block, dict):
                raise ValueError("搜索内容块必须是对象")
            item = cast(dict[str, object], block)
            if item.get("type") != "text" or not isinstance(item.get("text"), str):
                raise ValueError("搜索返回了不支持的内容块")
            texts.append(cast(str, item["text"]))
        text = "\n".join(texts)
        if failed:
            return {"error": text or "搜索服务报告工具失败"}
        reply: dict[str, object] = {"result": text}
        if "structuredContent" in result:
            structured = result["structuredContent"]
            if not isinstance(structured, dict):
                raise ValueError("structuredContent 必须是对象")
            reply["structured_content"] = structured
        return reply
    raise ValueError("响应没有本次调用的 result")


def _events(text: str) -> Iterator[str]:
    """按完整 SSE event 合并 data 行，保留多行 JSON 与注释心跳。"""
    data: list[str] = []
    event = "message"
    for line in (*text.splitlines(), ""):
        if not line:
            if data:
                if event != "message":
                    raise ValueError(f"搜索返回了不支持的 SSE event: {event}")
                yield "\n".join(data)
                data.clear()
            event = "message"
        elif line.startswith("event:"):
            event = line[6:].lstrip(" ") or "message"
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)

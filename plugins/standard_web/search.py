"""
WebSearch 工具 — 基于 Exa MCP 公开端点，无需 API Key
"""

import json
from collections.abc import Iterator
from typing import Any, cast

import httpx

from agent.tools.base import Tool

_MCP_URL = "https://mcp.exa.ai/mcp"
_DEFAULT_NUM_RESULTS = 8


class WebSearchTool(Tool):
    """用关键词通过 Exa 搜索互联网，返回标题、内容摘要、URL 列表"""

    name = "web_search"
    description = (
        "用关键词搜索互联网，返回最新的搜索结果（标题 + 摘要 + URL）。"
        "适合查询时效性信息：新闻、产品发布、价格、人物动态等。"
        "拿到 URL 后可用 web_fetch 获取完整内容。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "num_results": {
                "type": "integer",
                "description": f"返回结果数量，默认 {_DEFAULT_NUM_RESULTS}，最大 20",
                "minimum": 1,
                "maximum": 20,
            },
            "livecrawl": {
                "type": "string",
                "enum": ["fallback", "preferred"],
                "description": "实时抓取模式：fallback（缓存优先）或 preferred（优先实时），默认 fallback",
            },
            "type": {
                "type": "string",
                "enum": ["auto", "fast", "deep"],
                "description": "搜索类型：auto（均衡）、fast（快速）、deep（深度），默认 auto",
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        num_results: int = min(int(kwargs.get("num_results", _DEFAULT_NUM_RESULTS)), 20)
        livecrawl: str = kwargs.get("livecrawl", "fallback")
        search_type: str = kwargs.get("type", "auto")

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": {
                    "query": query,
                    "numResults": num_results,
                    "livecrawl": livecrawl,
                    "type": search_type,
                },
            },
        }

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                response = await client.post(
                    _MCP_URL,
                    json=payload,
                    headers={
                        "accept": "application/json, text/event-stream",
                        "content-type": "application/json",
                    },
                )
                _ = response.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            return json.dumps(
                {"error": f"搜索失败：{e}", "query": query}, ensure_ascii=False
            )

        # 2. JSON 与 SSE 都必须给出本次调用的明确结果；坏响应不是空搜索。
        try:
            return json.dumps({"query": query, **_read_response(response)}, ensure_ascii=False)
        except (ValueError, TypeError) as error:
            return json.dumps({"error": f"搜索响应无效：{error}", "query": query}, ensure_ascii=False)


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

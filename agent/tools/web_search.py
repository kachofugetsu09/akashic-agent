"""
WebSearch 工具 — 基于 Exa MCP 公开端点，无需 API Key
"""

import json
from typing import Any

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
        import httpx

        query: str = kwargs["query"]
        num_results: int = min(int(kwargs.get("num_results", _DEFAULT_NUM_RESULTS)), 20)
        livecrawl: str = kwargs.get("livecrawl", "fallback")
        search_type: str = kwargs.get("type", "auto")

        payload = {
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
                response.raise_for_status()
        except Exception as e:
            return json.dumps(
                {"error": f"搜索失败：{e}", "query": query}, ensure_ascii=False
            )

        # 解析 SSE 响应
        text = response.text
        for line in text.splitlines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    content = data.get("result", {}).get("content", [])
                    if content:
                        return json.dumps(
                            {"query": query, "result": content[0].get("text", "")},
                            ensure_ascii=False,
                        )
                except json.JSONDecodeError:
                    continue

        return json.dumps(
            {"query": query, "results": [], "count": 0}, ensure_ascii=False
        )

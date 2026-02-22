"""
WebSearch 工具 — 基于 DuckDuckGo，无需 API key
"""
import json
from typing import Any

from agent.tools.base import Tool

_DEFAULT_MAX_RESULTS = 8


class WebSearchTool(Tool):
    """用关键词在 DuckDuckGo 搜索，返回标题、摘要、URL 列表"""

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
                "description": "搜索关键词，建议用英文或中英混合以获得更好结果",
            },
            "max_results": {
                "type": "integer",
                "description": f"返回结果数量，默认 {_DEFAULT_MAX_RESULTS}，最大 20",
                "minimum": 1,
                "maximum": 20,
            },
            "region": {
                "type": "string",
                "description": "搜索区域，如 cn-zh（中文）、us-en（英文），默认 wt-wt（全球）",
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> str:
        from ddgs import DDGS

        query: str = kwargs["query"]
        max_results: int = min(int(kwargs.get("max_results", _DEFAULT_MAX_RESULTS)), 20)
        region: str = kwargs.get("region", "wt-wt")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region=region, max_results=max_results))
        except Exception as e:
            return json.dumps({"error": f"搜索失败：{e}", "query": query}, ensure_ascii=False)

        if not results:
            return json.dumps({"query": query, "results": [], "count": 0}, ensure_ascii=False)

        formatted = [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ]
        return json.dumps({"query": query, "count": len(formatted), "results": formatted}, ensure_ascii=False)

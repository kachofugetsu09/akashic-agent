import json
from typing import TYPE_CHECKING, Any

from agent.tools.base import Tool

if TYPE_CHECKING:
    from agent.tools.registry import ToolRegistry


class ToolSearchTool(Tool):
    """在工具目录中搜索可用工具，帮助模型发现并解锁需要的工具。

    调用此工具后，匹配到的工具将在本轮对话中解锁，可直接调用。
    """

    def __init__(self, registry: "ToolRegistry") -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "tool_search"

    @property
    def description(self) -> str:
        return (
            "在工具目录中搜索可用工具。当你需要某类功能但不确定具体工具名时，"
            "先调用此工具查找，返回的工具将立即解锁可用。"
            "示例查询：'定时任务'、'文件读写'、'RSS订阅'、'健康数据'。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，描述你需要的功能，例如：'定时任务'、'文件读取'、'订阅管理'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的最大工具数量，默认 5，最大 10",
                    "default": 5,
                },
                "allowed_risk": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["read-only", "write", "external-side-effect"],
                    },
                    "description": "允许的风险等级，不填则不过滤。read-only=只读，write=写操作，external-side-effect=外部副作用",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        allowed_risk: list[str] | None = None,
        **_: Any,
    ) -> str:
        top_k = min(max(1, int(top_k)), 10)
        results = self._registry.search(query=query, top_k=top_k, allowed_risk=allowed_risk)
        if not results:
            return json.dumps(
                {"matched": [], "tip": "没有找到匹配工具，请换个关键词重试"},
                ensure_ascii=False,
            )
        return json.dumps({"matched": results}, ensure_ascii=False, indent=2)

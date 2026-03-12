import json
from typing import TYPE_CHECKING, Any

from agent.tools.base import Tool

if TYPE_CHECKING:
    from agent.tools.registry import ToolRegistry


class ListToolsTool(Tool):
    """列出当前所有已注册的可用工具及其简要说明。"""

    def __init__(self, registry: "ToolRegistry") -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "list_tools"

    @property
    def description(self) -> str:
        return (
            "列出所有已注册的可用工具，包括名称、功能摘要和标签。"
            "当你需要了解自己有哪些能力时调用此工具。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "按标签过滤，例如 'filesystem'、'scheduling'、'mcp'。不填则返回全部。",
                },
            },
            "required": [],
        }

    async def execute(self, tag: str = "", **_: Any) -> str:
        tag = tag.strip().lower()
        tools = []
        for name, tool in self._registry._tools.items():
            if name in ("tool_search", "list_tools"):
                continue
            meta = self._registry._metadata.get(name)
            if tag and meta and tag not in meta.tags:
                continue
            tools.append(
                {
                    "name": name,
                    "summary": tool.description[:80],
                    "tags": meta.tags if meta else [],
                    "risk": meta.risk if meta else "unknown",
                }
            )

        return json.dumps(
            {"total": len(tools), "tools": tools},
            ensure_ascii=False,
            indent=2,
        )

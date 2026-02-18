import logging

from agent.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """管理所有可用工具"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"注册工具: {tool.name}")

    def get_schemas(self) -> list[dict]:
        """返回 OpenAI function calling 格式的工具定义列表"""
        return [t.to_schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"工具 '{name}' 不存在"
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            logger.error(f"工具 {name} 执行出错: {e}", exc_info=True)
            return f"工具执行出错: {e}"

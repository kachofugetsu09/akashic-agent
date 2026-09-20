import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any

from agent.tool_catalog import (
    ToolResult,
    normalize_tool_parameters,
    normalize_tool_result,
    validate_tool_parameters,
)
from agent.tool_context import (
    ToolExecutionContext,
    get_current_tool_context,
    tool_execution_context_scope,
)


class Tool(ABC):
    """工具抽象基类"""

    name: str
    description: str
    parameters: dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is Tool or inspect.isabstract(cls):
            return

        missing_fields = [
            field
            for field in ("name", "description", "parameters")
            if getattr(cls, field, None) is None
        ]
        if missing_fields:
            fields_text = ", ".join(missing_fields)
            raise TypeError(f"{cls.__name__} 必须定义字段：{fields_text}")

        empty_fields: list[str] = []
        name = getattr(cls, "name")
        if not isinstance(name, property) and not str(name).strip():
            empty_fields.append("name")
        description = getattr(cls, "description")
        if not isinstance(description, property) and not str(description).strip():
            empty_fields.append("description")
        parameters = getattr(cls, "parameters")
        if not isinstance(parameters, property) and not parameters:
            empty_fields.append("parameters")
        if empty_fields:
            fields_text = ", ".join(empty_fields)
            raise TypeError(f"{cls.__name__} 字段不能为空：{fields_text}")

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str | ToolResult:
        """执行工具，返回字符串结果"""

    async def execute_with_timeout(
        self,
        arguments: dict[str, Any],
        execution_timeout: float | None = None,
    ) -> str | ToolResult:
        execution = self.execute(**arguments)
        if execution_timeout is None:
            return await execution
        return await asyncio.wait_for(execution, timeout=execution_timeout)

    def validate_params(
        self,
        params: dict[str, Any],
        *,
        schema: dict[str, Any] | None = None,
    ) -> list[str]:
        """校验参数，返回错误列表（空列表表示校验通过）"""
        return validate_tool_parameters(
            params,
            schema=schema if schema is not None else self.parameters or {},
        )

    def to_schema(self) -> dict[str, Any]:
        """转换为 OpenAI function calling 格式"""
        fn: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": normalize_tool_parameters(self.parameters),
        }
        return {"type": "function", "function": fn}

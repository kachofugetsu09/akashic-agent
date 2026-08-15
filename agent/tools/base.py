import asyncio
from copy import deepcopy
import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast


def normalize_tool_parameters(
    parameters: dict[str, Any],
    *,
    open_object: bool = False,
) -> dict[str, Any]:
    """Normalize object schemas while preserving explicit JSON Schema policy."""

    schema = cast(dict[str, Any], deepcopy(parameters))

    def visit(node: dict[str, Any], *, root: bool = False) -> None:
        if node.get("type") == "object" or isinstance(node.get("properties"), dict):
            if root and "additionalProperties" not in node:
                node["additionalProperties"] = open_object
            properties = node.get("properties")
            if isinstance(properties, dict):
                for child in properties.values():
                    if isinstance(child, dict):
                        visit(cast(dict[str, Any], child))
            additional = node.get("additionalProperties")
            if isinstance(additional, dict):
                visit(cast(dict[str, Any], additional))
        elif isinstance(node.get("items"), dict):
            visit(cast(dict[str, Any], node["items"]))

    visit(schema, root=True)
    return schema


def validate_tool_parameters(
    params: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    """按 Core 持有的 JSON Schema 合同校验工具参数。"""

    # 1. 顶层工具参数必须是对象合同。
    if schema.get("type", "object") != "object":
        raise ValueError(f"Schema 顶层类型必须为 object，当前为 {schema.get('type')!r}")

    # 2. 递归校验值与子 schema。
    return _validate_tool_value(params, {**schema, "type": "object"}, "")


def _validate_tool_value(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> list[str]:
    """递归校验一个 JSON Schema 值。"""

    # 1. 类型错误终止当前节点的后续约束计算。
    type_errors = _validate_tool_type(value, schema, path)
    if type_errors:
        return type_errors

    # 2. 标量约束与容器子节点独立累积错误。
    errors = _validate_tool_constraints(value, schema, path)
    schema_type = schema.get("type")
    if schema_type == "object":
        errors.extend(_validate_tool_object(value, schema, path))
    if schema_type == "array" and "items" in schema:
        errors.extend(_validate_tool_array(value, schema, path))
    return errors


def _validate_tool_type(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> list[str]:
    schema_type = schema.get("type")
    if not isinstance(schema_type, str):
        return []
    type_map: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(schema_type)
    if expected is None:
        return []
    valid_type = isinstance(value, expected)
    if schema_type in ("integer", "number") and isinstance(value, bool):
        valid_type = False
    return [] if valid_type else [f"{path or '参数'} 应为 {schema_type} 类型"]


def _validate_tool_constraints(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> list[str]:
    label = path or "参数"
    schema_type = schema.get("type")
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{label} 须为以下值之一：{schema['enum']}")

    if schema_type in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{label} 须 >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{label} 须 <= {schema['maximum']}")

    if schema_type == "string":
        string_value = cast(str, value)
        if "minLength" in schema and len(string_value) < schema["minLength"]:
            errors.append(f"{label} 最短 {schema['minLength']} 个字符")
        if "maxLength" in schema and len(string_value) > schema["maxLength"]:
            errors.append(f"{label} 最长 {schema['maxLength']} 个字符")
    return errors


def _validate_tool_object(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> list[str]:
    object_value = cast(dict[str, Any], value)
    properties = schema.get("properties", {})
    errors = [
        f"缺少必填字段：{path + '.' + name if path else name}"
        for name in schema.get("required", [])
        if name not in object_value
    ]
    additional = schema.get("additionalProperties")
    for name, child in object_value.items():
        child_path = f"{path}.{name}" if path else name
        if name in properties:
            errors.extend(_validate_tool_value(child, properties[name], child_path))
        elif additional is False:
            errors.append(f"不允许额外字段：{child_path}")
        elif isinstance(additional, dict):
            errors.extend(_validate_tool_value(child, additional, child_path))
    return errors


def _validate_tool_array(
    value: Any,
    schema: dict[str, Any],
    path: str,
) -> list[str]:
    errors: list[str] = []
    for index, item in enumerate(cast(list[Any], value)):
        errors.extend(
            _validate_tool_value(
                item,
                schema["items"],
                f"{path}[{index}]" if path else f"[{index}]",
            )
        )
    return errors


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Immutable runtime provenance captured for one tool execution."""

    origin_channel: str = ""
    origin_chat_id: str = ""
    origin_session_key: str = ""
    turn_id: str = ""
    current_timestamp: str = ""
    current_user_source_ref: str = ""
    execution_id: str = ""

    @property
    def timestamp(self) -> datetime | None:
        if not self.current_timestamp:
            return None
        return datetime.fromisoformat(self.current_timestamp)


_CURRENT_TOOL_CONTEXT: ContextVar[ToolExecutionContext | None] = ContextVar(
    "akashic_current_tool_execution_context",
    default=None,
)


def get_current_tool_context() -> ToolExecutionContext | None:
    """Return the immutable provenance for the current async execution."""

    return _CURRENT_TOOL_CONTEXT.get()


def set_current_tool_context(
    context: ToolExecutionContext | None,
) -> Token[ToolExecutionContext | None]:
    """Bind a runtime context in the current async task."""

    return _CURRENT_TOOL_CONTEXT.set(context)


@contextmanager
def tool_execution_context_scope(
    context: ToolExecutionContext | None,
) -> Any:
    """Bind one execution context and restore the caller context on exit."""

    token: Token[ToolExecutionContext | None] = _CURRENT_TOOL_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_TOOL_CONTEXT.reset(token)


@dataclass
class ToolResult:
    text: str = ""
    content_blocks: list[dict[str, Any]] = field(default_factory=list)
    mobile_attention: Literal["confirmation"] | None = None
    runtime_provenance: dict[str, str] = field(default_factory=dict)

    def preview(self) -> str:
        if self.text:
            return self.text
        if self.content_blocks:
            return f"[多模态结果 {len(self.content_blocks)} blocks]"
        return ""


def normalize_tool_result(result: str | ToolResult) -> ToolResult:
    if isinstance(result, ToolResult):
        return result
    return ToolResult(text=result)


class Tool(ABC):
    """工具抽象基类"""

    name: str
    description: str
    parameters: dict[str, Any]

    # JSON Schema 类型 → Python 类型映射
    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

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
        active_schema = schema if schema is not None else self.parameters or {}
        if active_schema.get("type", "object") != "object":
            raise ValueError(
                f"Schema 顶层类型必须为 object，当前为 {active_schema.get('type')!r}"
            )
        return self._validate(params, {**active_schema, "type": "object"}, "")

    def _validate(self, val: Any, schema: dict[str, Any], path: str) -> list[str]:
        """递归校验值是否符合 schema，返回错误列表"""
        return _validate_tool_value(val, schema, path)

    def to_schema(self) -> dict[str, Any]:
        """转换为 OpenAI function calling 格式"""
        fn: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": normalize_tool_parameters(self.parameters),
        }
        return {"type": "function", "function": fn}

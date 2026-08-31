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
        label = path or "参数"
        t = schema.get("type")

        if t in self._TYPE_MAP:
            valid_type = isinstance(val, self._TYPE_MAP[t])
            if t in ("integer", "number") and isinstance(val, bool):
                valid_type = False
            if not valid_type:
                return [f"{label} 应为 {t} 类型"]

        errors = []

        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} 须为以下值之一：{schema['enum']}")

        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} 须 >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} 须 <= {schema['maximum']}")

        if t == "string":
            string_value = cast(str, val)
            if "minLength" in schema and len(string_value) < schema["minLength"]:
                errors.append(f"{label} 最短 {schema['minLength']} 个字符")
            if "maxLength" in schema and len(string_value) > schema["maxLength"]:
                errors.append(f"{label} 最长 {schema['maxLength']} 个字符")

        if t == "object":
            object_value = cast(dict[str, Any], val)
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in object_value:
                    errors.append(f"缺少必填字段：{path + '.' + k if path else k}")
            for k, v in object_value.items():
                if k in props:
                    errors.extend(
                        self._validate(v, props[k], f"{path}.{k}" if path else k)
                    )
                elif schema.get("additionalProperties") is False:
                    errors.append(
                        f"不允许额外字段：{path + '.' + k if path else k}"
                    )
                elif isinstance(schema.get("additionalProperties"), dict):
                    errors.extend(
                        self._validate(
                            v,
                            cast(dict[str, Any], schema["additionalProperties"]),
                            f"{path}.{k}" if path else k,
                        )
                    )

        if t == "array" and "items" in schema:
            array_value = cast(list[Any], val)
            for i, item in enumerate(array_value):
                errors.extend(
                    self._validate(
                        item, schema["items"], f"{path}[{i}]" if path else f"[{i}]"
                    )
                )

        return errors

    def to_schema(self) -> dict[str, Any]:
        """转换为 OpenAI function calling 格式"""
        fn: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": normalize_tool_parameters(self.parameters),
        }
        return {"type": "function", "function": fn}

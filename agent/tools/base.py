import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    text: str = ""
    content_blocks: list[dict[str, Any]] = field(default_factory=list)

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

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """校验参数，返回错误列表（空列表表示校验通过）"""
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(
                f"Schema 顶层类型必须为 object，当前为 {schema.get('type')!r}"
            )
        return self._validate(params, {**schema, "type": "object"}, "")

    def _validate(self, val: Any, schema: dict[str, Any], path: str) -> list[str]:
        """递归校验值是否符合 schema，返回错误列表"""
        label = path or "参数"
        t = schema.get("type")

        if t in self._TYPE_MAP and not isinstance(val, self._TYPE_MAP[t]):
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
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} 最短 {schema['minLength']} 个字符")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} 最长 {schema['maxLength']} 个字符")

        if t == "object":
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in val:
                    errors.append(f"缺少必填字段：{path + '.' + k if path else k}")
            for k, v in val.items():
                if k in props:
                    errors.extend(
                        self._validate(v, props[k], f"{path}.{k}" if path else k)
                    )

        if t == "array" and "items" in schema:
            for i, item in enumerate(val):
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
            "parameters": self.parameters,
        }
        return {"type": "function", "function": fn}

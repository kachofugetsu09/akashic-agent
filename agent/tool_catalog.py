"""Source-neutral schema and result atoms shared by tool providers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast


@dataclass
class ToolResult:
    """Represent the source-neutral result of one tool operation."""

    text: str = ""
    content_blocks: list[dict[str, Any]] = field(default_factory=list)
    mobile_attention: Literal["confirmation"] | None = None
    runtime_provenance: dict[str, str] = field(default_factory=dict)
    is_error: bool = False

    def __post_init__(self) -> None:
        if type(self.is_error) is not bool:
            raise TypeError("工具 is_error 必须是 bool")

    def preview(self) -> str:
        if self.text:
            return self.text
        if self.content_blocks:
            return f"[多模态结果 {len(self.content_blocks)} blocks]"
        return ""


def normalize_tool_result(result: str | ToolResult) -> ToolResult:
    """Convert the two supported tool result shapes to one source-neutral value."""

    if isinstance(result, ToolResult):
        return result
    return ToolResult(text=result)


def normalize_tool_parameters(
    parameters: Mapping[str, Any],
    *,
    open_object: bool = False,
) -> dict[str, Any]:
    """Normalize nested object schemas while preserving explicit JSON Schema policy."""

    schema = cast(dict[str, Any], deepcopy(dict(parameters)))

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


_TOOL_TYPE_MAP: dict[str, type[object] | tuple[type[object], ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_tool_parameters(
    params: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate one tool call and return user-facing parameter errors."""

    active_schema = dict(schema) if schema is not None else {}
    if active_schema.get("type", "object") != "object":
        raise ValueError(
            f"Schema 顶层类型必须为 object，当前为 {active_schema.get('type')!r}"
        )
    return _validate_tool_value(
        dict(params), {**active_schema, "type": "object"}, ""
    )


def _validate_tool_value(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
) -> list[str]:
    label = path or "参数"
    schema_type = schema.get("type")

    if schema_type in _TOOL_TYPE_MAP:
        valid_type = isinstance(value, _TOOL_TYPE_MAP[schema_type])
        if schema_type in ("integer", "number") and isinstance(value, bool):
            valid_type = False
        if not valid_type:
            return [f"{label} 应为 {schema_type} 类型"]

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

    if schema_type == "object":
        object_value = cast(dict[str, Any], value)
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in object_value:
                errors.append(f"缺少必填字段：{path + '.' + key if path else key}")
        for key, child_value in object_value.items():
            if key in properties:
                errors.extend(
                    _validate_tool_value(
                        child_value,
                        cast(Mapping[str, Any], properties[key]),
                        f"{path}.{key}" if path else key,
                    )
                )
            elif schema.get("additionalProperties") is False:
                errors.append(f"不允许额外字段：{path + '.' + key if path else key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(
                    _validate_tool_value(
                        child_value,
                        cast(Mapping[str, Any], schema["additionalProperties"]),
                        f"{path}.{key}" if path else key,
                    )
                )

    if schema_type == "array" and "items" in schema:
        array_value = cast(list[Any], value)
        for index, item in enumerate(array_value):
            errors.extend(
                _validate_tool_value(
                    item,
                    cast(Mapping[str, Any], schema["items"]),
                    f"{path}[{index}]" if path else f"[{index}]",
                )
            )

    return errors


__all__ = [
    "ToolResult",
    "normalize_tool_parameters",
    "normalize_tool_result",
    "validate_tool_parameters",
]

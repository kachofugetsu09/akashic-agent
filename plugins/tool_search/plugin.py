from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.models.contract import ToolCall as ModelToolCall
from agent.plugin_contracts import ContentPart, json_value
from core.common.frozen_json import freeze_json
from plugins.tools.contract import TOOL_LOADING_PRESENTATION as TOOL_LOADING_PRESENTATION

from ._tool_boundary import (
    TOOLS,
    BoundTool,
    CallSource,
    ToolCatalog,
    ToolPresentation,
    ToolRef,
    ToolResultValue,
    ToolView,
)

api_version = 3
name = "tool_search"
version = "3.0.0"
desc = "按插件 ID 展示获授工具的完整 schema，并解码间接调用"
inject = (TOOLS,)


def _tool_schema(description: Mapping[str, object]) -> Mapping[str, Any]:
    """Convert an owner description into the model-facing function schema."""
    return {
        "type": "function",
        "function": {
            "name": description["name"],
            "description": description["description"],
            "parameters": description["parameters"],
        },
    }


class LoadToolsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    plugin: str = Field(min_length=1)


class IndirectCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    arguments: dict[str, object]


def _groups(catalog: ToolCatalog, view: ToolView) -> tuple[dict[str, object], ...]:
    """Build the frozen non-direct plugin directory from one granted view."""
    rows: dict[str, list[Mapping[str, Any]]] = {}
    purposes: dict[str, str] = {}
    for ref in sorted(view.refs, key=lambda item: item.name):
        if catalog.group_always_on(ref):
            continue
        plugin = cast(str, ref.description["owner"])
        rows.setdefault(plugin, []).append(_tool_schema(ref.description))
        purposes.setdefault(plugin, catalog.group_description(ref))
    return tuple(
        {"plugin": plugin, "purpose": purposes[plugin], "tools": tuple(rows[plugin])}
        for plugin in sorted(rows)
    )


def _validate_groups(value: object) -> tuple[dict[str, object], ...]:
    """Validate the frozen binding directory before it reaches the loader."""
    if not isinstance(value, tuple):
        raise ValueError("工具加载固定目录损坏")
    groups: list[dict[str, object]] = []
    plugins: set[str] = set()
    for group in value:
        if not isinstance(group, Mapping) or set(group) != {"plugin", "purpose", "tools"}:
            raise ValueError("工具加载固定目录分组损坏")
        plugin = group["plugin"]
        purpose = group["purpose"]
        tools = group["tools"]
        if (
            not isinstance(plugin, str) or not plugin
            or not isinstance(purpose, str) or not purpose
            or not isinstance(tools, tuple) or not tools
            or plugin in plugins
        ):
            raise ValueError("工具加载固定目录分组无效")
        schemas: list[Mapping[str, Any]] = []
        for schema in tools:
            if not isinstance(schema, Mapping) or schema.get("type") != "function":
                raise ValueError("工具加载固定 schema 损坏")
            function = schema.get("function")
            if (
                not isinstance(function, Mapping)
                or not isinstance(function.get("name"), str)
                or not isinstance(function.get("description"), str)
                or not isinstance(function.get("parameters"), Mapping)
            ):
                raise ValueError("工具加载固定 schema 损坏")
            schemas.append(cast(Mapping[str, Any], schema))
        plugins.add(plugin)
        groups.append({"plugin": plugin, "purpose": purpose, "tools": tuple(schemas)})
    return tuple(groups)


class LoadTools:
    idempotent = True

    def __init__(self, groups: tuple[dict[str, object], ...]):
        self._groups = {cast(str, group["plugin"]): group for group in groups}

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object] | str:
        try:
            return LoadToolsInput.model_validate(json_value(arguments)).model_dump(mode="json")
        except ValidationError as error:
            return str(error)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        plugin = LoadToolsInput.model_validate(json_value(arguments)).plugin
        group = self._groups.get(plugin)
        if group is None:
            return ToolResultValue(
                "error",
                (ContentPart("text", json.dumps({
                    "plugin": plugin,
                    "error": "插件不属于当前获授工具目录。请使用 system 中的准确插件 ID。",
                }, ensure_ascii=False)),),
            )
        return ToolResultValue(
            "success",
            (ContentPart("text", json.dumps({
                "plugin": plugin,
                "tools": json_value(group["tools"]),
                "tip": "使用 tool_call，并传入 name 与 arguments。",
            }, ensure_ascii=False)),),
        )

    async def query(self, key: str) -> ToolResultValue | None:
        return None


class ToolLoadingPresentation:
    """Expose fixed direct schemas and load complete granted plugin groups."""

    def __init__(self, catalog: ToolCatalog, view: ToolView, load_ref: ToolRef):
        self._view = view
        self._load_ref = load_ref
        if any(ref.name == "tool_call" for ref in view.refs):
            raise ValueError("获授 view 的原生工具名与间接调用协议冲突: tool_call")
        self._direct = catalog.view(*(
            ref for ref in view.refs if catalog.group_always_on(ref)
        ))
        self._groups = _groups(catalog, view)
        self._schemas: tuple[Mapping[str, Any], ...] | None = None

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]:
        if self._schemas is None:
            self._schemas = cast(tuple[Mapping[str, Any], ...], freeze_json((
                *(_tool_schema(ref.description) for ref in self._direct.refs),
                {
                    "type": "function",
                    "function": {
                        "name": "tool_call",
                        "description": "调用获授 view 中已经取得 schema 的工具。",
                        "parameters": IndirectCall.model_json_schema(),
                    },
                },
            )))
        return self._schemas

    @property
    def system_prompt(self) -> str:
        lines = [
            "## 插件工具目录",
            "固定工具已经提供 schema。用 load_tools 传入目录中的准确 plugin ID，取得该插件完整 schema 后，再用 tool_call 传入 name 与 arguments。",
        ]
        lines.extend(
            f"{group['plugin']} · {group['purpose']} · {len(cast(tuple[object, ...], group['tools']))} 个工具"
            for group in self._groups
        )
        return "\n".join(lines)

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]] | str:
        if call.name != "tool_call":
            if call.name not in {ref.name for ref in self._direct.refs}:
                return f"工具不属于当前直接调用目录: {call.name}；请用 load_tools 查看插件目录。"
            return call.name, cast(Mapping[str, object], call.arguments)
        try:
            decoded = IndirectCall.model_validate(json_value(call.arguments))
        except ValidationError as error:
            return f"tool_call 需要 name 和对象类型的 arguments：{error}"
        if decoded.name not in {ref.name for ref in self._view.refs}:
            return f"工具不属于获授 view: {decoded.name}；请用 load_tools 查看插件目录。"
        return decoded.name, decoded.arguments

    def configuration(self, name: str) -> Mapping[str, object] | None:
        return {"groups": self._groups} if name == self._load_ref.name else None


async def apply(ctx: Context) -> None:
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)

    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if set(configuration) != {"groups"}:
            raise ValueError("工具加载 binding 缺少固定目录")
        return {"groups": _validate_groups(configuration["groups"])}

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncIterator[BoundTool]:
        groups = state.get("groups")
        yield LoadTools(_validate_groups(groups))

    load_ref = await catalog.register(
        ctx,
        name="load_tools",
        description="按准确插件 ID 加载当前获授工具组的完整 schema。",
        parameters=LoadToolsInput.model_json_schema(),
        open=open_tool,
        capture=capture,
        public=False,
        idempotent=True,
        parallel=True,
    )

    def present(awarded: ToolView) -> tuple[ToolView, ToolPresentation]:
        view = catalog.view(*awarded.refs, load_ref)
        return view, ToolLoadingPresentation(catalog, view, load_ref)

    _ = await ctx.provide(TOOL_LOADING_PRESENTATION, present)

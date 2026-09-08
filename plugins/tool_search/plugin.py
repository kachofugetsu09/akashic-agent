from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
import json
import re
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.models import ToolCall as ModelToolCall
from plugins.context.api import Reminder
from plugins.tools.api import BoundTool, CallSource, InvalidArguments, Result
from plugins.tools.menu import ToolPresentation, tool_schema
from plugins.tools.plugin import TOOLS, ToolCatalog, ToolRef, ToolView
from session.message import ContentPart
from session.message_codec import json_value

api_version = 3
name = "tool_search"
version = "2.0.0"
desc = "在获授工具 view 内搜索完整 schema，并解码间接调用"
inject = (TOOLS,)

TOOL_SEARCH_TOOLS = ServiceKey[ToolView]("tool-search.tools.v1")
TOOL_SEARCH_PRESENTATION = ServiceKey[
    Callable[[ToolView], ToolPresentation]
]("tool-search.presentation.v1")


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    allowed_risk: list[
        Literal["read-only", "read-write", "external-side-effect"]
    ] | None = None


class IndirectCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    arguments: dict[str, object]


def _groups(view: ToolView) -> tuple[dict[str, object], ...]:
    rows: dict[str, list[Mapping[str, Any]]] = {}
    for ref in view.refs:
        rows.setdefault(cast(str, ref.description["owner"]), []).append({
            "schema": tool_schema(ref.description),
            "risk": ref.description["risk"],
            "search_hint": ref.description["search_hint"],
        })
    return tuple(
        {"owner": owner, "tools": tools}
        for owner, tools in sorted(rows.items())
    )


def _search(groups: tuple[dict[str, object], ...], query: Query) -> tuple[dict[str, object], ...]:
    """按 owner、工具名、描述和提示匹配，并返回完整插件组。"""
    text = query.query.strip().lower()
    tokens: set[str] = {
        text,
        *text.split(),
        *(part.strip() for part in re.split(r"([\u4e00-\u9fff]+)", text)),
    }
    cjk = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    tokens.update(cjk)
    tokens.update(left + right for left, right in zip(cjk, cjk[1:]))
    tokens.discard("")
    ranked: list[tuple[int, str, dict[str, object]]] = []
    for group in groups:
        owner = cast(str, group["owner"])
        score = 0
        allowed = False
        for entry in cast(list[Mapping[str, Any]], group["tools"]):
            schema = cast(Mapping[str, Any], entry["schema"])
            tool = cast(Mapping[str, Any], schema["function"])
            risk = entry["risk"]
            if query.allowed_risk is not None and risk not in query.allowed_risk:
                continue
            allowed = True
            name = cast(str, tool["name"]).lower()
            description = cast(str, tool["description"]).lower()
            hint = cast(str | None, entry["search_hint"]) or ""
            for token in tokens:
                if token == owner.lower() or token == name:
                    score += 10
                elif token in owner.lower() or token in name:
                    score += 5
                if token in description:
                    score += 2
                if token in hint.lower():
                    score += 4
        if allowed and score:
            ranked.append((-score, owner, group))
    return tuple(row for _, _, row in sorted(ranked)[: query.top_k])


class SearchTool:
    idempotent = True

    def __init__(self, groups: tuple[dict[str, object], ...]):
        self._groups = groups

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object]:
        try:
            return Query.model_validate(json_value(arguments)).model_dump(mode="json")
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        matched = _search(self._groups, Query.model_validate(json_value(arguments)))
        visible = tuple({
            "owner": group["owner"],
            "tools": tuple(
                cast(Mapping[str, Any], entry)["schema"]
                for entry in cast(tuple[Mapping[str, Any], ...], group["tools"])
            ),
        } for group in matched)
        return Result(
            "success",
            (
                ContentPart(
                    "text",
                    json.dumps(
                        {
                            "matched_groups": json_value(visible),
                            "tip": (
                                "使用 tool_call，并传入 name 与 arguments。"
                                if matched
                                else "没有匹配工具，请调整关键词。"
                            ),
                        },
                        ensure_ascii=False,
                    ),
                ),
            ),
        )

    async def query(self, key: str) -> Result | None:
        return None


class SearchPresentation:
    """固定搜索顶层 schema，并只解码获授 view 中的间接调用。"""

    def __init__(self, ctx: Context, catalog: ToolCatalog, view: ToolView, search_ref: ToolRef):
        self._ctx = ctx
        self._view = view
        self._search_ref = search_ref
        if any(ref.name == "tool_call" for ref in view.refs):
            raise ValueError("获授 view 的原生工具名与间接调用协议冲突: tool_call")
        direct = tuple(
            ref
            for ref in view.refs
            if catalog.group_always_on(ref)
        )
        self._direct = ToolView(direct)
        self._groups = _groups(view)

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]:
        return (
            *(tool_schema(ref.description) for ref in self._direct.refs),
            {
                "type": "function",
                "function": {
                    "name": "tool_call",
                    "description": "调用已获授 view 中已知名称和参数 schema 的工具。",
                    "parameters": IndirectCall.model_json_schema(),
                },
            },
        )

    @property
    def reminders(self) -> tuple[tuple[Context, Reminder], ...]:
        lines = ["## 可搜索工具目录"]
        for group in self._groups:
            lines.append(cast(str, group["owner"]))
            for entry in cast(list[Mapping[str, Any]], group["tools"]):
                schema = cast(Mapping[str, Any], entry["schema"])
                tool = cast(Mapping[str, Any], schema["function"])
                description = cast(str, tool["description"])
                short = description[:20] + ("…" if len(description) > 20 else "")
                lines.append(f"- {tool['name']}: {short}")
        return ((self._ctx, Reminder("directory", "\n".join(lines), 500)),)

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]]:
        if call.name != "tool_call":
            return self._direct.select(call.name).name, cast(Mapping[str, object], call.arguments)
        try:
            decoded = IndirectCall.model_validate(json_value(call.arguments))
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error
        return self._view.select(decoded.name).name, decoded.arguments

    def configuration(self, name: str) -> Mapping[str, object] | None:
        return {"groups": self._groups} if name == self._search_ref.name else None


async def apply(ctx: Context, config: object) -> None:
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True)

    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if set(configuration) != {"groups"} or not isinstance(configuration["groups"], tuple):
            raise ValueError("工具搜索 binding 缺少固定组目录")
        return configuration

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncIterator[BoundTool]:
        groups = state.get("groups")
        if not isinstance(groups, tuple):
            raise ValueError("工具搜索固定组目录损坏")
        yield SearchTool(cast(tuple[dict[str, object], ...], groups))

    search_ref = await catalog.register(
        ctx,
        name="tool_search",
        description="搜索获授工具目录；命中后用 tool_call 调用返回的完整 schema。",
        parameters=Query.model_json_schema(),
        open=open_tool,
        capture=capture,
        idempotent=True,
        risk="read-only",
    )
    view = catalog.view(search_ref)
    _ = await ctx.provide(TOOL_SEARCH_TOOLS, view)
    _ = await ctx.provide(
        TOOL_SEARCH_PRESENTATION,
        lambda awarded: SearchPresentation(ctx, catalog, awarded, search_ref),
    )

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
import json
import re
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.models import ToolCall as ModelToolCall
from agent.plugin_contracts import ContentPart
from agent.plugin_contracts import json_value

from ._tool_boundary import (
    BoundTool,
    CallSource,
    ToolCatalog,
    ToolPresentation,
    ToolRef,
    ToolResultValue,
    ToolView,
    TOOLS,
)

api_version = 3
name = "tool_search"
version = "2.0.0"
desc = "在获授工具 view 内搜索完整 schema，并解码间接调用"
inject = (TOOLS,)

TOOL_SEARCH_TOOLS = ServiceKey[ToolView]("tool-search.tools.v1")
TOOL_SEARCH_PRESENTATION = ServiceKey[
    Callable[[ToolView], ToolPresentation]
]("tool-search.presentation.v1")


def _tool_schema(description: Mapping[str, object]) -> Mapping[str, Any]:
    """把 owner 提供的描述转换成模型展示 schema。"""
    return {
        "type": "function",
        "function": {
            "name": description["name"],
            "description": description["description"],
            "parameters": description["parameters"],
        },
    }


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    allowed_risk: list[
        Literal["read-only", "read-write", "external-side-effect"]
    ] | None = Field(default=None, description="按工具整体能力过滤，通常省略；此字段不是本次操作的授权。")


class IndirectCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    arguments: dict[str, object]


def _groups(catalog: ToolCatalog, view: ToolView) -> tuple[dict[str, object], ...]:
    """按实际插件分组，只包含获授工具并固定目录顺序。"""
    rows: dict[str, dict[str, object]] = {}
    for ref in sorted(view.refs, key=lambda item: item.name):
        owner = cast(str, ref.description["owner"])
        if owner not in rows:
            rows[owner] = {"owner": owner, "description": catalog.group_description(ref), "tools": []}
        cast(list[Mapping[str, Any]], rows[owner]["tools"]).append({
            "schema": _tool_schema(ref.description),
            "risk": ref.description["risk"],
            "search_hint": ref.description["search_hint"],
        })
    return tuple(rows[owner] for owner in sorted(rows))


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
    ranked: list[tuple[int, int, str, dict[str, object]]] = []
    for group in groups:
        owner = cast(str, group["owner"])
        score = 0
        exact = False
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
            exact |= owner.lower() in tokens or name in tokens
            tool_score = 0
            for token in tokens:
                if token == owner.lower() or token == name:
                    tool_score += 10
                elif token in owner.lower() or token in name:
                    tool_score += 5
                if token in description:
                    tool_score += 2
                if token in hint.lower():
                    tool_score += 4
            score = max(score, tool_score)
        if allowed and score:
            ranked.append((-int(exact), -score, owner, group))
    return tuple(row for _, _, _, row in sorted(ranked)[: query.top_k])


class SearchTool:
    idempotent = True

    def __init__(self, groups: tuple[dict[str, object], ...]):
        self._groups = groups

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object] | str:
        try:
            return Query.model_validate(json_value(arguments)).model_dump(mode="json")
        except ValidationError as error:
            return str(error)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        query = Query.model_validate(json_value(arguments))
        matched = _search(self._groups, query)
        excluded = () if query.allowed_risk is None else tuple(
            {"owner": group["owner"], "name": entry["schema"]["function"]["name"], "risk": entry["risk"]}
            for group in _search(self._groups, query.model_copy(update={"allowed_risk": None}))
            for entry in cast(tuple[Mapping[str, Any], ...], group["tools"])
            if entry["risk"] not in query.allowed_risk
        )
        visible = tuple({
            "owner": group["owner"],
            "tools": tuple(
                cast(Mapping[str, Any], entry)["schema"]
                for entry in cast(tuple[Mapping[str, Any], ...], group["tools"])
            ),
        } for group in matched)
        return ToolResultValue(
            "success",
            (
                ContentPart(
                    "text",
                    json.dumps(
                        {
                            "matched_groups": json_value(visible),
                            "excluded_by_risk": json_value(excluded),
                            "risk_tip": "部分匹配工具被风险过滤排除；需要时省略过滤重搜。返回组内保留完整 schema，执行仍须通过当前授权。" if excluded else "",
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

    async def query(self, key: str) -> ToolResultValue | None:
        return None


class SearchPresentation:
    """固定搜索顶层 schema，并只解码获授 view 中的间接调用。"""

    def __init__(self, catalog: ToolCatalog, view: ToolView, search_ref: ToolRef):
        self._view = view
        self._search_ref = search_ref
        if any(ref.name == "tool_call" for ref in view.refs):
            raise ValueError("获授 view 的原生工具名与间接调用协议冲突: tool_call")
        direct = tuple(
            ref
            for ref in view.refs
            if catalog.group_always_on(ref)
        )
        self._direct = catalog.view(*direct)
        self._groups = _groups(catalog, view)

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]:
        return (
            *(_tool_schema(ref.description) for ref in self._direct.refs),
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
    def system_prompt(self) -> str:
        lines = ["## 可搜索工具目录", "用 tool_search 获取插件组的完整 schema，再用 tool_call 传入 name 与 arguments。"]
        for group in self._groups:
            lines.append(f"{group['owner']}：{group['description']}")
            for entry in cast(list[Mapping[str, Any]], group["tools"]):
                tool = cast(Mapping[str, Any], entry["schema"])["function"]
                description = " ".join(tool["description"].split())
                short = description[:80] + ("…" if len(description) > 80 else "")
                lines.append(f"   {tool['name']}：{short}")
        return "\n".join(lines)

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]] | str:
        if call.name != "tool_call":
            if call.name not in {ref.name for ref in self._direct.refs}:
                return f"工具不属于当前直接调用目录: {call.name}；请用 tool_search 查询，再用 tool_call 调用。"
            return call.name, cast(Mapping[str, object], call.arguments)
        try:
            decoded = IndirectCall.model_validate(json_value(call.arguments))
        except ValidationError as error:
            return f"tool_call 需要 name 和对象类型的 arguments：{error}"
        if decoded.name not in {ref.name for ref in self._view.refs}:
            return f"工具不属于获授 view: {decoded.name}；请用 tool_search 查询当前目录。"
        return decoded.name, decoded.arguments

    def configuration(self, name: str) -> Mapping[str, object] | None:
        return {"groups": self._groups} if name == self._search_ref.name else None


async def apply(ctx: Context, config: object) -> None:
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)

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
        public=False,
        idempotent=True,
        risk="read-only",
    )
    view = catalog.view(search_ref)
    _ = await ctx.provide(TOOL_SEARCH_TOOLS, view)
    _ = await ctx.provide(
        TOOL_SEARCH_PRESENTATION,
        lambda awarded: SearchPresentation(catalog, awarded, search_ref),
    )

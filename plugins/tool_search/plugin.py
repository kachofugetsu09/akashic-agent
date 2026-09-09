from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
import json
import re
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.tools.api import BoundTool, CallSource, InvalidArguments, Result
from plugins.tools.plugin import TOOLS
from session.message import ContentPart
from session.message_codec import json_value

api_version = 3
name = "tool_search"
version = "1.0.0"
desc = "搜索本程序的固定工具目录，选择事实随工具结果保存"
inject = (TOOLS,)


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    allowed_risk: list[Literal["read-only", "read-write", "external-side-effect"]] | None = Field(
        default=None, description="按工具整体能力过滤，通常省略。浏览器等通用工具即使只读页面也可能标为 external-side-effect；此字段不是本次操作的授权。",
    )


def search(candidates: Mapping[str, Mapping[str, object]], query: Query) -> tuple[str, ...]:
    """精确选择或按名称、提示和描述匹配；中文支持子串与双字词。"""
    # 1. 筛选只影响发现，不代替 Tool 执行前的当前授权。
    allowed = {
        name: cast(Mapping[str, object], value["tool"])
        for name, value in candidates.items()
        if query.allowed_risk is None or cast(Mapping[str, object], value["tool"])["risk"] in query.allowed_risk
    }
    text = query.query.strip()
    if text.lower().startswith("select:"):
        return tuple(dict.fromkeys(name.strip() for name in text[7:].split(",") if name.strip() in allowed))[:query.top_k]
    if text in allowed:
        return (text,)
    text = text.lower()
    tokens: set[str] = {text, *text.split(), *(part.strip() for part in re.split(r"([\u4e00-\u9fff]+)", text))}
    cjk = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    tokens.update(cjk)
    tokens.update(left + right for left, right in zip(cjk, cjk[1:]))
    tokens.discard("")

    # 2. 相同公开描述得到相同排序，不为内置或 MCP 来源额外加分。
    ranked: list[tuple[int, str]] = []
    for name, tool in allowed.items():
        parts = name.lower().split("_")
        hint = cast(str | None, tool["search_hint"]) or ""
        description = cast(str, tool["description"]).lower()
        score = 0
        for token in tokens:
            if token in parts:
                score += 10
            elif any(token in part or part in token for part in parts):
                score += 5
            elif token in name.lower():
                score += 3
            if token in hint.lower():
                score += 4
            if token in description:
                score += 2
        if score:
            ranked.append((-score, name))
    return tuple(name for _, name in sorted(ranked)[:query.top_k])


class SearchTool:
    idempotent = True

    def __init__(self, candidates: Mapping[str, Mapping[str, object]]):
        self._candidates = candidates

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        try:
            return Query.model_validate(json_value(arguments)).model_dump(mode="json")
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        query = Query.model_validate(json_value(arguments))
        selected = search(self._candidates, query)
        rows = [self._candidates[name] for name in selected]
        excluded = []
        if query.allowed_risk is not None:
            unrestricted = query.model_copy(update={"allowed_risk": None})
            excluded = [
                {"name": name, "risk": self._candidates[name]["tool"]["risk"]}
                for name in search(self._candidates, unrestricted)
                if self._candidates[name]["tool"]["risk"] not in query.allowed_risk
            ]
        text = json.dumps({
            "matched": [json_value(row["tool"]) for row in rows],
            "selected": selected,
            "excluded_by_risk": excluded,
            "risk_tip": "部分匹配工具被 allowed_risk 排除。若符合任务需要，省略该过滤重搜；执行仍须通过当前授权。" if excluded else "",
            "tip": "菜单按容量载入选中工具，优先保留最相关项；载入后可直接调用。" if selected else "没有匹配工具，请调整关键词或用 select:工具名。",
        }, ensure_ascii=False)
        return Result("success", (
            ContentPart("text", text),
            ContentPart("tool.selection", tuple(row["binding_id"] for row in rows)),
        ))

    async def query(self, key: str) -> Result | None:
        return None


async def apply(ctx: Context, config: object) -> None:
    @asynccontextmanager
    async def open(candidates: Mapping[str, object]) -> AsyncIterator[BoundTool]:
        yield SearchTool(cast(Mapping[str, Mapping[str, object]], candidates))

    _ = await ctx.require(TOOLS).register(
        ctx, name="tool_search", description=(
            "搜索当前允许的工具并解锁。已可见的工具直接调用；不可见时用功能关键词或 "
            "select:工具名 精确选择，可用逗号选择多个。结果中的工具随后可直接调用。"
        ), parameters=Query.model_json_schema(), open=open, discovery=True,
        idempotent=True, risk="read-only", always_on=True,
    )

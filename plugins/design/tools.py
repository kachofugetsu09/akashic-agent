"""Agent 的设计工具：open-pencil 的核心工具按原 schema 暴露，调用经实时通道交给用户打开的画布执行。"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import Context
from plugins.ledger.contract import ContentPart, Input, json_value
from plugins.tools.contract import TOOLS, CallSource, Result

from .live import DESIGN_LIVE, CanvasUnavailable, DesignLive

MANIFEST = "static/tools.json"
MAX_RESULT = 48 * 1024
GROUP = (
    "设计画布（open-pencil）：在用户工作台的“设计”标签里画界面、海报、原型和图示，改动实时出现在用户打开的画布上，"
    "用户可以接着改、可以撤销。先用 design_document 打开或新建文档，再用 design_render 以 JSX 画出结构，"
    "用 design_get_jsx / design_describe 检查结果。只在网页端可用。"
)
UNAVAILABLE = "。设计画布只能在网页端用：请用户在网页里打开这个会话（工作台的“设计”标签会自动加载），然后再试。"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, object]


class DocumentAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["list", "open", "create"] = Field(description="list 列出本会话的设计文档；open 打开一份；create 新建并打开")
    name: str = Field(default="", max_length=80, description="create 时的文档名；open 时也可以按名字打开")
    document: str = Field(default="", max_length=16, description="open 时的文档 ID（来自 list）")


# 构建时从 open-pencil 的工具表生成 static/tools.json：名字、描述与 JSON Schema 与编辑器里执行的是同一份定义。
def manifest(plugin_dir: Path) -> tuple[ToolSpec, ...]:
    path = plugin_dir / MANIFEST
    if not path.is_file():
        return ()
    rows = cast(list[dict[str, object]], json.loads(path.read_text(encoding="utf-8")))
    return tuple(ToolSpec(str(row["name"]), str(row["description"]), cast(Mapping[str, object], row["parameters"]))
                 for row in rows)


class DesignTool:
    idempotent = False

    def __init__(self, live: DesignLive, tool: str) -> None:
        self._live = live
        self._tool = tool

    # 调用属于哪个会话、哪一轮在提交时记下：画布按会话路由，同一轮只自动展开一次设计标签。
    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        if source is None:
            return "设计工具只能由对话里的调用触发"
        target = next((message for message in source.messages if message.message_id == source.call_ref.message_id), None)
        if target is None:
            return "设计工具调用缺少所在消息"
        inputs = [message for message in source.messages if isinstance(message.body, Input) and message.seq <= target.seq]
        values = dict(arguments)
        values["_design"] = {"session": target.session_id, "turn": (inputs[-1] if inputs else target).message_id}
        return values

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        values = cast(dict[str, object], json_value(arguments))
        identity = cast(dict[str, str], values.pop("_design"))
        try:
            reply = await self._live.call(identity["session"], identity["turn"], self._tool, values)
        except CanvasUnavailable as error:
            return Result("error", (ContentPart("text", f"{error}{UNAVAILABLE}"),))
        if reply.get("ok") is not True:
            return Result("error", (ContentPart("text", str(reply.get("error") or "设计工具执行失败")),))
        text = json.dumps(reply.get("result"), ensure_ascii=False)
        if len(text) > MAX_RESULT:
            text = text[:MAX_RESULT] + f"…（结果共 {len(text)} 字符，已截断；请缩小查询范围）"
        return Result("success", (ContentPart("text", text),))

    async def query(self, key: str) -> Result | None:
        return None


async def register_tools(ctx: Context) -> None:
    live = ctx.require(DESIGN_LIVE)
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, description=GROUP)
    specs = (ToolSpec("document", "Open, create or list the design documents of this conversation. "
                      "Every other design tool works on the document open on the canvas; when none is open, the first call creates one.",
                      DocumentAction.model_json_schema()), *manifest(ctx.runtime.plugin_dir))
    for spec in specs:
        tool = DesignTool(live, spec.name)

        @asynccontextmanager
        async def open_tool(_state: Mapping[str, object], tool: DesignTool = tool) -> AsyncIterator[DesignTool]:
            yield tool

        _ = await catalog.register(ctx, name=f"design_{spec.name}", description=spec.description,
                                   parameters=spec.parameters, open=open_tool)

"""Agent 的设计工具：open-pencil 的核心工具按原 schema 暴露，在服务端的无头画布上执行，用户不必开着网页。"""
from __future__ import annotations

import asyncio
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
from plugins.working_directory.contract import WORKING_DIRECTORY, WorkingDirectories

from .headless import HeadlessPages, HeadlessUnavailable
from .live import DESIGN_LIVE
from .snapshot import MAX_BYTES, SnapshotError, build
from .store import DESIGN_STORE, DesignDocument, DesignStore, DesignStoreError
from .worker import DesignWorker, DesignWorkerError

MANIFEST = "static/tools.json"
MAX_RESULT = 48 * 1024
GROUP = (
    "设计画布（open-pencil）：画界面、海报、原型和图示，文档保存在这个会话里，用户在工作台的“设计”标签里看和改。"
    "工具在服务端执行，用户不在线也能做；每次修改直接保存，用户开着画布时会跟进。"
    "先用 design_document 打开或新建文档，再用 design_render 以 JSX 画出结构，用 design_get_jsx / design_describe 检查结果。"
    "给用户看 workspace 里做好的网页（原型、showcase）用 design_show_html。"
)


HTML_DESCRIPTION = (
    "给用户看你做的网页、原型或交互稿时用：传 workspace 里一个 HTML 文件的路径，它会成为“设计”标签里的一份设计文档，"
    "在“预览”里原样运行（用户在自己的浏览器里直接操作）；import_layers=true 时还把页面导入成可编辑的设计图层。"
    "优先用它，不要为了给用户看页面去起 http 服务、开容器，或用电脑/浏览器截图。"
    f"入口同目录或子目录里引用的 css、js、图片、字体会内联进快照（合计 {MAX_BYTES // 1024 // 1024} MB 以内），"
    "外部 https 资源原样加载；只发布入口这一页。改完文件再调用一次，同一份文档的预览会换成新版本。"
)


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


class ShowHtml(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, max_length=4096, description="HTML 文件路径：绝对路径，或相对会话工作目录；必须在 workspace 内")
    title: str = Field(default="", max_length=80, description="文档名；省略时用页面的 <title>")
    import_layers: bool = Field(default=False, description="同时把页面导入成可编辑的设计图层（放在文档的新页面里）")


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

    def __init__(self, worker: DesignWorker, tool: str) -> None:
        self._worker = worker
        self._tool = tool

    # 调用属于哪个会话、哪一轮在提交时记下：文档按会话存放，同一轮只自动展开一次设计标签。
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
            if self._tool == "document":
                request = DocumentAction.model_validate(values)
                reply: dict[str, object] = {"ok": True, "result": await self._worker.document(
                    identity["session"], identity["turn"], request.action, request.name, request.document)}
            else:
                reply = await self._worker.call(identity["session"], identity["turn"], self._tool, values)
        except (DesignWorkerError, HeadlessUnavailable, DesignStoreError) as error:
            return Result("error", (ContentPart("text", str(error)),))
        if reply.get("ok") is not True:
            return Result("error", (ContentPart("text", str(reply.get("error") or "设计工具执行失败")),))
        text = json.dumps(reply.get("result"), ensure_ascii=False)
        if len(text) > MAX_RESULT:
            text = text[:MAX_RESULT] + f"…（结果共 {len(text)} 字符，已截断；请缩小查询范围）"
        return Result("success", (ContentPart("text", text),))

    async def query(self, key: str) -> Result | None:
        return None


class HtmlTool(DesignTool):
    def __init__(self, worker: DesignWorker, store: DesignStore, workspace: Path, directories: WorkingDirectories) -> None:
        super().__init__(worker, "preview")
        self._store = store
        self._workspace = workspace.resolve()
        self._directories = directories

    # 相对路径按会话工作目录解析，与文件工具一致；解析好的绝对路径随调用保存。
    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        prepared = await super().prepare(arguments, source)
        if isinstance(prepared, str) or source is None:
            return prepared
        values = dict(prepared)
        current = self._directories.snapshot(source.messages[-1].session_id).path
        try:
            values["path"] = str(self._entry(str(arguments.get("path", "")), Path(current) if current else self._workspace))
        except SnapshotError as error:
            return str(error)
        return values

    # 1. 快照写成会话里的一份设计文档；2. 需要时在无头画布里导入图层；3. 开着的画布切到这份文档
    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        values = cast(dict[str, object], json_value(arguments))
        identity = cast(dict[str, str], values["_design"])
        request = ShowHtml.model_validate({name: values[name] for name in ("path", "title", "import_layers") if name in values})
        try:
            entry = self._entry(request.path, self._workspace)
            snapshot = await asyncio.to_thread(build, entry, request.title or None)
            document, created = await self._store.publish(identity["session"], str(entry.relative_to(self._workspace)),
                                                          snapshot.title, snapshot.html)
        except (SnapshotError, DesignStoreError, OSError) as error:
            return Result("error", (ContentPart("text", str(error)),))
        lines = [f"已{'发布' if created else '更新'}“{document.name}”（{document.source}，{len(snapshot.html.encode()) / 1024:.0f} KB，文档 {document.id}）。"]
        lines.append(await self._show(identity, document, request.import_layers))
        if snapshot.inlined:
            lines.append("已内联：" + "、".join(dict.fromkeys(snapshot.inlined)))
        if snapshot.missing:
            lines.append("没找到、保留原引用：" + "、".join(dict.fromkeys(snapshot.missing)))
        return Result("success", (ContentPart("text", "\n".join(line for line in lines if line)),))

    # 导入失败不影响已经发布的预览，原因随结果交给 Agent
    async def _show(self, identity: dict[str, str], document: DesignDocument, import_layers: bool) -> str:
        if not import_layers:
            await self._worker.preview(identity["session"], identity["turn"], document.id)
            return "用户的“设计”标签里可以看网页预览；开着画布的话已经切过去。"
        try:
            count = await self._worker.import_html(identity["session"], identity["turn"], document)
        except (DesignWorkerError, HeadlessUnavailable, DesignStoreError) as error:
            return f"网页导入成图层没有完成：{error}"
        return f"已把网页导入成 {count} 个设计图层（整页是一个顶层框架）；用户可以在“预览”里看原网页。"

    # 入口必须是 workspace 里真实存在的 HTML 文件；先跟随符号链接再判断，链接指到外面同样拒绝。
    def _entry(self, raw: str, base: Path) -> Path:
        entry = (base / Path(raw).expanduser()).resolve()
        if not entry.is_relative_to(self._workspace):
            raise SnapshotError(f"只能发布 workspace（{self._workspace}）里的文件；{entry} 不在其中（含符号链接指向外部）")
        if not entry.is_file():
            raise SnapshotError(f"文件不存在：{entry}")
        if entry.suffix.lower() not in {".html", ".htm"}:
            raise SnapshotError(f"入口需要是 .html 文件：{entry.name}")
        return entry


async def register_tools(ctx: Context) -> None:
    store = ctx.require(DESIGN_STORE)
    worker = DesignWorker(store, ctx.require(DESIGN_LIVE), HeadlessPages(), ctx.runtime.plugin_dir, ctx.runtime.data_dir)
    _ = await ctx.effect(lambda: worker.aclose, label="design-worker")
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, description=GROUP)
    specs = (ToolSpec("document", "Open, create or list the design documents of this conversation. "
                      "Every other design tool works on the document you last opened or created here (by default the most recently edited one; "
                      "the first call creates one when the conversation has none).",
                      DocumentAction.model_json_schema()), *manifest(ctx.runtime.plugin_dir))
    for spec in specs:
        tool = DesignTool(worker, spec.name)

        @asynccontextmanager
        async def open_tool(_state: Mapping[str, object], tool: DesignTool = tool) -> AsyncIterator[DesignTool]:
            yield tool

        _ = await catalog.register(ctx, name=f"design_{spec.name}", description=spec.description,
                                   parameters=spec.parameters, open=open_tool)
    html = HtmlTool(worker, store, ctx.runtime.workspace, ctx.require(WORKING_DIRECTORY))

    @asynccontextmanager
    async def open_html(_state: Mapping[str, object]) -> AsyncIterator[HtmlTool]:
        yield html

    _ = await catalog.register(ctx, name="design_show_html", description=HTML_DESCRIPTION,
                               parameters=ShowHtml.model_json_schema(), open=open_html, parallel=True)

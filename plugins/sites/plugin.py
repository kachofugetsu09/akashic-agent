"""show_site：把 workspace 里的网页发布成不可变快照，在用户右侧边栏的“网站”标签里打开。"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.ledger.contract import ARTIFACT_IMPORT, ArtifactImport, AttachmentKind, ContentPart, json_value
from plugins.working_directory.contract import WORKING_DIRECTORY, WorkingDirectories
from plugins.tools.contract import TOOLS, CallSource, Result
from plugins.ui.contract import UI, UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest

from . import dashboard
from .snapshot import MAX_BYTES, SnapshotError, build

api_version = 3
name = "sites"
version = "1.0.0"
desc = "把 Agent 做的网页（原型、设计稿、showcase）发布到用户右侧边栏，用户在自己的浏览器里直接操作"
inject = (TOOLS, ARTIFACT_IMPORT, WORKING_DIRECTORY)

DESCRIPTION = (
    "给用户看你做的网页、原型、设计稿或交互稿时用：传 workspace 里一个 HTML 文件的路径，"
    "发布一份不可变快照，并在用户右侧边栏的“网站”标签打开，用户在自己的浏览器里直接操作。"
    "优先用它，不要为了给用户看页面去起 http 服务、开容器，或用电脑/浏览器截图。"
    f"入口同目录或子目录里引用的 css、js、图片、字体会内联进快照（合计 {MAX_BYTES // 1024 // 1024} MB 以内），"
    "外部 https 资源（CDN、字体）原样加载；只发布入口这一页，页面之间的相对链接不可用。"
    "改完文件再调用一次，用户已经打开的同一个标签会换成新版本。"
)


class Site(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, max_length=4096, description="HTML 文件路径：绝对路径，或相对会话工作目录；必须在 workspace 内")
    title: str = Field(default="", max_length=120, description="标签与卡片上的标题；省略时用页面的 <title>")


class SiteTool:
    idempotent = False

    def __init__(self, workspace: Path, directories: WorkingDirectories, artifacts: ArtifactImport) -> None:
        self._workspace = workspace.resolve()
        self._directories = directories
        self._artifacts = artifacts

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        try:
            site = Site.model_validate(json_value(arguments))
        except ValidationError as error:
            return str(error)
        # 相对路径按会话工作目录解析，与文件工具一致；没有工作目录时按 workspace。
        current = None if source is None else self._directories.snapshot(source.messages[-1].session_id).path
        try:
            entry = self._entry(site.path, Path(current) if current else self._workspace)
        except SnapshotError as error:
            return str(error)
        return {"path": str(entry), "title": site.title.strip()}

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        site = Site.model_validate(json_value(arguments))
        try:
            entry = self._entry(site.path, self._workspace)
            snapshot = await asyncio.to_thread(build, entry, site.title or None)
        except (SnapshotError, OSError) as error:
            return Result("error", (ContentPart("text", str(error)),))
        published = datetime.now(timezone.utc)
        relative = str(entry.relative_to(self._workspace))
        content = json.dumps({"schema": "akashic.site.v1", "title": snapshot.title, "path": relative,
                              "published_at": published.isoformat(), "html": snapshot.html}, ensure_ascii=False)
        # 附件导入是唯一持久效果：每次调用一份新快照，历史消息打开的仍是当时的版本。
        with TemporaryDirectory(prefix="akashic-site-") as directory:
            filename = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "-", snapshot.title).strip(" .")[:60] or "site"
            path = Path(directory) / f"{filename}.akashic-site.json"
            _ = await asyncio.to_thread(path.write_text, content, encoding="utf-8")
            artifact = await self._artifacts.import_source(str(path), AttachmentKind.FILE)
        size = len(snapshot.html.encode("utf-8"))
        lines = [f"已发布“{snapshot.title}”（{relative}，{size / 1024:.0f} KB，"
                 f"版本 {published.astimezone():%H:%M:%S}，快照 {artifact.artifact_id[:8]}）。",
                 "网页聊天会在用户右侧边栏的“网站”标签打开它，用户可以直接操作；改完文件再调用 show_site，同一个标签会换成新版本。"]
        if snapshot.inlined:
            lines.append("已内联：" + "、".join(dict.fromkeys(snapshot.inlined)))
        if snapshot.missing:
            lines.append("没找到、保留原引用：" + "、".join(dict.fromkeys(snapshot.missing)))
        return Result("success", (ContentPart("text", "\n".join(lines)), ContentPart("artifact_ref", artifact.artifact_id)))

    async def query(self, key: str) -> Result | None:
        # 没有重放凭据时不重新发布；tools owner 负责保留已提交的结果。
        return None

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


async def apply(ctx: Context) -> None:
    tool = SiteTool(ctx.runtime.workspace, ctx.require(WORKING_DIRECTORY), ctx.require(ARTIFACT_IMPORT))

    @asynccontextmanager
    async def open_tool(_state: Mapping[str, object]) -> AsyncIterator[SiteTool]:
        yield tool

    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)
    _ = await catalog.register(ctx, name="show_site", description=DESCRIPTION,
                               parameters=Site.model_json_schema(), open=open_tool, parallel=True)

    # UI 缺席时工具照常发布：对话里保留结果文字与快照下载，只是没有卡片和隔离入口。
    async def register_ui(child: Context) -> None:
        _ = await child.require(UI).register(child, dashboard=lambda: dashboard)

        def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> object:
            raise PluginUiRpcInvalidRequest("网站卡片只读取消息附件，不提供 UI RPC")

        _ = await child.require(UI_SLOTS).register_plugin_ui(
            child, PluginUiDefinition(module="message_ui.js", stylesheet="message_ui.css", slots=("message.attachment",)),
            query=query,
        )

    _ = await ctx.inject((UI, UI_SLOTS), register_ui, name="sites-ui")

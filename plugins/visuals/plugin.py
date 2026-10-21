"""生成不可变交互附件；工具与可选聊天展示各自拥有生命周期。"""
from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from hashlib import sha256
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from plugins.assets.contract import INSTALLED_ASSETS
from plugins.ledger.contract import ARTIFACT_IMPORT, ArtifactImport, AttachmentKind, ContentPart, json_value
from plugins.tools.contract import TOOLS, CallSource, Result
from plugins.ui.contract import SETTINGS_SLOT, UI, UI_SLOTS, ModelSettings, PluginUiDefinition, PluginUiRpcInvalidRequest

from . import dashboard
from .preview import Preview, PreviewFailed, PreviewUnavailable

api_version = 3
name = "visuals"
version = "1.1.0"
desc = "生成可交互的图表、解释和模拟，作为 HTML 消息附件保存"
inject = (PLUGIN_CONFIG, TOOLS, ARTIFACT_IMPORT, INSTALLED_ASSETS)
workspace_roots = ("visuals",)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    executable_path: str | None = None


class Visual(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str = Field(min_length=1, max_length=120, description="简短中文标题")
    summary: str = Field(min_length=1, max_length=2000, description="一两句话的结论，只在无法显示交互图时代替图出现；不要写成后文正文")
    # 写法随 visualize 技能按需读取，不常驻工具 schema。
    html: str = Field(min_length=1, max_length=200_000, description=(
        "完整 HTML body 片段，可含内联 style/script、SVG 和 Canvas；不依赖网络与外部库。"
        "写法见 visualize 技能：数据图读 charts.md，示意图读 diagrams.md，交互讲解与卡片读 interactive.md。"
    ))
    draft: bool = Field(default=False, description=(
        "true 时只渲染并回传截图，不发布给用户；看过截图、改好后再省略本参数正式发布。正式发布后用户就会看到这张图，再调用会多出一张。"
    ))


class VisualTool:
    idempotent = False

    def __init__(self, preview: Preview, artifacts: ArtifactImport, root: Path) -> None:
        self._preview = preview
        self._artifacts = artifacts
        self._root = root
        # 默认主题放进 CSS layer：单独打开文件时可读，聊天宿主注入的主题始终优先。
        plugin_dir = Path(__file__).parent
        self._template = (plugin_dir / "frame.html").read_text(encoding="utf-8").replace(
            "/*visual-theme*/", "@layer ak-default{" + (plugin_dir / "preview-theme.css").read_text(encoding="utf-8") + "}")

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        try:
            return Visual.model_validate(json_value(arguments)).model_dump()
        except ValidationError as error:
            return str(error)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """预检完成后保存自包含 HTML 文件并发布为附件；草稿只回传截图；失败沿普通工具结果返回模型。"""
        visual = Visual.model_validate(json_value(arguments))
        document = self._template.replace("<title></title>", f"<title>{escape(visual.title)}</title>", 1).replace(
            "<!--visual-content-->", visual.html, 1)
        try:
            screenshot = await self._preview.check(document)
        except (PreviewFailed, PreviewUnavailable) as error:
            return Result("error", (ContentPart("text", str(error)),))
        name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "-", visual.title).strip(" .")[:60] or "visual"
        stem = f"{name}-{sha256(key.encode()).hexdigest()[:8]}"
        # 1. 草稿不写文件也不交付附件：结果里只有截图，宿主把它留在调用卡片，用户界面不多出一张图。
        image = await self._import_screenshot(stem, screenshot)
        if visual.draft:
            return Result("success", (
                ContentPart("text", f"### 草稿：{visual.title}\n\n草稿没有展示给用户。附图是 720px 宽的渲染截图；"
                            "按 visualize 技能的清单检查，改好后省略 draft 正式发布同一张图。"),
                ContentPart("artifact_ref", image),
            ))
        # 2. 文件留在 workspace，模型可以把同一路径交给其他工具推送或分享。
        path = self._root / f"{stem}.html"
        await asyncio.to_thread(path.write_text, document, encoding="utf-8")
        artifact = await self._artifacts.import_source(str(path), AttachmentKind.FILE)
        return Result("success", (
            ContentPart("text", f"### {visual.title}\n\n{visual.summary}\n\nHTML 文件：{path}\n\n"
                        "交互图已显示给用户，正文不要复述图里已有的数字和标签，只补充图外的含义。"
                        "附图是 720px 宽的预检截图；不要为改善外观再次发布，否则用户会看到两张图。"),
            ContentPart("artifact_ref", artifact.artifact_id),
            ContentPart("artifact_ref", image),
        ))

    # 预检截图是给模型自检的观察；宿主把同一结果里的图片留在调用卡片，用户只看到交互图。
    async def _import_screenshot(self, stem: str, screenshot: bytes) -> str:
        with TemporaryDirectory(prefix="akashic-visual-") as directory:
            shot = Path(directory) / f"{stem}.png"
            await asyncio.to_thread(shot.write_bytes, screenshot)
            return (await self._artifacts.import_source(str(shot), AttachmentKind.IMAGE)).artifact_id

    async def query(self, key: str) -> Result | None:
        # 没有重放凭据时不重新发布；tools owner 负责保留已提交的结果。
        return None


async def apply(ctx: Context) -> None:
    """工具依赖附件导入；聊天 UI 可单独出现、卸载和重新出现。"""
    config = Config.model_validate(ctx.config)
    preview = Preview(config.executable_path)
    await ctx.provide(dashboard.SETTINGS, ModelSettings(ctx, Config, ("executable_path",)))
    await ctx.effect(lambda: preview.close, label="visual-preview")
    root = ctx.workspace_root("visuals")
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
    tool = VisualTool(preview, ctx.require(ARTIFACT_IMPORT), root)

    @asynccontextmanager
    async def open_tool(_state: Mapping[str, object]) -> AsyncIterator[VisualTool]:
        yield tool

    # 工具组按需解锁：visualize 技能说明什么时候画、怎么画，用到时再 load_tools。
    await ctx.require(INSTALLED_ASSETS).register(ctx, "skills", "skills")
    catalog = ctx.require(TOOLS)
    await catalog.declare_group(ctx, description=desc)
    await catalog.register(ctx, name="render_visual", description=(
        "在图表、关系或参数试验能帮助理解时，生成可交互附件。先读 visualize 技能，先解释直觉再调用；"
        "成功后可继续说明。" + desc),
                           parameters=Visual.model_json_schema(), open=open_tool, parallel=True)

    async def register_ui(child: Context) -> None:
        await child.require(UI).register(child, web="web_module.js", dashboard=lambda: dashboard,
                                         requires=("shell.settings-plugins.v1",), contract_digests=SETTINGS_SLOT)

        def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> object:
            raise PluginUiRpcInvalidRequest("可视化只读取消息附件，不提供 UI RPC")

        await child.require(UI_SLOTS).register_plugin_ui(child,
            PluginUiDefinition(module="message_ui.js", slots=("message.attachment",)), query=query)

    await ctx.inject((UI, UI_SLOTS, dashboard.SETTINGS), register_ui, name="visual-ui")

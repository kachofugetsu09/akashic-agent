"""生成不可变交互附件；工具与可选聊天展示各自拥有生命周期。"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.ledger.contract import ARTIFACT_IMPORT, ArtifactImport, AttachmentKind, ContentPart, json_value
from plugins.tools.contract import TOOLS, CallSource, Result
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest

from .preview import Preview, PreviewFailed, PreviewUnavailable

api_version = 3
name = "visuals"
version = "1.0.0"
desc = "生成可交互的图表、解释和模拟，作为消息附件保存"
inject = (TOOLS, ARTIFACT_IMPORT)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    executable_path: str | None = None


class Visual(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str = Field(min_length=1, max_length=120, description="简短中文标题")
    summary: str = Field(min_length=1, max_length=2000, description="一两句话的结论，只在无法显示交互图时代替图出现；不要写成后文正文")
    html: str = Field(min_length=1, max_length=200_000, description=(
        "完整 HTML body 片段，可含内联 style/script、SVG 和 Canvas；自己编写 HTML/CSS/JavaScript，按内容选择 SVG、"
        "Canvas 或普通 DOM。不得依赖网络、外部库、宿主 API、iframe、表单提交或导航。"
        "文档已有与聊天一致的默认样式，直接使用、不要重写：h2 图标题，p.caption 副标题，svg.chart 图区（默认高 280px），"
        ".grid/.axis 网格与轴线，svg 文字默认是小号刻度字，div.legend > span > i[style=--series:var(--ak-chart-N)] 图例，"
        "label.param（文字 + output + input[type=range]）参数行，div.controls 按钮组，button.primary 主按钮，"
        "p.readout 当前读数，p.note 注释。背景透明、不加卡片外框，不重复正文标题。"
        "图形和读数必须由真实数据或公式计算，参数改变立即更新；刻度、单位、图例和悬停读数由文档自己实现，"
        "边缘刻度用 text-anchor start/end 留在图内。适应 280px 与 720px 宽度。"
        "系列色只用 --ak-chart-1 到 --ak-chart-6，优先 2–3 个系列，颜色只用于数据线、点和面；"
        "文字和线条用 --ak-color-text-primary/secondary/muted、--ak-color-border-default。"
        "每图最多 12 个滑块，必须带可访问名称和当前值；文本用 textContent，避免 NaN/Infinity。"
        "静态解释先输出正文，再调用本工具，返回后继续说明含义，不复述摘要。每一步最多并行调用两次。"
    ))


class VisualTool:
    idempotent = False

    def __init__(self, preview: Preview, artifacts: ArtifactImport) -> None:
        self._preview = preview
        self._artifacts = artifacts

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        try:
            return Visual.model_validate(json_value(arguments)).model_dump()
        except ValidationError as error:
            return str(error)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """预检完成后发布单个附件；失败沿普通工具结果返回模型。"""
        visual = Visual.model_validate(json_value(arguments))
        try:
            await self._preview.check(visual.html)
        except (PreviewFailed, PreviewUnavailable) as error:
            return Result("error", (ContentPart("text", str(error)),))
        # 附件导入是唯一持久效果；插件没有消息改写或附件删除权限。
        with TemporaryDirectory(prefix="akashic-visual-") as directory:
            filename = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "-", visual.title).strip(" .")[:60] or "visual"
            path = Path(directory) / f"{filename}.akashic-visual.json"
            content = json.dumps({"schema": "akashic.visual.v1", **visual.model_dump()}, ensure_ascii=False)
            await asyncio.to_thread(path.write_text, content, encoding="utf-8")
            artifact = await self._artifacts.import_source(str(path), AttachmentKind.FILE)
        return Result("success", (ContentPart("text", f"### {visual.title}\n\n{visual.summary}"), ContentPart("artifact_ref", artifact.artifact_id)))

    async def query(self, key: str) -> Result | None:
        # 没有重放凭据时不重新发布；tools owner 负责保留已提交的结果。
        return None


async def apply(ctx: Context) -> None:
    """工具依赖附件导入；聊天 UI 可单独出现、卸载和重新出现。"""
    config = Config.model_validate(ctx.config)
    preview = Preview(config.executable_path)
    await ctx.effect(lambda: preview.close, label="visual-preview")
    tool = VisualTool(preview, ctx.require(ARTIFACT_IMPORT))

    @asynccontextmanager
    async def open_tool(_state: Mapping[str, object]) -> AsyncIterator[VisualTool]:
        yield tool

    catalog = ctx.require(TOOLS)
    await catalog.declare_group(ctx, always_on=True, description=desc)
    await catalog.register(ctx, name="render_visual", description="在图表、关系或参数试验能帮助理解时，生成可交互附件。先解释直觉，再调用；成功后可继续说明。" + desc,
                           parameters=Visual.model_json_schema(), open=open_tool, parallel=True)

    async def register_ui(child: Context) -> None:
        def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> object:
            raise PluginUiRpcInvalidRequest("可视化只读取消息附件，不提供 UI RPC")

        await child.require(UI_SLOTS).register_plugin_ui(child,
            PluginUiDefinition(module="message_ui.js", slots=("message.attachment",)), query=query)

    await ctx.inject((UI_SLOTS,), register_ui, name="visual-ui")

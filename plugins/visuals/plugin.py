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
from plugins.ui.contract import UI, UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest

from . import dashboard
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
        "完整 HTML body 片段，可含内联 style/script、SVG 和 Canvas；不得依赖网络、外部库、宿主 API、iframe、表单提交或导航。"
        "背景透明、不加卡片外框，不重复正文标题；适应 280px 与 720px 宽度，不横向溢出。"
        "回答涉及具体数字时写编辑式数据图：h3.headline 标题直接写结论（结论词用 em），可选 p.dek 一句补充；"
        "1–4 个关键数字用 div.hero > div > b.v（单位放 small）+ span.k 短标签 + span.d 说明，只有一两个数时只用 hero 不画图；"
        "每张图上方 p.chart-title（可加 p.chart-sub），峰值、拐点和阈值直接注释在图上；明细用 table.dtable（数字列的 th 和 td 都加 n，合计放 tfoot），"
        "div.split 左图右表，p.foot 写口径与来源。"
        "数据图用宿主内置的 ak，不要自己画坐标轴：放空 div，在最后的 script 里调用。"
        "ak.bars('#id', {x, series:[{name, values}], stack, annotate:[{at, text, sub}], refLine:{value, label}}) 画分类或按天计数，默认堆叠；"
        "加 horizontal:true、sub、range:[{lo, hi}] 画估计区间与阈值。"
        "ak.line('#id', {x, series, milestones:[{at, text, sub}], spans:[{from, to, label}], annotate:[{at, text, series}], projectFrom, yMin}) 画趋势，"
        "多系列在线尾自动标名，projectFrom 之后画虚线外推。ak.share('#id', {x, series, annotate}) 画构成随时间的 100% 份额。"
        "ak.scatter('#id', {points:[{x, y, label, size, group}], xTitle, yTitle, xFmt, yFmt, highlight, refX, refY, trend}) 画两个量的关系，size 变气泡；"
        "ak.dumbbell('#id', {x, from:{name, values}, to:{name, values}}) 画每行前后或两种口径的对比；"
        "ak.heatmap('#id', {rows, cols, values}) 画星期 × 小时这类两个维度的规律；ak.treemap('#id', {nodes:[{name, value, group}]}) 画层级与大小；"
        "ak.barlist('#id', {items:[{label, value, note}], limit}) 画排名；ak.spark('#id', values) 是放进 hero 或表格的小趋势线。通用参数 fmt 取 'raw'、'%'、'wan'、'si' 或函数，unit 是单位，label 是可访问名称；"
        "series 可写 color: 1–6、'gray' 或 'hot'。x 写 YYYY-MM-DD 会自动缩写；ak.fmt(value, fmt, unit) 让 hero 和表格用同一格式。"
        "ak 对不一致的数据直接报错，按报错修正。"
        "原理、机制、流程和参数试验自己编写 HTML/CSS/JavaScript，按内容选择 SVG、Canvas 或普通 DOM，可用默认样式："
        "h2 图标题，p.caption 副标题，svg.chart 图区（默认高 280px），.grid/.axis 网格与轴线，"
        "div.legend > span > i[style=--series:var(--k1)] 图例，label.param（文字 + output + input[type=range]）参数行，"
        "div.controls 按钮组，button.primary 主按钮，p.readout 当前读数，p.note 注释；边缘刻度用 text-anchor start/end 留在图内。"
        "系列色按顺序用 --k1 到 --k6，--k-gray 表示其他，最多 4 类；文字和线条用 --ak-color-text-primary/secondary/muted、--ak-color-border-default。"
        "图形和读数必须由真实数据或公式计算，参数改变立即更新；每图最多 12 个滑块，必须带可访问名称和当前值；文本用 textContent，避免 NaN/Infinity。"
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
        await child.require(UI).register(child, dashboard=lambda: dashboard)

        def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> object:
            raise PluginUiRpcInvalidRequest("可视化只读取消息附件，不提供 UI RPC")

        await child.require(UI_SLOTS).register_plugin_ui(child,
            PluginUiDefinition(module="message_ui.js", slots=("message.attachment",)), query=query)

    await ctx.inject((UI, UI_SLOTS), register_ui, name="visual-ui")

"""生成不可变交互附件；工具与可选聊天展示各自拥有生命周期。"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.ledger.contract import ARTIFACT_IMPORT, ArtifactImport, AttachmentKind, ContentPart, json_value
from plugins.tools.contract import TOOLS, CallSource, Result
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest

from .preview import Preview, PreviewFailed

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
    summary: str = Field(min_length=1, max_length=2000, description="图的主要结论；无法显示交互图时仍可阅读")
    html: str = Field(min_length=1, max_length=200_000, description=(
        "完整 HTML body 片段，可含内联 style/script、SVG 和 Canvas。"
        "不得依赖网络、外部库、宿主 API、iframe、表单提交或导航。"
        "使用原生 JS 和响应式布局，适应 280px 与 720px；图表固定高度，控件至少 44px。"
        "用 --ak-chart-1 到 --ak-chart-6 配色；文字、背景、线条使用 --ak-color-text-primary、"
        "--ak-color-text-secondary、--ak-color-bg-canvas、--ak-color-bg-surface、--ak-color-border-default。"
        "每图最多 12 个滑块，必须带可访问名称和当前值，改变参数应立即更新真实计算结果。"
        "保留图例、单位、轴标签；文本用 textContent，避免数值 NaN/Infinity。"
        "静态解释先输出正文，再调用本工具，返回后可继续解释。"
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
        except PreviewFailed as error:
            return Result("error", (ContentPart("text", str(error)),))
        # 附件导入是唯一持久效果；插件没有消息改写或附件删除权限。
        with TemporaryDirectory(prefix="akashic-visual-") as directory:
            path = Path(directory) / "visual.akashic-visual.json"
            content = json.dumps({"schema": "akashic.visual.v1", **visual.model_dump()}, ensure_ascii=False)
            await asyncio.to_thread(path.write_text, content, encoding="utf-8")
            artifact = await self._artifacts.import_source(str(path), AttachmentKind.FILE)
        return Result("success", (ContentPart("text", visual.summary), ContentPart("artifact_ref", artifact.artifact_id)))

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

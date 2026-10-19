"""让只接收文本的主模型通过已配置的 vision 角色查看图片。"""
from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.host_execution.contract import FILES, FileError, FileImage
from plugins.ledger.contract import ContentPart, json_value
from plugins.models.contract import CHAT_MODELS, ModelError, ModelRequest

from plugins.working_directory.contract import WORKING_DIRECTORY

from ._tool_boundary import TOOLS, CallSource, ToolResultValue


class VisionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, description="图片文件路径，与 read_file 使用相同目录规则。")
    prompt: str = Field(min_length=1, description="希望从图片确认的问题，例如文字、目标位置或控件状态。")


class VisionTool:
    idempotent = False

    def __init__(self, ctx: Context):
        self._ctx = ctx

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        """在输入边界检查参数；有工作目录插件时按会话目录解析，否则交给 Files 按 workspace 解析。"""
        try:
            values = VisionInput.model_validate(json_value(arguments))
            session = None if source is None else source.messages[-1].session_id
            with self._ctx.borrow(WORKING_DIRECTORY) as directories:
                if directories is None:
                    return {"path": values.path, "prompt": values.prompt}
                path, _ = await directories.resolve_target(
                    session, values.path, legacy_base=str(self._ctx.runtime.workspace),
                )
            return {"path": path, "prompt": values.prompt}
        except (ValidationError, ValueError) as error:
            return str(error)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        """用正式文件接口读取图片，再把视觉回复作为文本交回主模型。"""
        values = VisionInput.model_validate(json_value(arguments))
        # 1. 图片格式、大小和路径权限继续由 Files owner 检查。
        operation = self._ctx.require(FILES).open("read_file", Path(self._ctx.runtime.workspace))
        try:
            image = await operation.execute(path=values.path)
        finally:
            await operation.aclose()
        if isinstance(image, FileError):
            return ToolResultValue("error", (ContentPart("text", image.text),))
        if not isinstance(image, FileImage):
            return ToolResultValue("error", (ContentPart("text", "该文件不是图片，请使用 read_file 查看文本。"),))
        uri = f"data:{image.mime_type};base64,{base64.b64encode(image.data).decode()}"
        # 2. 沿用 Models 冻结的角色选择；没有可用视觉模型时明确失败。
        try:
            async with self._ctx.require(CHAT_MODELS).execution() as execution:
                response = await execution.chat("vision").complete(ModelRequest(
                    messages=({"role": "user", "content": [
                        {"type": "text", "text": values.prompt},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ]},),
                    max_output_tokens=2048, disable_reasoning=True,
                    request_key="read-image-vision:" + key,
                ))
        except (RuntimeError, TimeoutError) as error:
            failure = ModelError.read(error)
            if failure is None:
                raise
            return ToolResultValue("error", (ContentPart("text", f"视觉模型调用失败：{failure}"),))
        if not response.content:
            return ToolResultValue("error", (ContentPart("text", "视觉模型没有返回可用正文。"),))
        return ToolResultValue("success", (ContentPart("text", response.content),))

    async def query(self, key: str) -> ToolResultValue | None:
        return None


async def register_vision(ctx: Context) -> None:
    @asynccontextmanager
    async def open_tool(_state: Mapping[str, object]) -> AsyncIterator[VisionTool]:
        yield VisionTool(ctx)

    await ctx.require(TOOLS).register(
        ctx, name="read_image_vision",
        description="主模型无法直接看图（结果提示未提供图片内容）时，通过 vision 角色读取图片并返回文本。提供图片 path 和具体 prompt；没有可用视觉模型会明确失败。",
        parameters=VisionInput.model_json_schema(), open=open_tool,
    )

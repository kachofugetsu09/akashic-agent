from __future__ import annotations

import base64
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_validator

from agent.plugin_composition import Context
from agent.plugin_composition.artifacts import ARTIFACT_IMPORT
from agent.tools.base import Tool, normalize_tool_parameters
from agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from plugins.tools.api import CallSource, InvalidArguments, Result
from plugins.tools.plugin import TOOLS
from session.artifacts import AttachmentKind
from session.message import ContentPart
from session.message_codec import json_value

FileBackend = ReadFileTool | ListDirTool | WriteFileTool | EditFileTool


class FileSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    allowed_dir: str | None = None

    @field_validator("allowed_dir")
    @classmethod
    def absolute_path(cls, value: str | None) -> str | None:
        if value is not None and not Path(value).is_absolute():
            raise ValueError("allowed_dir 必须是绝对路径")
        return value


def prepare_arguments(tool: Tool, arguments: Mapping[str, object]) -> Mapping[str, object]:
    """参数只在物理工具的 schema 边界校验一次；之后使用同一最终值。"""
    raw = cast(dict[str, Any], json_value(arguments))
    errors = tool.validate_params(raw, schema=normalize_tool_parameters(tool.parameters))
    if errors:
        raise InvalidArguments("; ".join(errors))
    return raw


class FileTool:
    idempotent = False

    def __init__(self, ctx: Context, backend: FileBackend):
        self._ctx = ctx
        self._backend = backend

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        return prepare_arguments(self._backend, arguments)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """读取真实结果，保留明确错误和可由 Model 投影的图片附件。"""
        # 1. 不在文件工具内根据当前主模型丢弃图片，也不按提示文字猜成功。
        raw = cast(dict[str, Any], json_value(arguments))
        value = (
            await self._backend.read_raw(**raw) if isinstance(self._backend, ReadFileTool)
            else await self._backend.execute(**raw)
        )
        if isinstance(value, str):
            return Result("success", (ContentPart("text", value),))
        if value.mobile_attention is not None or value.runtime_provenance:
            raise ValueError("文件后端返回了未声明的交互或来源字段")
        parts = [ContentPart("text", value.text)] if value.text else []
        # 2. 保存后端实际返回的 model-safe 图片；临时文件不是权威 Artifact。
        for block in value.content_blocks:
            parts.append(await self._import_image(block))
        return Result("error" if value.is_error else "success", tuple(parts))

    async def _import_image(self, block: Mapping[str, object]) -> ContentPart:
        image = block.get("image_url")
        if block.get("type") != "image_url" or not isinstance(image, Mapping):
            raise ValueError("文件后端返回了不支持的内容块")
        uri = cast(Mapping[str, object], image).get("url")
        if not isinstance(uri, str):
            raise TypeError("文件图片缺少 data URI")
        header, separator, data = uri.partition(",")
        suffixes = {"data:image/png;base64": ".png", "data:image/jpeg;base64": ".jpg",
                    "data:image/webp;base64": ".webp", "data:image/gif;base64": ".gif"}
        if not separator or header not in suffixes:
            raise ValueError("文件图片 data URI 格式无效")
        image_bytes = base64.b64decode(data, validate=True)
        with TemporaryDirectory(prefix="akashic-file-image-") as folder:
            path = Path(folder) / ("image" + suffixes[header])
            _ = path.write_bytes(image_bytes)
            ref = await self._ctx.require(ARTIFACT_IMPORT).import_source(str(path), AttachmentKind.IMAGE)
        return ContentPart("artifact_ref", ref.artifact_id)

    async def query(self, key: str) -> Result | None:
        return None


async def register_file(ctx: Context, backend_type: type[FileBackend], *, allowed_dir: Path | None) -> None:
    """注册 schema 和配置；实际文件/Bridge 只在已打开工具中访问。"""
    prototype = backend_type(enable_bridge=False)

    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        return FileSettings.model_validate({
            "allowed_dir": None if allowed_dir is None else str(allowed_dir), **configuration,
        }).model_dump()

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[FileTool]:
        settings = FileSettings.model_validate(json_value(state))
        backend = backend_type(allowed_dir=None if settings.allowed_dir is None else Path(settings.allowed_dir))
        try:
            yield FileTool(ctx, backend)
        finally:
            await backend.aclose()

    description = (
        "读取文件。文本带行号，支持 offset/limit 分页；图片保存为附件并交给当前模型查看。"
        if backend_type is ReadFileTool else prototype.description
    )
    _ = await ctx.require(TOOLS).register(
        ctx, name=prototype.name, description=description,
        parameters=normalize_tool_parameters(prototype.parameters), open=open_tool, capture=capture,
        risk="read-only" if backend_type in (ReadFileTool, ListDirTool) else "read-write", always_on=True,
    )

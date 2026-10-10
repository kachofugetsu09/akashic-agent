from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, field_validator

from agent.plugin_composition import Context
from plugins.host_execution.contract import FILES, FileError, FileImage
from agent.plugin_composition.artifacts import ARTIFACT_IMPORT
from agent.plugin_composition.artifacts import AttachmentKind
from plugins.tools.contract import (
    normalize_tool_parameters,
    validate_tool_parameters,
)
from .filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from agent.plugin_contracts import ContentPart
from agent.plugin_contracts import json_value

from ._tool_boundary import CallSource, TOOLS, ToolRef, ToolResultValue
from .working_directory import WorkingDirectories

FileBackend = ReadFileTool | ListDirTool | WriteFileTool | EditFileTool


class _FileBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, object]: ...


class FileSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    allowed_dir: str | None = None
    session_base: bool = False

    @field_validator("allowed_dir")
    @classmethod
    def absolute_path(cls, value: str | None) -> str | None:
        if value is not None and not Path(value).is_absolute():
            raise ValueError("allowed_dir 必须是绝对路径")
        return value


def prepare_arguments(tool: _FileBackend, arguments: Mapping[str, object]) -> Mapping[str, object] | str:
    """参数只在物理工具的 schema 边界校验一次；之后使用同一最终值。"""
    raw = cast(dict[str, Any], json_value(arguments))
    errors = validate_tool_parameters(raw, schema=normalize_tool_parameters(tool.parameters))
    if errors:
        return '; '.join(errors)
    return raw


class FileTool:
    idempotent = False

    def __init__(self, ctx: Context, backend: FileBackend, *,
                 directories: WorkingDirectories | None = None, settings: FileSettings | None = None):
        self._ctx = ctx
        self._backend = backend
        self._directories = directories
        self._settings = settings or FileSettings()

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        prepared = prepare_arguments(self._backend, arguments)
        if isinstance(prepared, str) or self._directories is None:
            return prepared
        session_id = None if source is None else source.messages[-1].session_id
        allowed = self._settings.allowed_dir
        if session_id is not None and self._settings.session_base:
            current = self._directories.snapshot(session_id)
            if current.path is not None:
                allowed = current.path
        try:
            target, required_dir = await self._directories.resolve_target(
                session_id, cast(str, prepared["path"]), legacy_base=allowed,
            )
        except ValueError as error:
            return str(error)
        return {**prepared, "path": target, "_allowed_dir": allowed,
                **({"required_dir": required_dir} if isinstance(self._backend, WriteFileTool) and required_dir is not None else {})}

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        """读取真实结果，保留明确错误和可由 Model 投影的图片附件。"""
        # 1. 不在文件工具内根据当前主模型丢弃图片，也不按提示文字猜成功。
        raw = cast(dict[str, Any], json_value(arguments))
        backend = self._backend
        if "_allowed_dir" in raw:
            allowed = raw.pop("_allowed_dir")
            backend = type(self._backend)(self._ctx.require(FILES), allowed_dir=None if allowed is None else Path(allowed))
        try:
            value = (
                await backend.read_raw(**raw) if isinstance(backend, ReadFileTool)
                else await backend.execute(**raw)
            )
        finally:
            if backend is not self._backend:
                await backend.aclose()
        if isinstance(value, str):
            return ToolResultValue("success", (ContentPart("text", value),))
        if isinstance(value, FileError):
            return ToolResultValue("error", (ContentPart("text", value.text),))
        return ToolResultValue("success", (ContentPart("text", value.text), await self._import_image(value)))

    async def _import_image(self, image: FileImage) -> ContentPart:
        """将后端已校验的图片保存为权威附件，临时文件随 scope 释放。"""
        suffixes = {"image/png": ".png", "image/jpeg": ".jpg",
                    "image/webp": ".webp", "image/gif": ".gif"}
        with TemporaryDirectory(prefix="akashic-file-image-") as folder:
            path = Path(folder) / ("image" + suffixes[image.mime_type])
            _ = path.write_bytes(image.data)
            ref = await self._ctx.require(ARTIFACT_IMPORT).import_source(str(path), AttachmentKind.IMAGE)
        return ContentPart("artifact_ref", ref.artifact_id)

    async def query(self, key: str) -> ToolResultValue | None:
        return None


async def register_file(
    ctx: Context, backend_type: type[FileBackend], *, allowed_dir: Path | None,
    directories: WorkingDirectories | None = None,
) -> ToolRef:
    """注册 schema 和配置；实际文件/Bridge 只在已打开工具中访问。"""
    prototype = backend_type(ctx.require(FILES))

    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        return FileSettings.model_validate({
            "allowed_dir": None if allowed_dir is None else str(allowed_dir), **configuration,
            "session_base": allowed_dir is not None and "allowed_dir" not in configuration,
        }).model_dump()

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[FileTool]:
        settings = FileSettings.model_validate(json_value(state))
        backend = backend_type(ctx.require(FILES), allowed_dir=None if settings.allowed_dir is None else Path(settings.allowed_dir))
        try:
            yield FileTool(ctx, backend, directories=directories, settings=settings)
        finally:
            await backend.aclose()

    description = (
        "读取文件。文本带行号，支持 offset/limit 分页；图片保存为附件并交给当前模型查看。"
        if backend_type is ReadFileTool else prototype.description
    )
    return cast(ToolRef, await ctx.require(TOOLS).register(
        ctx,
        name=prototype.name,
        description=description,
        parameters=normalize_tool_parameters(prototype.parameters),
        open=open_tool,
        capture=capture,
        parallel=backend_type in (ReadFileTool, ListDirTool),
    ))

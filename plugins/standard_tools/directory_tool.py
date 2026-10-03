"""The Agent can change only the Session that owns the submitted call."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from agent.plugin_composition.messages import MessageConflict
from agent.plugin_contracts import ContentPart, json_value

from ._tool_boundary import TOOLS, CallSource, ToolResultValue
from .working_directory import WorkingDirectories


class DirectoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, description="已有目录的绝对路径，或相对当前有效目录的路径。")


class PreparedDirectory(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str
    path: str
    expected_version: int | None = Field(ge=0)


class DirectoryTool:
    idempotent = True

    def __init__(self, directories: WorkingDirectories):
        self._directories = directories

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        if source is None:
            return "目录切换需要实际 Session 工具调用"
        try:
            path = DirectoryInput.model_validate(json_value(arguments)).path
            return await self._directories.prepare_switch(source.messages[-1].session_id, path)
        except (ValidationError, ValueError) as error:
            return str(error)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        try:
            prepared = PreparedDirectory.model_validate(json_value(arguments))
            result = await self._directories.switch(key, prepared.model_dump())
        except (ValueError, MessageConflict) as error:
            return ToolResultValue("error", (ContentPart("text", str(error)),))
        return ToolResultValue("success", (ContentPart("text", json.dumps(dict(result), ensure_ascii=False)),))

    async def query(self, key: str) -> ToolResultValue | None:
        result = self._directories.receipt(key)
        return None if result is None else ToolResultValue(
            "success", (ContentPart("text", json.dumps(dict(result), ensure_ascii=False)),),
        )


async def register_directory(ctx: Context, directories: WorkingDirectories) -> None:
    @asynccontextmanager
    async def open_tool(_state: Mapping[str, object]) -> AsyncIterator[DirectoryTool]:
        yield DirectoryTool(directories)

    _ = await ctx.require(TOOLS).register(
        ctx, name="set_working_directory",
        description="切换当前 Session 的工作目录。必须独占本次工具批次；先用 Shell 创建 worktree，再单独切换。不会改变 Project、记忆或其他 Session。",
        parameters=DirectoryInput.model_json_schema(), open=open_tool,
        idempotent=True, exclusive_batch=True,
    )

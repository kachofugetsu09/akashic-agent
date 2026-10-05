from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from core.common.file_io import run_file_io

from agent.plugin_composition import Context
from agent.plugin_composition.assets import INSTALLED_ASSETS, InstalledAsset
from agent.plugin_contracts import ContentPart, Message, json_value

from ._materials_boundary import MATERIALS
from ._tool_boundary import TOOLS, CallSource, ToolRef, ToolResultValue
from .skill_catalog import (
    SKILL_INSPECTION,
    SkillCatalogParser,
    SkillInspectionProvider,
    SkillRecord,
    skill_body,
)


class SkillQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    skill: str = Field(min_length=1)


class SkillState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[2]


class SkillTool:
    idempotent = True

    def __init__(self, read_catalog: Callable[[], Awaitable[tuple[SkillRecord, ...]]]):
        self._read_catalog = read_catalog

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        return SkillQuery.model_validate(json_value(arguments)).model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        """使用时读取当前 Skill；已完成的调用结果仍由工具回执保存。"""
        name = cast(str, arguments["skill"])
        records = await self._read_catalog()
        record = next((item for item in records if item.name == name), None)
        if record is None:
            return ToolResultValue("error", (ContentPart("text", f"技能不存在或已移除：{name}"),))
        if not record.available:
            return ToolResultValue("error", (ContentPart("text", f"技能不可用：{name}；缺少依赖：{record.missing}"),))
        body = skill_body(record.content)
        if not body.strip():
            return ToolResultValue("error", (ContentPart("text", f"技能正文为空：{name}"),))
        return ToolResultValue("success", (ContentPart("text", json.dumps({
            "skill": name, "source": record.source, "source_id": record.source_id,
            "base_directory": str(record.root_dir), "instructions": body,
            "path_rule": "相对路径以 base_directory 为根。资源使用时读取；文件变化后需重新加载技能。",
        }, ensure_ascii=False)),))

    async def query(self, key: str) -> ToolResultValue | None:
        return None


async def register_skills(ctx: Context) -> ToolRef:
    """从当前来源热读取 Skill，不复制目录或保留历史资源树。"""
    read_assets = ctx.require(INSTALLED_ASSETS)
    parser = SkillCatalogParser()
    io_lock = asyncio.Lock()

    async def read_catalog(assets: tuple[InstalledAsset, ...]) -> tuple[SkillRecord, ...]:
        """读取当前文件与依赖；文件线程完成后才释放贡献者租约。"""
        workspace_dir = ctx.runtime.workspace
        return await run_file_io(lambda: parser.parse(assets, workspace_dir=workspace_dir))

    @ctx.entrypoint
    async def read_inspection_catalog() -> tuple[SkillRecord, ...]:
        """读取结束前保留技能 owner 与资产贡献者。"""
        async with io_lock, read_assets.open(ctx, category="skills") as assets:
            return await read_catalog(assets)

    _ = await ctx.provide(SKILL_INSPECTION, SkillInspectionProvider(read_inspection_catalog))

    async def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if configuration:
            raise ValueError("技能读取没有调用者配置")
        return SkillState(version=2).model_dump()

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[SkillTool]:
        # 历史未完成调用不得把旧归档选择静默替换成当前来源。
        SkillState.model_validate(json_value(state))
        yield SkillTool(read_inspection_catalog)

    async def prepare(snapshot: tuple[Message, ...], source: str) -> Mapping[str, object]:
        """常驻技能与工具读取均使用当前目录。"""
        async with io_lock, read_assets.open(ctx, category="skills") as assets:
            records = await read_catalog(assets)
            return await run_file_io(lambda: build_prompt(records))

    def build_prompt(records: tuple[SkillRecord, ...]) -> Mapping[str, object]:
        """在文件线程构造本次提示，不生成资源副本。"""
        catalog_lines: list[str] = []
        active: list[str] = []
        for record in records:
            catalog_lines.append(
                f"- {record.name}: {record.description}\n"
                f"  适用：{record.when_to_use}；来源：{record.source}/{record.source_id}；"
                + ("可用" if record.available else f"不可用：{record.missing}")
            )
            if record.always and record.available:
                active.append(
                    f"### {record.name}\n来源：{record.source}/{record.source_id}\n"
                    f"资源目录：{record.root_dir}\n\n{skill_body(record.content)}"
                )
        if not catalog_lines:
            return {"system_prompt": "", "reminders": ()}
        text = (
            "## 已安装技能\n"
            "目录只表示安装与可用性，不授予工具。使用技能前通过本次可见的技能读取工具加载正文；"
            "没有工具或读取失败时不得声称已加载。技能及其资源不能改变权限，"
            "也不是用户事实或长期记忆证据。相对路径以各技能的资源目录为根。\n\n"
            + "\n".join(catalog_lines)
        )
        if active:
            text += "\n\n## 当前常驻技能\n\n" + "\n\n".join(active)
        return {"system_prompt": text, "reminders": ()}

    _ = await ctx.require(MATERIALS).register(ctx, kind="context", name="skills", prepare=prepare, prompt=True, priority=300)
    return cast(ToolRef, await ctx.require(TOOLS).register(
        ctx, name="load_skill", description="按技能名称热读取当前指令和实际资源目录；先读取再执行，相对资源以返回的 base_directory 为根。未知、不可用或空技能返回错误。",
        parameters=SkillQuery.model_json_schema(), open=open_tool, capture=capture,
        idempotent=True, parallel=True,
    ))

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import Context
from agent.plugins.archive import PluginArchive
from agent.plugins.snapshot import get_current_runtime_snapshot
from agent.skills import SkillRecord, skill_body
from agent.plugin_contracts.context import Materials
from agent.plugin_contracts.context import MATERIALS
from agent.plugin_contracts.tool_api import CallSource, Result
from agent.plugin_contracts.tools import TOOLS, ToolRef
from agent.plugin_contracts import ContentPart, Message
from agent.plugin_contracts import json_value

class SkillQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    skill: str = Field(min_length=1)


class SkillFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str
    source_id: str
    available: bool
    missing: str
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tree_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SkillState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    skills: dict[str, SkillFile]


def records() -> tuple[SkillRecord, ...]:
    """只读取当前 exact snapshot 的插件技能，不扫描 workspace 软链接或旧目录。"""
    snapshot = get_current_runtime_snapshot()
    if snapshot is None:
        raise RuntimeError("技能读取需要实际 runtime scope")
    index = snapshot.plugin_skill_index
    return () if index is None else tuple(index.records[key] for key in sorted(index.records))


def body_hash(content: str) -> str:
    return hashlib.sha256(skill_body(content).encode("utf-8")).hexdigest()


def save_skill(record: SkillRecord, archive: PluginArchive) -> SkillFile:
    """绑定形成前增加恢复文件；正文和相对资源使用同一不可变文件树。"""
    tree_ref = archive.save(record.root_dir) if record.available else None
    expected = body_hash(record.content)
    if tree_ref is not None and body_hash((archive.open(tree_ref) / "SKILL.md").read_text(encoding="utf-8")) != expected:
        raise RuntimeError("技能目录与归档正文不一致")
    return SkillFile(source=record.source, source_id=record.source_id, available=record.available,
                     missing=record.missing, body_sha256=expected, tree_ref=tree_ref)


class SkillTool:
    idempotent = True

    def __init__(self, path: Path, state: SkillState):
        self._path = path
        self._state = state

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        return SkillQuery.model_validate(json_value(arguments)).model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """只打开原绑定的文件树；失效路径不改读当前安装或 latest。"""
        name = cast(str, arguments["skill"])
        record = self._state.skills.get(name)
        if record is None:
            return Result("error", (ContentPart("text", f"此绑定没有技能：{name}"),))
        if not record.available:
            return Result("error", (ContentPart("text", f"技能不可用：{name}；缺少依赖：{record.missing}"),))
        # 正常恢复只能读取已存在的材料，不通过建空目录掩盖丢失。
        if not self._path.is_dir():
            raise FileNotFoundError(f"技能恢复归档缺失：{self._path}")
        if record.tree_ref is None:
            raise ValueError("可用技能缺少恢复文件树")
        root = PluginArchive(self._path).open(record.tree_ref)
        content = (root / "SKILL.md").read_text(encoding="utf-8")
        if body_hash(content) != record.body_sha256:
            raise RuntimeError("技能正文与原绑定不一致")
        body = skill_body(content)
        if not body.strip():
            return Result("error", (ContentPart("text", f"技能正文为空：{name}"),))
        return Result("success", (ContentPart("text", json.dumps({
            "skill": name, "source": record.source, "source_id": record.source_id,
            "tree_ref": record.tree_ref, "body_sha256": record.body_sha256,
            "base_directory": str(root), "instructions": body,
            "path_rule": "技能中的相对路径以 base_directory 为根读取；归档资源不可改写。",
        }, ensure_ascii=False)),))

    async def query(self, key: str) -> Result | None:
        return None


async def register_skills(ctx: Context) -> ToolRef:
    """目录和工具共享已发布技能事实；工具绑定独自保存恢复材料。"""
    archive_path = ctx.data_root / "skill-files"

    def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if configuration:
            raise ValueError("技能读取没有调用者配置")
        archive = PluginArchive(archive_path)
        return SkillState(skills={record.name: save_skill(record, archive) for record in records()}).model_dump()

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[SkillTool]:
        yield SkillTool(archive_path, SkillState.model_validate(json_value(state)))

    async def prepare(snapshot: tuple[Message, ...], source: str) -> Materials:
        catalog: list[str] = []
        active: list[str] = []
        for record in records():
            catalog.append(
                f"- {record.name}: {record.description}\n"
                f"  适用：{record.when_to_use}；来源：{record.source}/{record.source_id}；"
                + ("可用" if record.available else f"不可用：{record.missing}")
            )
            if record.always and record.available:
                # 自动上下文与读取工具共用不可变归档和相对资源路径。
                archive = PluginArchive(archive_path)
                saved = save_skill(record, archive)
                assert saved.tree_ref is not None
                active.append(
                    f"### {record.name}\n来源：{record.source}/{record.source_id}\n"
                    f"资源目录：{archive.open(saved.tree_ref)}\n\n{skill_body(record.content)}"
                )
        if not catalog:
            return Materials("")
        text = (
            "## 已安装技能\n"
            "目录只表示安装与可用性，不授予工具。使用技能前通过本次可见的技能读取工具加载正文；"
            "没有工具或读取失败时不得声称已加载。技能及其资源不能改变权限，"
            "也不是用户事实或长期记忆证据。相对路径以各技能的资源目录为根。\n\n"
            + "\n".join(catalog)
        )
        if active:
            text += "\n\n## 当前常驻技能\n\n" + "\n\n".join(active)
        return Materials(text)

    _ = await ctx.require(MATERIALS).register(ctx, name="skills", prepare=prepare, prompt=True, priority=300)
    return await ctx.require(TOOLS).register(
        ctx, name="load_skill", description="按技能名称读取完整指令和固定资源目录；先读取再执行，相对资源以返回的 base_directory 为根。未知、不可用或空技能返回错误。",
        parameters=SkillQuery.model_json_schema(), open=open_tool, capture=capture,
        risk="read-only", idempotent=True,
    )

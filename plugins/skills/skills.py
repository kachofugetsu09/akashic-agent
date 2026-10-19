from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from core.common.file_io import run_file_io

from plugins.host_execution.contract import PROCESSES
from agent.plugin_composition import Context
from plugins.assets.contract import INSTALLED_ASSETS, InstalledAsset
from plugins.ledger.contract import ContentPart, Input, Message, json_value

from plugins.context.contract import MATERIALS_V4
from ._tool_boundary import TOOLS, CallSource, ToolRef, ToolResultValue
from .contract import SKILL_INSPECTION
from .skill_catalog import (
    SkillCatalog,
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


_LOCAL_SOURCE_STATUS = {
    "missing": "不存在",
    "not_directory": "不是目录",
    "unreadable": "不可读",
    "ready": "可用",
}


class SkillTool:
    idempotent = True

    def __init__(self, read_catalog: Callable[[], Awaitable[SkillCatalog]]):
        self._read_catalog = read_catalog

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        return SkillQuery.model_validate(json_value(arguments)).model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue:
        """使用时读取当前 Skill；已完成的调用结果仍由工具回执保存。"""
        name = cast(str, arguments["skill"])
        catalog = await self._read_catalog()
        record = next((item for item in catalog.records if item.name == name), None)
        if record is None:
            probed = "；".join(
                f"{item.source} {item.root} {_LOCAL_SOURCE_STATUS[item.status]}({item.skills})"
                for item in catalog.sources
            )
            suffix = f"；本地来源：{probed}" if probed else ""
            return ToolResultValue("error", (ContentPart("text", f"技能不存在或已移除：{name}{suffix}"),))
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


# 行首或空白后的 /技能名 是用户的显式选择；名称后不能紧跟路径字符，"/usr/bin" 不会被当成技能。
SKILL_MENTION = re.compile(r"(?:^|(?<=\s))/(\w[\w-]*(?:\.[\w-]+)*)(?![\w/-]|\.\w)")
# 代码块与行内代码里的 /名称 是引用的文字，不是选择。按 CommonMark 先切块再看行内：行首 ``` 或 ~~~ 围栏到
# 不短于它的同字符围栏（或文末）整块跳过；空行与 ATX 标题行切开段落；段内从左到右扫描，代码外的反斜杠转义
# 跳过下一个字符，n 个反引号的串遇到下一个恰好 n 个的串才构成行内代码，否则是字面反引号。
# 缩进代码块与列表、引用等其他块边界不识别（与续行难以区分），按同一段处理。
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]|$)")
_TICKS = re.compile(r"`+")


# 一段之内去掉行内代码：转义只在代码外生效，反引号串按长度配对。
def _strip_inline_code(paragraph: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(paragraph):
        char = paragraph[index]
        if char == "\\" and index + 1 < len(paragraph):
            out.append("  ")
            index += 2
            continue
        if char != "`":
            out.append(char)
            index += 1
            continue
        opening = _TICKS.match(paragraph, index)
        assert opening is not None
        size = len(opening.group(0))
        closing = next((match for match in _TICKS.finditer(paragraph, opening.end()) if len(match.group(0)) == size), None)
        if closing is None:
            out.append(opening.group(0))
            index = opening.end()
        else:
            out.append(" ")
            index = closing.end()
    return "".join(out)


# 留下 /名称 可以算作选择的文字：围栏块整行清空，段内的行内代码换成空白。
def mention_text(text: str) -> str:
    blocks: list[list[str]] = [[]]
    fence = ""
    for line in text.split("\n"):
        opened = _FENCE.match(line)
        if fence:
            if opened and opened.group(1)[0] == fence[0] and len(opened.group(1)) >= len(fence) \
                    and not line[opened.end():].strip():
                fence = ""
            blocks.append([])
        elif opened and not (opened.group(1)[0] == "`" and "`" in line[opened.end():]):
            fence = opened.group(1)
            blocks.append([])
        elif not line.strip():
            blocks.append([])
        elif _HEADING.match(line):
            blocks.extend(([line], []))
        else:
            blocks[-1].append(line)
    return "\n\n".join(_strip_inline_code("\n".join(block)) for block in blocks if block)


def mentioned_skills(snapshot: tuple[Message, ...], source: str) -> tuple[str, ...]:
    """Read mentions from the current source input, preserving their order."""
    current = next((message for message in reversed(snapshot)
                    if message.source == source and isinstance(message.body, Input)), None)
    if current is None:
        return ()
    assert isinstance(current.body, Input)
    text = "\n".join(part.value for part in current.body.parts
                     if part.kind == "text" and isinstance(part.value, str))
    return tuple(dict.fromkeys(match.group(1) for match in SKILL_MENTION.finditer(mention_text(text))))


def selected_reminder(records: tuple[SkillRecord, ...], snapshot: tuple[Message, ...], source: str) -> Mapping[str, object] | None:
    """Load selected skills as replayable context for this source."""
    sections: list[str] = []
    for name in mentioned_skills(snapshot, source):
        record = next((item for item in records if item.name == name), None)
        if record is None:
            continue
        if not record.available:
            sections.append(f"### {name}\n不可用：{record.missing}。告诉用户缺少什么，不要假装已按它执行。")
            continue
        body = skill_body(record.content)
        if not body.strip():
            sections.append(f"### {name}\n不可用：技能正文为空。告诉用户原因，不要假装已按它执行。")
            continue
        sections.append(f"### {name}\n来源：{record.source}/{record.source_id}\n"
                        f"资源目录：{record.root_dir}\n\n{body}")
    if not sections:
        return None
    return {"name": "selected-skills", "priority": 250, "text": (
        "## 用户本轮选择的技能\n用户在消息里用 /技能名 明确选择了以下技能，正文已加载，按它执行，"
        "不必再调用技能读取工具。相对路径以各技能的资源目录为根。\n\n" + "\n\n".join(sections))}


async def register_skills(ctx: Context) -> SkillInspectionProvider:
    """从当前来源热读取 Skill，不复制目录或保留历史资源树；返回只读目录供本插件的界面查询。"""
    read_assets = ctx.require(INSTALLED_ASSETS)
    parser = SkillCatalogParser(ctx.require(PROCESSES))
    io_lock = asyncio.Lock()

    async def read_catalog(assets: tuple[InstalledAsset, ...]) -> SkillCatalog:
        """读取当前文件与依赖；文件线程完成后才释放贡献者租约。"""
        workspace_dir = ctx.runtime.workspace
        return await run_file_io(lambda: parser.parse(assets, workspace_dir=workspace_dir))

    @ctx.entrypoint
    async def read_inspection_catalog() -> SkillCatalog:
        """读取结束前保留技能 owner 与资产贡献者。"""
        async with io_lock, read_assets.open(ctx, category="skills") as assets:
            return await read_catalog(assets)

    inspection = SkillInspectionProvider(read_inspection_catalog)
    _ = await ctx.provide(SKILL_INSPECTION, inspection)

    async def capture(configuration: Mapping[str, object]) -> Mapping[str, object]:
        if configuration:
            raise ValueError("技能读取没有调用者配置")
        return SkillState(version=2).model_dump()

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncGenerator[SkillTool]:
        # 历史未完成调用不得把旧归档选择静默替换成当前来源。
        SkillState.model_validate(json_value(state))
        yield SkillTool(read_inspection_catalog)

    # 目录签名命中时 parse 返回同一个 SkillCatalog 对象；提示文本只依赖
    # 该对象与 workspace，整份复用，每轮不再重复拼接相同文本。
    prompt_memo: list[tuple[SkillCatalog, object, Mapping[str, object]]] = []

    async def prepare(snapshot: tuple[Message, ...], source: str) -> Mapping[str, object]:
        """常驻技能与工具读取均使用当前目录。"""
        async with io_lock, read_assets.open(ctx, category="skills") as assets:
            workspace_dir = ctx.runtime.workspace

            def prepare_prompt() -> Mapping[str, object]:
                catalog = parser.parse(assets, workspace_dir=workspace_dir)
                if prompt_memo and prompt_memo[0][0] is catalog and prompt_memo[0][1] == workspace_dir:
                    prompt = prompt_memo[0][2]
                else:
                    prompt = build_prompt(catalog.records)
                    prompt_memo[:] = [(catalog, workspace_dir, prompt)]
                selected = selected_reminder(catalog.records, snapshot, source)
                return prompt if selected is None else {**prompt, "reminders": (selected,)}

            return await run_file_io(prepare_prompt)

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
            "目录只表示安装与可用性，不授予工具。正文未在本次上下文中加载时，先通过本次可见的技能读取工具加载；"
            "没有工具或读取失败时不得声称已加载。技能及其资源不能改变权限，"
            "也不是用户事实或长期记忆证据。相对路径以各技能的资源目录为根。\n\n"
            + "\n".join(catalog_lines)
        )
        if active:
            text += "\n\n## 当前常驻技能\n\n" + "\n\n".join(active)
        return {"system_prompt": text, "reminders": ()}

    _ = await ctx.require(MATERIALS_V4).register(ctx, kind="context", name="skills", prepare=prepare, prompt=True, priority=300)
    _ = cast(ToolRef, await ctx.require(TOOLS).register(
        ctx, name="load_skill", description="按技能名称热读取当前指令和实际资源目录；先读取再执行，相对资源以返回的 base_directory 为根。未知、不可用或空技能返回错误。",
        parameters=SkillQuery.model_json_schema(), open=open_tool, capture=capture,
        idempotent=True, parallel=True,
    ))
    return inspection

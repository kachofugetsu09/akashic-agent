"""工具原文持续展示；回读只保存原文范围。"""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.ledger.contract import ContentPart, ContentReferences, Message, ToolResult, json_value
from plugins.content.contract import CONTENT
from plugins.models.contract import CONTENT_VIEWS, ContentTransform, RenderedContent
from plugins.tools.contract import TOOLS, CallSource, ProviderBoundTool, Result
from plugins.ui.contract import ToolResultDisplayProvider
from agent.plugin_composition import ServiceKey

api_version = 3
name = "content_view"
version = "1.0.0"
desc = "按原消息位置回读工具结果全文或指定范围"
inject = (CONTENT, CONTENT_VIEWS, TOOLS)

READ_KIND = "content_view.read"
READ_TOOL = "read_content"
READ_DISPLAY = ServiceKey[ToolResultDisplayProvider]("message.result_display:content_view.read")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # 已有安装可能保存此键；只兼容读取，不再控制任何投影行为。
    fold_after_chars: int = Field(default=8192, ge=1024, strict=True, deprecated=True,
                                  description="已停用；工具结果始终完整展示")


class ReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message_id: str = Field(min_length=1)
    part_index: int = Field(ge=0)
    start: int = Field(default=0, ge=0)
    end: int | None = Field(default=None, ge=0)


class ReadReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message_id: str = Field(min_length=1)
    part_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)


def check_read(part: ContentPart) -> ContentReferences:
    ref = ReadReference.model_validate(json_value(part.value))
    if ref.end < ref.start:
        raise ValueError("回读范围结束位置不能小于开始位置")
    return ContentReferences()


def text_part(messages: Mapping[str, Message], message_id: str, index: int) -> str:
    """只定位当前 Session 快照中的原始工具文本，不解析任意路径或外部 URI。"""
    message = messages.get(message_id)
    if message is None or not isinstance(message.body, ToolResult):
        raise ValueError("引用不是当前会话中可读的工具结果")
    if index >= len(message.body.parts):
        raise ValueError("引用的内容位置不存在")
    part = message.body.parts[index]
    if part.kind != "text":
        raise ValueError("引用不是原始文本内容")
    return cast(str, part.value)


def read_text(messages: Mapping[str, Message], ref: ReadReference) -> str:
    text = text_part(messages, ref.message_id, ref.part_index)
    if not 0 <= ref.start <= ref.end <= len(text):
        raise ValueError("回读范围超出原文；未返回部分结果")
    return text[ref.start:ref.end]


async def display_read(part: ContentPart, read_message: Callable[[str], Awaitable[Message | None]]) -> object:
    """只读原文范围；缺失引用明确展示原因，不重跑工具或改写历史。"""
    ref = ReadReference.model_validate(json_value(part.value))
    target = await read_message(ref.message_id)
    if target is None:
        return {"error": "引用的原始内容不可用", "reference": ref.model_dump()}
    return read_text({target.message_id: target}, ref)


class ReadContent:
    idempotent = True

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None
                      ) -> Mapping[str, object] | str:
        """调用前固定原始位置与范围；参数错误返回明确拒绝，不扩大会话权限。"""
        if source is None:
            return "回读需要当前调用的会话前缀"
        try:
            requested = ReadInput.model_validate(json_value(arguments))
        except ValidationError as error:
            return str(error)
        messages = {message.message_id: message for message in source.messages}
        target = messages.get(requested.message_id)
        if target is None or not isinstance(target.body, ToolResult):
            return "引用不是当前会话中可读的工具结果"
        if requested.part_index >= len(target.body.parts):
            return "引用的内容位置不存在"
        part = target.body.parts[requested.part_index]
        # 1. 允许再次读取先前回读，但直接落回原文，不建立引用链。
        if part.kind == READ_KIND:
            original = ReadReference.model_validate(json_value(part.value))
            text = read_text(messages, original)
        elif part.kind == "text":
            text = cast(str, part.value)
            original = ReadReference(message_id=target.message_id, part_index=requested.part_index,
                                     start=0, end=len(text))
        else:
            return "该内容不是可回读的文本"
        end = len(text) if requested.end is None else requested.end
        if not 0 <= requested.start <= end <= len(text):
            return "范围必须满足 0 <= start <= end <= 原文字符数；不会静默截断"
        # 2. 持久准备只含原文坐标；invoke 无文件、网络或消息写入副作用。
        return ReadReference(message_id=original.message_id, part_index=original.part_index,
                             start=original.start + requested.start,
                             end=original.start + end).model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        ref = ReadReference.model_validate(json_value(arguments))
        return Result("success", (ContentPart(READ_KIND, ref.model_dump()),))

    async def query(self, key: str) -> Result | None:
        # Tools 已保存准备参数；只读幂等调用可用原参数重做，无独立效果账本。
        return None


def prepare_view(messages: tuple[Message, ...], source: str, tools: frozenset[str],
                 seen: frozenset[tuple[str, int]]) -> ContentTransform:
    """只展开回读引用；工具文本始终由基础投影完整展示。"""
    by_id = {message.message_id: message for message in messages}

    def render(message: Message, index: int) -> RenderedContent | None:
        if not isinstance(message.body, ToolResult):
            return None
        part = message.body.parts[index]
        if part.kind == READ_KIND:
            ref = ReadReference.model_validate(json_value(part.value))
            text = read_text(by_id, ref)
            return RenderedContent(({"type": "text", "text": text},), complete=True)
        return None

    return render


async def apply(ctx: Context) -> None:
    """通过普通内容声明、投影注册和工具目录接入；不申请任何写入或存储权限。"""
    Config.model_validate(ctx.config)
    await ctx.provide(READ_DISPLAY, display_read)
    await ctx.require(CONTENT).register(ctx, {"name": "read_content", "content": {READ_KIND: check_read}})
    catalog = ctx.require(TOOLS)
    await catalog.declare_group(ctx, always_on=True, description=desc)

    @asynccontextmanager
    async def open_tool(state: Mapping[str, object]) -> AsyncIterator[ProviderBoundTool]:
        yield ReadContent()

    await catalog.register(
        ctx, name=READ_TOOL,
        description=("回读当前会话的工具结果。message_id/part_index 指向已有工具结果；"
                     "省略 start/end 读取全文，或指定字符区间 [start,end)。"
                     "超出范围明确拒绝，不静默截断。不要重新执行原工具来找回结果。"),
        parameters=ReadInput.model_json_schema(), open=open_tool, idempotent=True, parallel=True,
    )
    await ctx.require(CONTENT_VIEWS).register(
        ctx, name="tool_results", prepare=prepare_view,
        # 只展开 READ_KIND 回读块；其他消息恒返回 None，投影可缓存其分段。
        dynamic_kinds=frozenset({READ_KIND}),
    )

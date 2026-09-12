from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

from agent.plugin_composition import Context, RuntimeScope, ServiceKey
from agent.plugins.snapshot import get_current_runtime_lease
from agent.plugin_composition.models import (
    BoundChatModel,
    ContextLengthError,
    EmptyResponseError,
    LLMResponse,
    StreamCallback,
)
from plugins.context.api import ContextOverflow, Materials, SummaryReducer
from plugins.tools.menu import InvalidToolCall
from agent.plugin_composition.messages import MessageReader, MessageWriter
from agent.plugin_contracts import CallRef, Control, Message, Output, Part, ContentPart, ToolCall, ToolResult

if TYPE_CHECKING:
    from plugins.content.plugin import ContentView
    from plugins.context.plugin import ContextBuilder
    from plugins.models.projection import MessageProjection
    from plugins.tools.menu import ToolMenu

api_version = 3
name = "react"
version = "1.0.0"
desc = "组合上下文、模型、内容与工具，不拥有会话或外部效果状态"
inject = ()


Preview = Callable[[str], AbstractContextManager[StreamCallback]]


class StepLimit(RuntimeError):
    """本次程序达到明确的模型请求上限，保留日志供来源继续控制。"""


def _pending_calls(messages: Sequence[Message], source: str) -> tuple[CallRef, ...]:
    """只恢复本来源尚未关闭的请求；abandon 的晚到结果不唤醒新决策。"""
    boundary = -1
    calls: dict[CallRef, int] = {}
    results: dict[CallRef, ToolResult] = {}
    for message in messages:
        if message.source != source:
            continue
        body = message.body
        if isinstance(body, Output):
            if body.finish != "continue":
                boundary = message.seq
            calls.update(
                (CallRef(message.message_id, index), message.seq)
                for index, part in enumerate(body.parts) if isinstance(part, ToolCall)
            )
        elif isinstance(body, Control) and body.action == "abandon":
            boundary = max(boundary, body.through_seq)
        elif isinstance(body, ToolResult):
            results[body.call_ref] = body
    pending: list[CallRef] = []
    for ref, seq in calls.items():
        if seq <= boundary:
            continue
        result = results.get(ref)
        if result is None:
            pending.append(ref)
    return tuple(pending)


def _steps(messages: Sequence[Message], source: str) -> int:
    """完成步数来自本来源未闭段的模型 Output，中断或进程重启不会重置。"""
    outputs: list[Message] = []
    for message in messages:
        if message.source != source:
            continue
        body = message.body
        if isinstance(body, Output):
            if body.finish != "continue":
                outputs.clear()
            elif any(isinstance(part, ContentPart) and part.kind == "model.facts" for part in body.parts):
                outputs.append(message)
        elif isinstance(body, Control) and body.action == "abandon":
            outputs = [item for item in outputs if item.seq > body.through_seq]
    return len(outputs)


def _terminal_result(messages: Sequence[Message], source: str, tools: ToolMenu,
                     names: frozenset[str]) -> bool:
    """终结规则只读取本来源未闭段的成功回执，不把调用意图当作效果。"""
    boundary = -1
    calls: dict[CallRef, int] = {}
    succeeded: set[CallRef] = set()
    for message in messages:
        if message.source != source:
            continue
        body = message.body
        if isinstance(body, Output):
            if body.finish != "continue":
                boundary = message.seq
            else:
                calls.update((CallRef(message.message_id, index), message.seq)
                             for index, part in enumerate(body.parts)
                             if isinstance(part, ToolCall) and tools.name(part.binding_id) in names)
        elif isinstance(body, Control) and body.action == "abandon":
            boundary = max(boundary, body.through_seq)
        elif isinstance(body, ToolResult) and body.outcome == "success":
            succeeded.add(body.call_ref)
    return any(seq > boundary and ref in succeeded for ref, seq in calls.items())


async def _settle(tools: ToolMenu, call: CallRef) -> None:
    """普通取消等待原调用；明确放弃由 Tools 提交终态并释放等待者。"""
    lease = get_current_runtime_lease()
    scope = None if lease is None else RuntimeScope(lease.fork())

    async def execute():
        if scope is None:
            return await tools.execute(call)
        async with scope:
            return await tools.execute(call)

    operation = execute()
    try:
        try:
            work = asyncio.create_task(operation)
        except BaseException:
            operation.close()
            raise
        try:
            _ = await asyncio.shield(work)
        except asyncio.CancelledError as cancellation:
            while not work.done():
                try:
                    _ = await asyncio.shield(work)
                except asyncio.CancelledError:
                    continue
                except Exception as failure:
                    raise cancellation from failure
            if not work.cancelled():
                failure = work.exception()
                if failure is not None:
                    raise cancellation from failure
            raise
    finally:
        if scope is not None:
            await scope.close()


@asynccontextmanager
async def _complete(
    snapshot: tuple[Message, ...], prepared: Materials, *, source: str,
    context: ContextBuilder, model: BoundChatModel, projection: MessageProjection,
    tools: ToolMenu, max_output_tokens: int, reduce: SummaryReducer | None,
    preview: Preview | None,
) -> AsyncGenerator[tuple[LLMResponse, Materials, str]]:
    """缩减只更新已取得材料中的摘要；provider 容量拒绝最多重试一次。"""
    # 1. 本地容量与软水位先交给同一摘要 owner，其他材料不重新获取。
    try:
        request = context.build(
            snapshot, materials=prepared, model=projection,
            tools=tools.schemas, max_output_tokens=max_output_tokens,
        )
    except ContextOverflow as overflow:
        if reduce is None:
            raise
        summary = await reduce(snapshot, prepared, overflow.request, model, projection, source=source, force=True)
        if summary == prepared.summary:
            raise
        prepared = replace(prepared, summary=summary)
        request = context.build(snapshot, materials=prepared, model=projection,
                                tools=tools.schemas, max_output_tokens=max_output_tokens)
    else:
        if reduce is not None:
            summary = await reduce(snapshot, prepared, request, model, projection, source=source, force=False)
            if summary != prepared.summary:
                prepared = replace(prepared, summary=summary)
                request = context.build(snapshot, materials=prepared, model=projection,
                                        tools=tools.schemas, max_output_tokens=max_output_tokens)
    # 2. 每次 provider 调用预分配消息 ID；重试先撤掉旧草稿，再开始下一次请求。
    with ExitStack() as previews:
        def begin() -> tuple[str, StreamCallback | None]:
            message_id = uuid4().hex
            callback = None if preview is None else previews.enter_context(preview(message_id))
            return message_id, callback

        message_id, callback = begin()
        try:
            response = await model.complete(replace(request, on_delta=callback))
        except ContextLengthError:
            previews.close()
            if reduce is None:
                raise
            summary = await reduce(snapshot, prepared, request, model, projection, source=source, force=True)
            if summary == prepared.summary:
                raise
            prepared = replace(prepared, summary=summary)
            request = context.build(snapshot, materials=prepared, model=projection,
                                    tools=tools.schemas, max_output_tokens=max_output_tokens)
            message_id, callback = begin()
            response = await model.complete(replace(request, on_delta=callback))
        # 3. 草稿持续到调用者完成解码与 CAS；异常和取消也会释放预览。
        yield response, prepared, message_id


async def react(
    reader: MessageReader,
    writer: MessageWriter,
    *,
    model: BoundChatModel,
    context: ContextBuilder,
    projection: MessageProjection,
    materials: Callable[[tuple[Message, ...]], Awaitable[Materials]],
    content: ContentView,
    tools: ToolMenu,
    max_output_tokens: int,
    max_steps: int,
    reduce: SummaryReducer | None = None,
    preview: Preview | None = None,
    terminal_tools: frozenset[str] = frozenset(),
) -> Message:
    """先结算已提交调用，再读日志推理并逐条提交；没有 Turn、Attempt 或历史副本。"""
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("模型请求上限必须是非负整数")
    if reader.session_id != writer.session_id:
        raise ValueError("ReAct reader 与 writer 必须属于同一 Session")
    while True:
        # 1. 串行策略停止补发并排空已开始调用；换算法无需改 Tool effect owner。
        for call in _pending_calls(reader.snapshot(), writer.source):
            await _settle(tools, call)
        snapshot = reader.snapshot()
        if terminal_tools and _terminal_result(snapshot, writer.source, tools, terminal_tools):
            return writer.append(uuid4().hex, Output((), "quiet"),
                                 expected_source_head=reader.head(source=writer.source))
        if max_steps > 0 and _steps(snapshot, writer.source) >= max_steps:
            raise StepLimit(f"本来源未完成工作已达到 {max_steps} 个模型输出")
        head = max((m.seq for m in snapshot if m.source == writer.source), default=-1)
        # 2. 取得材料与组装请求分开，Context 不获得模型调用或检索权。
        prepared = await materials(snapshot)
        async with _complete(
            snapshot, prepared, source=writer.source, context=context, model=model,
            projection=projection, tools=tools, max_output_tokens=max_output_tokens, reduce=reduce, preview=preview,
        ) as (response, prepared, message_id):
            decoded, metadata = await content.decode(response.content or "", prepared.references)
            parts: list[Part] = list(decoded)
            indices: list[int] = []
            actual_calls: list[ToolCall | ContentPart] = []
            for call in response.tool_calls:
                indices.append(len(parts))
                try:
                    binding_id, arguments = tools.decode(call)
                except InvalidToolCall as error:
                    actual = ContentPart("model.tool_rejection", {
                        "name": call.name, "arguments": call.arguments, "error": str(error),
                    })
                else:
                    actual = ToolCall(binding_id, arguments)
                actual_calls.append(actual)
                parts.append(actual)
            if not parts:
                raise EmptyResponseError("模型没有产生内容或工具调用；空响应不是 quiet")
            parts.append(projection.facts(
                response,
                indices,
                reminder=context.reminder_content(prepared),
                actual_calls=actual_calls,
            ))
            if prepared.summary is not None:
                parts.append(ContentPart("context.summary", {"reference": prepared.summary.reference}))
            # 3. 内容完成后按来源 CAS 提交；失败的草稿绝不触发工具。
            message = writer.append(
                message_id, Output(tuple(parts), "continue" if indices else "complete"),
                expected_source_head=head, metadata=metadata,
            )
            if not indices:
                return message


REACT = ServiceKey[Callable[..., Awaitable[Message]]]("react.v1")


async def apply(ctx: Context, config: object) -> None:
    _ = await ctx.provide(REACT, react)

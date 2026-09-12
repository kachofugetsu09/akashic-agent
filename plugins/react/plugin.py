from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence, Mapping
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from functools import partial
from dataclasses import replace
from typing import Protocol, Any, cast
from uuid import uuid4

from agent.plugin_composition import Context, RuntimeScope, ServiceKey
from agent.plugin_composition.models import (
    BoundChatModel,
    ContextLengthError,
    EmptyResponseError,
    LLMResponse,
    ModelRequest,
    ModelError,
    StreamCallback,
)
from agent.plugin_composition.messages import MessageReader, MessageWriter
from agent.plugin_contracts import CallRef, Control, Message, Output, Part, ContentPart, ToolCall, ToolResult

Materials = Mapping[str, object]


class ContentView(Protocol):
    async def decode(self, text: str, references: tuple[Mapping[str, object], ...] = ()) -> tuple[tuple[ContentPart, ...], Mapping[str, object]]: ...


class ContextBuilder(Protocol):
    def build_attempt(self, snapshot: Sequence[Message], *, materials: Materials,
                      model: MessageProjection, tools: Sequence[Mapping[str, Any]] = (),
                      max_output_tokens: int, window_start: str | None = None) -> tuple[ModelRequest, str | None]: ...
    def reminder_content(self, materials: Materials) -> str | None: ...


class DecodedCall(Protocol):
    @property
    def binding_id(self) -> str | None: ...
    @property
    def arguments(self) -> Mapping[str, object]: ...
    @property
    def rejection(self) -> Mapping[str, object] | None: ...


class ToolMenu(Protocol):
    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]: ...
    def name(self, binding_id: str) -> str: ...
    def decode(self, call: Any) -> DecodedCall: ...
    async def execute(self, call: CallRef) -> object: ...


class SummaryReducer(Protocol):
    async def __call__(self, snapshot: tuple[Message, ...], materials: Materials,
                       request: ModelRequest, model: BoundChatModel, projection: MessageProjection,
                       *, source: str, force: bool) -> Mapping[str, object] | None: ...


class MessageProjection(Protocol):
    @property
    def context_window(self) -> int | None: ...
    @property
    def max_tool_schemas(self) -> int | None: ...
    def estimate(self, request: ModelRequest) -> int: ...
    def render(self, messages: tuple[Message, ...], *, after_seq: int,
               summary_reference: str | None = None, fresh: bool = False) -> ModelRequest: ...
    def facts(self, response: LLMResponse, call_indices: Sequence[int], *,
              reminder: str | None = None,
              actual_calls: Sequence[ToolCall | ContentPart] | None = None) -> ContentPart: ...


api_version = 3
name = "react"
version = "1.0.0"
desc = "组合上下文、模型、内容与工具，不拥有会话或外部效果状态"
inject = ()


Preview = Callable[[str], AbstractContextManager[StreamCallback]]


class StepLimit(ModelError):
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


async def _settle(tools: ToolMenu, call: CallRef, capture_scope: Callable[[], RuntimeScope] | None = None) -> None:
    """普通取消等待原调用；明确放弃由 Tools 提交终态并释放等待者。"""
    scope = None if capture_scope is None else capture_scope()

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
    def build() -> tuple[ModelRequest, str | None]:
        return context.build_attempt(snapshot, materials=prepared, model=projection,
                                     tools=tools.schemas, max_output_tokens=max_output_tokens)

    request, rejection = build()
    if rejection is not None and reduce is None:
        raise ContextLengthError(rejection)
    if reduce is not None:
        summary = await reduce(snapshot, prepared, request, model, projection,
                               source=source, force=rejection is not None)
        if summary is not None and summary != prepared.get("summary"):
            prepared = {**prepared, "summary": summary}
            request, rejection = build()
        if rejection is not None:
            raise ContextLengthError(rejection)
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
            if summary is None or summary == prepared.get("summary"):
                raise
            prepared = {**prepared, "summary": summary}
            request, rejection = build()
            if rejection is not None:
                raise ContextLengthError(rejection)
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
    capture_scope: Callable[[], RuntimeScope] | None = None,
) -> Message:
    """先结算已提交调用，再读日志推理并逐条提交；没有 Turn、Attempt 或历史副本。"""
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("模型请求上限必须是非负整数")
    if reader.session_id != writer.session_id:
        raise ValueError("ReAct reader 与 writer 必须属于同一 Session")
    while True:
        # 1. 串行策略停止补发并排空已开始调用；换算法无需改 Tool effect owner。
        for call in _pending_calls(reader.snapshot(), writer.source):
            await _settle(tools, call, capture_scope)
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
            decoded, metadata = await content.decode(response.content or "", cast(tuple[Mapping[str, object], ...], prepared.get("references", ())))
            parts: list[Part] = list(decoded)
            indices: list[int] = []
            actual_calls: list[ToolCall | ContentPart] = []
            for call in response.tool_calls:
                indices.append(len(parts))
                decoded_call = tools.decode(call)
                if decoded_call.rejection is not None:
                    actual = ContentPart("model.tool_rejection", decoded_call.rejection)
                else:
                    assert decoded_call.binding_id is not None
                    actual = ToolCall(decoded_call.binding_id, decoded_call.arguments)
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
            summary = cast(Mapping[str, object] | None, prepared.get("summary"))
            if summary is not None:
                parts.append(ContentPart("context.summary", {"reference": summary["reference"]}))
            # 3. 内容完成后按来源 CAS 提交；失败的草稿绝不触发工具。
            message = writer.append(
                message_id, Output(tuple(parts), "continue" if indices else "complete"),
                expected_source_head=head, metadata=metadata,
            )
            if not indices:
                return message


REACT = ServiceKey[Callable[..., Awaitable[Message]]]("react.v2")


async def apply(ctx: Context, config: object) -> None:
    _ = await ctx.provide(REACT, partial(react, capture_scope=ctx.capture_runtime_scope))

from __future__ import annotations

import asyncio
import logging
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
    ModelContinuation,
    ModelRequest,
    ModelError,
    ModelUnavailableError,
    StreamCallback,
)
from agent.plugin_composition.messages import MessageConflict, MessageReader, MessageWriter, OwnerStore, OwnerTransaction
from agent.plugin_contracts import CallRef, Control, Input, Message, Output, Part, ContentPart, ToolCall, ToolResult

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
    async def settle_abandoned(self, call: CallRef) -> object: ...


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


logger = logging.getLogger(__name__)

api_version = 3
name = "react"
version = "1.0.0"
desc = "组合上下文、模型、内容与工具，不拥有会话或外部效果状态"
inject = ()


Preview = Callable[[str], AbstractContextManager[StreamCallback]]


class StepLimit(ModelError):
    """本次程序达到明确的模型请求上限，保留日志供来源继续控制。"""


class _Superseded(Exception):
    """提交前提检查发现同来源新事实；被替代的旧草稿不写 failure。"""


def _open_calls(messages: Sequence[Message], source: str) -> tuple[tuple[CallRef, ...], tuple[CallRef, ...]]:
    """按持久边界拆分未回执调用：未闭段继续排空，abandon 区幂等结算。"""
    boundary = -1
    abandoned_upto = -1
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
            abandoned_upto = max(abandoned_upto, body.through_seq)
        elif isinstance(body, ToolResult):
            results[body.call_ref] = body
    pending: list[CallRef] = []
    abandoned: list[CallRef] = []
    for ref, seq in calls.items():
        if ref in results:
            continue
        if seq <= abandoned_upto:
            abandoned.append(ref)
        elif seq > boundary:
            pending.append(ref)
    return tuple(pending), tuple(abandoned)


def _pending_calls(messages: Sequence[Message], source: str) -> tuple[CallRef, ...]:
    """只恢复本来源尚未关闭的请求；abandon 的晚到结果不唤醒新决策。"""
    return _open_calls(messages, source)[0]


def _related_results(messages: Sequence[Message], source: str) -> frozenset[CallRef]:
    """冻结前缀内缺回执的调用：其结算结果属于本代请求的读集。"""
    pending, abandoned = _open_calls(messages, source)
    return frozenset((*pending, *abandoned))


def _competing(message: Message, source: str, related: frozenset[CallRef]) -> bool:
    """同来源 Input/Control/任何 Output 或读集内 ToolResult 使旧草稿失效。"""
    if message.source != source:
        return False
    body = message.body
    if isinstance(body, (Input, Control)):
        return True
    if isinstance(body, Output):
        # 任何同来源 Output 都占据输出前驱位置；本代草稿的前提已被取代。
        return True
    return isinstance(body, ToolResult) and body.call_ref in related


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _encode_request(request: ModelRequest) -> Mapping[str, object]:
    """生成准备只冻结模型可见字段；on_delta/request_key 是执行细节。"""
    continuation = request.continuation
    return {
        "messages": _plain_json(request.messages),
        "tools": _plain_json(request.tools),
        "max_output_tokens": request.max_output_tokens,
        "system_prompt": request.system_prompt,
        "tool_choice": _plain_json(request.tool_choice),
        "prompt_cache_key": request.prompt_cache_key,
        "disable_reasoning": request.disable_reasoning,
        "continuation": (
            None
            if continuation is None
            else {
                "binding_id": continuation.binding_id,
                "payload": _plain_json(continuation.payload),
            }
        ),
    }


def _decode_request(value: object) -> ModelRequest:
    """恢复冻结请求；损坏记录在边界明确失败。"""
    if not isinstance(value, Mapping):
        raise ValueError("生成准备中的模型请求记录损坏")
    continuation = value.get("continuation")
    return ModelRequest(
        messages=tuple(cast(Sequence[Mapping[str, Any]], value["messages"])),
        tools=tuple(cast(Sequence[Mapping[str, Any]], value.get("tools") or ())),
        max_output_tokens=cast(int, value.get("max_output_tokens") or 0),
        system_prompt=cast(str, value.get("system_prompt") or ""),
        tool_choice=cast(Any, value.get("tool_choice", "auto")),
        prompt_cache_key=cast(str | None, value.get("prompt_cache_key")),
        disable_reasoning=bool(value.get("disable_reasoning")),
        continuation=(
            None
            if continuation is None
            else ModelContinuation(
                cast(str, cast(Mapping[str, object], continuation)["binding_id"]),
                cast(Mapping[str, Any], cast(Mapping[str, object], continuation)["payload"]),
            )
        ),
    )


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
    claim: Callable[[int], tuple[str, str | None]] | None = None,
    fallback_key: str | None = None,
    freeze: Callable[[ModelRequest, Materials], tuple[ModelRequest, Materials]] | None = None,
    request_override: ModelRequest | None = None,
) -> AsyncGenerator[tuple[LLMResponse, Materials, str]]:
    """缩减只更新已取得材料中的摘要；provider 容量拒绝最多重试一次。"""
    # 1. 本地容量与软水位先交给同一摘要 owner，其他材料不重新获取。
    def build() -> tuple[ModelRequest, str | None]:
        return context.build_attempt(snapshot, materials=prepared, model=projection,
                                     tools=tools.schemas, max_output_tokens=max_output_tokens)

    if request_override is None:
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
    else:
        request = request_override
    if freeze is not None:
        # 生成准备把首个真实请求与材料一并冻结；恢复后不再重建或漂移。
        request, prepared = freeze(request, prepared)
    # 2. 每次 provider 调用使用生成准备中已耐久固定的 Output ID 与请求 key。
    with ExitStack() as previews:
        def begin(attempt: int) -> tuple[str, str | None, StreamCallback | None]:
            if claim is None:
                # 无准备记录时仍以持久前提界定同一请求，避免跨代重放相同字节。
                message_id = uuid4().hex
                request_key = (
                    None if fallback_key is None else f"{fallback_key}:{attempt}"
                )
            else:
                message_id, request_key = claim(attempt)
            callback = None if preview is None else previews.enter_context(preview(message_id))
            return message_id, request_key, callback

        attempt = 0
        message_id, request_key, callback = begin(attempt)
        try:
            response = await model.complete(replace(request, on_delta=callback, request_key=request_key))
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
            attempt += 1
            message_id, request_key, callback = begin(attempt)
            response = await model.complete(replace(request, on_delta=callback, request_key=request_key))
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
    state: OwnerStore | None = None,
) -> Message:
    """先结算已提交调用，再读日志推理并逐条提交；没有 Turn、Attempt 或历史副本。"""
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("模型请求上限必须是非负整数")
    if reader.session_id != writer.session_id:
        raise ValueError("ReAct reader 与 writer 必须属于同一 Session")
    while True:
        # 1. 串行策略停止补发并排空已开始调用；换算法无需改 Tool effect owner。
        pending, abandoned = _open_calls(reader.snapshot(), writer.source)
        for call in abandoned:
            # 已放弃调用的结算故障必须先阻断本来源：缺回执的调用不能带着未知效果进入新请求。
            await tools.settle_abandoned(call)
        for call in pending:
            await _settle(tools, call, capture_scope)
        snapshot = reader.snapshot()
        head = max((m.seq for m in snapshot if m.source == writer.source), default=-1)
        # 本代准备的固定身份：最近一条同来源 Input/Control（持久边界）。
        # 无关事实抬高 head 不产生新准备、不重复付费。
        boundary_id = next(
            (
                message.message_id
                for message in reversed(snapshot)
                if message.source == writer.source
                and isinstance(message.body, (Input, Control))
            ),
            "initial",
        )
        frozen = snapshot

        def commit(message_id: str, body: Output, metadata: Mapping[str, object] | None = None) -> Message:
            """检查与追加同事务；竞争 Output、新边界或读集内结果都取代旧草稿。"""
            related = _related_results(frozen, writer.source)
            if state is None:
                current_head = head
                for _ in range(4):
                    try:
                        return writer.append(
                            message_id, body,
                            expected_source_head=current_head, metadata=metadata,
                        )
                    except MessageConflict:
                        newer = [
                            m for m in reader.snapshot(after_seq=current_head)
                            if m.source == writer.source
                        ]
                        if any(_competing(m, writer.source, related) for m in newer):
                            raise _Superseded from None
                        if not newer:
                            raise
                        current_head = max(m.seq for m in newer)
                raise MessageConflict("来源 head 持续变化，提交前提无法稳定")

            def narrow(transaction: OwnerTransaction) -> Message:
                existing = reader.get(message_id)
                if existing is not None:
                    return existing
                for message in reader.snapshot(after_seq=head):
                    if _competing(message, writer.source, related):
                        raise _Superseded
                return transaction.append(writer, message_id, body, metadata=metadata)

            return state.transact(narrow)

        try:
            if terminal_tools and _terminal_result(snapshot, writer.source, tools, terminal_tools):
                return commit(uuid4().hex, Output((), "quiet"))
        except _Superseded:
            raise asyncio.CancelledError from None
        if max_steps > 0 and _steps(snapshot, writer.source) >= max_steps:
            raise StepLimit(f"本来源未完成工作已达到 {max_steps} 个模型输出")

        # 2. 生成准备冻结请求、材料、binding 与 Output 身份；恢复不重建不漂移。
        claim: Callable[[int], tuple[str, str | None]] | None = None
        freeze: Callable[[ModelRequest, Materials], tuple[ModelRequest, Materials]] | None = None
        request_override: ModelRequest | None = None
        if state is not None:
            # 输出前驱位置用该来源已有 Output 计数；与边界身份共同固定本代。
            prep_key = (
                f"reply:{reader.session_id}:{writer.source}"
                f":{boundary_id}:{_steps(snapshot, writer.source)}"
            )
            base_seq = reader.head()
            existing = state.transact(lambda transaction: transaction.read(prep_key))
            if existing is not None:
                prep = dict(existing.value)
                if prep.get("binding_id") != model.descriptor.binding_id:
                    raise ModelUnavailableError("生成准备记录的 binding 已失效")
                frozen = reader.snapshot(through_seq=cast(int, prep["base_seq"]))
                prepared = cast(Materials, prep["materials"])
                request_override = _decode_request(prep["request"])
            else:
                prepared = await materials(frozen)

            def freeze_request(
                request: ModelRequest, built: Materials
            ) -> tuple[ModelRequest, Materials]:
                def open_prep(transaction: OwnerTransaction) -> Mapping[str, object]:
                    record = transaction.read(prep_key)
                    if record is None:
                        record = transaction.save(prep_key, {
                            "version": 2, "output_id": uuid4().hex,
                            "request_keys": [uuid4().hex], "base_seq": base_seq,
                            "binding_id": model.descriptor.binding_id,
                            "request": _encode_request(request),
                            "materials": dict(built),
                        }, expected_version=None)
                    return record.value

                value = state.transact(open_prep)
                return _decode_request(value["request"]), cast(Materials, value["materials"])

            def claim_attempt(attempt: int) -> tuple[str, str | None]:
                def advance(transaction: OwnerTransaction) -> tuple[str, str]:
                    record = transaction.read(prep_key)
                    if record is None:
                        raise RuntimeError("生成准备记录缺失")
                    value = dict(record.value)
                    keys = list(cast(Sequence[str], value["request_keys"]))
                    while len(keys) <= attempt:
                        keys.append(uuid4().hex)
                    if keys != value["request_keys"]:
                        record = transaction.save(
                            prep_key, {**value, "request_keys": keys},
                            expected_version=record.version,
                        )
                        value = record.value
                    return cast(str, value["output_id"]), keys[attempt]

                return state.transact(advance)

            freeze = freeze_request
            claim = claim_attempt
        else:
            # 3. 取得材料与组装请求分开，Context 不获得模型调用或检索权。
            prepared = await materials(frozen)
        async with _complete(
            frozen, prepared, source=writer.source, context=context, model=model,
            projection=projection, tools=tools, max_output_tokens=max_output_tokens, reduce=reduce, preview=preview,
            claim=claim,
            freeze=freeze,
            request_override=request_override,
            fallback_key=(
                f"reply:{reader.session_id}:{writer.source}"
                f":{boundary_id}:{_steps(snapshot, writer.source)}"
            ),
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
            # 4. 内容完成后在窄事务内核对前提并提交；失败的草稿绝不触发工具。
            try:
                message = commit(
                    message_id,
                    Output(tuple(parts), "continue" if indices else "complete"),
                    metadata,
                )
            except _Superseded:
                raise asyncio.CancelledError from None
            if not indices:
                return message


REACT = ServiceKey[Callable[..., Awaitable[Message]]]("react.v2")


async def apply(ctx: Context) -> None:
    _ = await ctx.provide(REACT, partial(react, capture_scope=ctx.capture_runtime_scope))

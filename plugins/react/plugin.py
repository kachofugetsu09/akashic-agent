from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, cast
from uuid import uuid4

from core.common.diagnostic_log import log_timing
from agent.plugin_composition import Context, RuntimeScope
from agent.plugin_composition.messages import (
    MessageConflict,
    MessageReader,
    MessageSnapshot,
    MessageWriter,
    OwnerStore,
    OwnerTransaction,
)
from plugins.models.contract import (
    LLMResponse,
    ModelRequest,
    StreamCallback,
)
from plugins.models.contract import (
    ContextLengthError,
    EmptyResponseError,
    OutputLengthError,
    ModelError,
)
from plugins.models.contract import BoundChatModel
from agent.plugin_contracts import (
    CallRef,
    ContentPart,
    Control,
    Input,
    Message,
    Output,
    Part,
    ToolCall,
    ToolResult,
)
from plugins.content.contract import (
    ContentView as ContentView,
)
from plugins.context.contract import (
    ContextBuilder as ContextBuilder,
    SummaryReducer as SummaryReducer,
)
from plugins.models.contract import (
    MessageProjection as MessageProjection,
)
from plugins.react.contract import (
    REACT_ORDERED_V2 as REACT,
)
from agent.plugin_contracts.tools import (
    DecodedCall as DecodedCall,
    StartCheck,
)
from plugins.tools.contract import ToolMenu as ToolMenu

Materials = Mapping[str, object]


api_version = 3
name = "react"
version = "1.0.0"
desc = "组合上下文、模型、内容与工具，不拥有会话或外部效果状态"
inject = ()


Preview = Callable[[str], AbstractContextManager[StreamCallback]]


@dataclass(frozen=True, slots=True)
class StepLimit(ModelError):
    """本次程序达到明确的模型请求上限，保留日志供来源继续控制。"""


class _Superseded(Exception):
    """提交前提检查发现同来源新事实；被替代的旧草稿不写 failure。"""


class _History:
    """按新消息推进本来源的执行视图；旧前缀变化时从事实重建。"""

    def __init__(self, source: str) -> None:
        self.source = source
        self.messages: Sequence[Message] = ()
        self.head = -1
        self.reminder_input_id: str | None = None
        self._boundary = -1
        self._abandoned_upto = -1
        self._calls: dict[CallRef, int] = {}
        self._results: set[CallRef] = set()
        self._succeeded: set[CallRef] = set()
        self._terminal_calls: list[tuple[CallRef, int, str]] = []
        self._outputs: list[int] = []

    def update(self, messages: Sequence[Message]) -> None:
        """已提交读面证明前缀；普通序列仍逐条核对真实 Message 身份。"""
        previous = self.messages
        compatible = (
            messages.extends(previous)
            if type(messages) is MessageSnapshot and type(previous) is MessageSnapshot
            else len(messages) >= len(previous) and all(
                old is new for old, new in zip(previous, messages)
            )
        )
        if not compatible:
            self.__init__(self.source)
            previous = ()
        for message in messages[len(previous):]:
            if message.source != self.source:
                continue
            self.head = message.seq
            body = message.body
            if isinstance(body, Input):
                self.reminder_input_id = message.message_id
            elif isinstance(body, Output):
                if body.finish != "continue":
                    self._boundary = message.seq
                    self._outputs.clear()
                elif any(isinstance(part, ContentPart) and part.kind == "model.facts"
                         for part in body.parts):
                    self._outputs.append(message.seq)
                for index, part in enumerate(body.parts):
                    if isinstance(part, ToolCall):
                        ref = CallRef(message.message_id, index)
                        if ref not in self._results:
                            self._calls[ref] = message.seq
                        if body.finish == "continue":
                            self._terminal_calls.append((ref, message.seq, part.binding_id))
            elif isinstance(body, Control) and body.action == "abandon":
                self._boundary = max(self._boundary, body.through_seq)
                self._abandoned_upto = max(self._abandoned_upto, body.through_seq)
                self._outputs = [seq for seq in self._outputs if seq > body.through_seq]
            elif isinstance(body, ToolResult):
                self._results.add(body.call_ref)
                self._calls.pop(body.call_ref, None)
                if body.outcome == "success":
                    self._succeeded.add(body.call_ref)
        self.messages = messages

    @property
    def steps(self) -> int:
        return len(self._outputs)

    def open_calls(self) -> tuple[tuple[CallRef, ...], tuple[CallRef, ...]]:
        """保留调用顺序；已关闭但未结算的旧调用仍可由之后的 abandon 结算。"""
        pending: list[CallRef] = []
        abandoned: list[CallRef] = []
        for ref, seq in self._calls.items():
            if seq <= self._abandoned_upto:
                abandoned.append(ref)
            elif seq > self._boundary:
                pending.append(ref)
        return tuple(pending), tuple(abandoned)

    def related(self) -> frozenset[CallRef]:
        """提交只关联本次已固定的日志前缀，不恢复旧冻结请求。"""
        pending, abandoned = self.open_calls()
        return frozenset((*pending, *abandoned))

    def terminal(self, tools: ToolMenu, names: frozenset[str]) -> bool:
        calls = [(ref, seq) for ref, seq, binding in self._terminal_calls
                 if tools.name(binding) in names]
        return any(seq > self._boundary and ref in self._succeeded for ref, seq in calls)


def _competing(message: Message, source: str, related: frozenset[CallRef]) -> bool:
    """同来源 Input/Control/任何 Output 或读集内 ToolResult 使旧草稿失效。"""
    if message.source != source:
        return False
    body = message.body
    if isinstance(body, (Input, Control, Output)):
        # 任何同来源 Output 都占据输出前驱位置；本代草稿的前提已被取代。
        return True
    return isinstance(body, ToolResult) and body.call_ref in related


class _OrderGate:
    """前驱提交完成后放行；失败时拒绝后继抢先写 ToolResult。"""

    def __init__(self) -> None:
        self._ready = asyncio.Event()
        self._aborted = False

    def release(self) -> None:
        if not self._aborted:
            self._ready.set()

    def abort(self) -> None:
        self._aborted = True
        self._ready.set()

    async def wait(self) -> None:
        await self._ready.wait()
        if self._aborted:
            raise _CommitDeferred


class _CommitDeferred(Exception):
    """前驱没有按序提交，本调用保留已开始回执。"""


async def _settle(
    tools: ToolMenu,
    call: CallRef,
    capture_scope: Callable[[], RuntimeScope] | None = None,
    *,
    commit_after: _OrderGate | None = None,
    scope: RuntimeScope | None = None,
) -> None:
    """普通取消等待原调用；明确放弃由 Tools 提交终态并释放等待者。"""
    # scope 由 owner task 预先取得；子任务里再取 lease 会丢失授权。
    if scope is None and capture_scope is not None:
        scope = capture_scope()

    async def execute():
        if commit_after is None:
            result = tools.execute(call)
        else:
            result = tools.execute(call, commit_after=commit_after)
        if scope is None:
            return await result
        async with scope:
            return await result

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


def _parallel(reader: MessageReader, tools: ToolMenu, call: CallRef) -> bool:
    """只按当前注册分类；失效引用一律串行，真实错误留给结算路径。"""
    message = reader.get(call.message_id)
    body = None if message is None else message.body
    if not isinstance(body, Output) or call.part_index >= len(body.parts):
        return False
    part = body.parts[call.part_index]
    if not isinstance(part, ToolCall):
        return False
    try:
        return tools.parallel(part.binding_id) is True
    except Exception:
        return False


def _committed(reader: MessageReader, call: CallRef) -> bool:
    """本调用的 ToolResult 已经在日志里；错误回执也算提交完成。"""
    return reader.scan(lambda rows: any(
        isinstance(message.body, ToolResult) and message.body.call_ref == call
        for message in rows
    ))


async def _run_ordered(
    reader: MessageReader,
    tools: ToolMenu,
    call: CallRef,
    gate: _OrderGate,
    commit_after: _OrderGate | None,
    scope: RuntimeScope | None,
) -> None:
    """成功、取消排空或错误回执落盘后放行；提交前失败则中止，避免后继抢先落盘。"""
    try:
        await _settle(tools, call, commit_after=commit_after, scope=scope)
    except BaseException:
        # 已落盘的回执放行后继；提交前失败或被前驱中止时不能把后继门打开。
        if _committed(reader, call):
            gate.release()
        else:
            gate.abort()
        raise
    else:
        gate.release()


async def _settle_group(
    reader: MessageReader,
    tools: ToolMenu,
    calls: Sequence[CallRef],
    limit: int,
    capture_scope: Callable[[], RuntimeScope] | None,
) -> None:
    """同组调用重叠执行；提交门只在前驱完成后打开。"""
    gates = [_OrderGate() for _ in calls]
    tasks: list[asyncio.Task[None]] = []
    in_flight: set[asyncio.Task[None]] = set()
    try:
        for index, call in enumerate(calls):
            while len(in_flight) >= limit:
                done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for finished in done:
                    failure = None if finished.cancelled() else finished.exception()
                    if failure is not None:
                        raise failure
            scope = None if capture_scope is None else capture_scope()
            task = asyncio.create_task(_run_ordered(
                reader, tools, call, gates[index], gates[index - 1] if index else None, scope,
            ))
            tasks.append(task)
            in_flight.add(task)
        for task in tasks:
            await task
    finally:
        for task in tasks:
            _ = task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)


async def _settle_pending(
    reader: MessageReader,
    tools: ToolMenu,
    calls: tuple[CallRef, ...],
    limit: int,
    capture_scope: Callable[[], RuntimeScope] | None,
) -> None:
    """结算未闭段调用：exclusive 调用是屏障，连续 parallel 调用走有界池。"""
    index = 0
    while index < len(calls):
        if not _parallel(reader, tools, calls[index]):
            await _settle(tools, calls[index], capture_scope)
            index += 1
            continue
        end = index + 1
        while end < len(calls) and _parallel(reader, tools, calls[end]):
            end += 1
        await _settle_group(reader, tools, calls[index:end], limit, capture_scope)
        index = end


@asynccontextmanager
async def _complete(
    snapshot: Sequence[Message], prepared: Materials, *, source: str,
    context: ContextBuilder, model: BoundChatModel, projection: MessageProjection,
    tools: ToolMenu, max_output_tokens: int, reduce: SummaryReducer | None,
    preview: Preview | None,
    reminder_input_id: str | None,
    operation_id: str = "",
) -> AsyncGenerator[tuple[LLMResponse, Materials, str, ModelRequest]]:
    """缩减只更新已取得材料中的摘要；provider 容量拒绝最多重试一次。"""
    mark = partial(log_timing, session_id=snapshot[-1].session_id if snapshot else "",
                   source=source, operation_id=operation_id)
    # 1. 本地容量与软水位先交给同一摘要 owner，其他材料不重新获取。
    def build(mats: Materials) -> tuple[ModelRequest, str | None]:
        mark("context.build.begin")
        result = context.build_attempt(snapshot, materials=mats, model=projection,
                                     tools=tools.schemas, max_output_tokens=max_output_tokens,
                                     current_reminder_input_id=reminder_input_id)
        mark("context.build.end")
        return result

    async def reduce_request(mats: Materials, request: ModelRequest, *, force: bool) -> tuple[Materials, Mapping[str, object] | None]:
        """压缩预览只显示状态；可见诊断随冻结材料进入最终 Output。"""
        assert reduce is not None
        notices = list(cast(Sequence[str], mats.get("notices", ())))
        with ExitStack() as status:
            callback = None if preview is None else status.enter_context(preview(uuid4().hex))

            async def report(text: str, retain: bool) -> None:
                if retain:
                    notices.append(text)
                if callback is not None:
                    await callback({"retry_status": text})

            mark("context.reduce.begin")
            summary = await reduce(tuple(snapshot), mats, request, model, projection,
                                   source=source, force=force, on_status=report)
            mark("context.reduce.end")
        return ({**mats, "notices": tuple(notices)} if notices else mats), summary

    request, rejection = build(prepared)
    if rejection is not None and reduce is None:
        raise ContextLengthError(rejection).exception()
    if reduce is not None:
        prepared, summary = await reduce_request(prepared, request, force=rejection is not None)
        if summary is not None and summary != prepared.get("summary"):
            prepared = {**prepared, "summary": summary}
            request, rejection = build(prepared)
        if rejection is not None:
            raise ContextLengthError(rejection).exception()
    # 2. 每份新组装的请求拥有新身份；本次调用内的网络重试仍由 Models 复用该 key。
    with ExitStack() as previews:
        def begin() -> tuple[str, str, StreamCallback | None]:
            """固定本次启动身份，再签发本地预览。"""
            message_id = uuid4().hex
            request_key = uuid4().hex
            callback = None if preview is None else previews.enter_context(preview(message_id))
            mark("request.claimed", request_id=request_key)
            return message_id, request_key, callback

        async def generate(
            request: ModelRequest, request_key: str, callback: StreamCallback | None,
        ) -> LLMResponse:
            mark("model.begin", request_id=request_key)
            response = await model.complete(replace(request, on_delta=callback, request_key=request_key))
            mark("model.end", request_id=request_key)
            return response

        message_id, request_key, callback = begin()
        try:
            response = await generate(request, request_key, callback)
        except (RuntimeError, TimeoutError) as error:
            if not (ModelError.matches(error, ContextLengthError)):
                raise
            previews.close()
            # 强制缩减重试每代至多一次，且只适用于可证明的容量拒绝——
            # send_evidence="rejected" 是 provider HTTP 拒绝应答的正面证据；
            # HTTP 200 流内失败无论是否观察到 delta 都不得缩减后重发同一
            # 请求。
            if reduce is None or getattr(error, "send_evidence", None) != "rejected":
                raise
            prepared, summary = await reduce_request(prepared, request, force=True)
            if summary is None or summary == prepared.get("summary"):
                raise
            prepared = {**prepared, "summary": summary}
            request, rejection = build(prepared)
            if rejection is not None:
                raise ContextLengthError(rejection).exception()
            message_id, request_key, callback = begin()
            # 第二次调用在 except 内；再次拒绝直接上抛，不产生第三次请求。
            response = await generate(request, request_key, callback)
        # 3. 草稿持续到调用者完成解码与 CAS；异常和取消也会释放预览。
        yield response, prepared, message_id, request


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
    max_parallel_calls: int = 1,
    check_start: StartCheck | None = None,
) -> Message:
    """先结算已提交调用，再读日志推理并逐条提交；没有 Turn、Attempt 或历史副本。"""
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("模型请求上限必须是非负整数")
    if type(max_parallel_calls) is not int or max_parallel_calls < 1:
        raise ValueError("并行工具上限必须是正整数")
    if reader.session_id != writer.session_id:
        raise ValueError("ReAct reader 与 writer 必须属于同一 Session")
    history = _History(writer.source)
    while True:
        operation_id = uuid4().hex
        mark = partial(log_timing, session_id=reader.session_id, source=writer.source, operation_id=operation_id)
        mark("react.begin")
        # 1. 放弃区保持串行结算；未闭段里连续的 parallel 调用才重叠。
        initial = await reader.committed_snapshot_async()
        head_before = initial.through_seq
        history.update(initial)
        pending, abandoned = history.open_calls()
        for call in abandoned:
            # 已放弃调用的结算故障必须先阻断本来源：缺回执的调用不能带着未知效果进入新请求。
            await tools.settle_abandoned(call)
        if abandoned:
            # 放弃结算已追加事实，重新读取；普通路径复用同一次扫描的待执行调用。
            history.update(await reader.committed_snapshot_async())
            pending, _ = history.open_calls()
        await _settle_pending(
            reader, tools, pending, max_parallel_calls, capture_scope,
        )
        mark("tools.settled")
        if pending or abandoned or reader.head() != head_before:
            snapshot = await reader.committed_snapshot_async()
        else:
            # 无结算写入且 head 未动：进入结算前的固定前缀仍然有效。
            snapshot = initial
        mark("history.loaded", counts={"messages": len(snapshot)})
        history.update(snapshot)
        head = history.head
        reminder_input_id = history.reminder_input_id

        async def commit(message_id: str, body: Output, metadata: Mapping[str, object] | None = None) -> Message:
            """检查与追加同事务；竞争 Output、新边界或读集内结果都取代旧草稿。"""
            related = history.related()
            if state is None:
                current_head = head
                for _ in range(4):
                    try:
                        return await writer.append_async(
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

            # 准备留在调用方 scope：内容引用与 metadata owner 回调属于插件
            # owner 的原执行上下文，不能随事务带进 worker 线程（评审 #1146）；
            # worker 只接收准备好的不可变 PreparedAppend。
            prepared_append = await writer.prepare_async(message_id, body, metadata=metadata)

            def narrow(transaction: OwnerTransaction) -> Message:
                existing = reader.get(message_id)
                if existing is not None:
                    return existing
                if check_start is not None:
                    check_start(transaction)
                for message in reader.snapshot(after_seq=head):
                    if _competing(message, writer.source, related):
                        raise _Superseded
                return transaction.append_prepared(prepared_append)

            return await state.transact_async(narrow)

        try:
            if terminal_tools and history.terminal(tools, terminal_tools):
                return await commit(uuid4().hex, Output((), "quiet"))
        except _Superseded:
            raise asyncio.CancelledError from None
        if max_steps > 0 and history.steps >= max_steps:
            raise StepLimit(f"本来源未完成工作已达到 {max_steps} 个模型输出").exception()

        # 2. 重启和显式重试都用当前材料发新请求，不恢复旧请求身份。
        mark("preparation.begin")
        prepared = await materials(tuple(snapshot))
        mark("preparation.end")
        async with _complete(
            snapshot, prepared, source=writer.source, context=context, model=model,
            projection=projection, tools=tools, max_output_tokens=max_output_tokens, reduce=reduce, preview=preview,
            reminder_input_id=reminder_input_id,
            operation_id=operation_id,
        ) as (response, prepared, message_id, request):
            # 先检查完整性：即使截断参数恰好是合法 JSON，也不能执行该批工具。
            if response.finish_reason == "length":
                raise OutputLengthError(
                    f"模型生成达到长度限制（输出预算 {request.max_output_tokens} tokens，含推理）；"
                    "回复未完成，本次返回的工具调用未执行。请检查输出预算与模型上下文容量后，用新输入继续。"
                ).exception()
            decoded, metadata = await content.decode(response.content or "", cast(tuple[Mapping[str, object], ...], prepared.get("references", ())))
            parts: list[Part] = list(decoded)
            indices: list[int] = []
            actual_calls: list[ToolCall | ContentPart] = []
            decoded_calls = tuple(tools.decode(call) for call in response.tool_calls)
            mixed_exclusive = len(decoded_calls) > 1 and any(
                decoded.binding_id is not None and decoded.exclusive_batch
                for decoded in decoded_calls
            )
            for call, decoded_call in zip(response.tool_calls, decoded_calls, strict=True):
                indices.append(len(parts))
                if mixed_exclusive:
                    actual = ContentPart("model.tool_rejection", {
                        "name": call.name, "arguments": call.arguments,
                        "error": "此批次包含要求独占的工具；整批未执行，请单独调用该工具。",
                    })
                elif decoded_call.rejection is not None:
                    actual = ContentPart("model.tool_rejection", decoded_call.rejection)
                else:
                    assert decoded_call.binding_id is not None
                    actual = ToolCall(decoded_call.binding_id, decoded_call.arguments)
                actual_calls.append(actual)
                parts.append(actual)
            if not parts:
                raise EmptyResponseError("模型没有产生内容或工具调用；空响应不是 quiet").exception()
            reminder = context.reminder_content(prepared)
            parts.append(projection.facts(
                response,
                indices,
                reminder=reminder,
                reminder_input_id=reminder_input_id if reminder is not None else None,
                actual_calls=actual_calls,
                **({"content_refs": request.content_refs,
                    "content_transformed": request.content_transformed}
                   if request.content_refs or request.content_transformed else {}),
            ))
            summary = cast(Mapping[str, object] | None, prepared.get("summary"))
            if summary is not None:
                parts.append(ContentPart("context.summary", {"reference": summary["reference"]}))
            parts.extend(ContentPart("context.notice", notice)
                         for notice in cast(Sequence[str], prepared.get("notices", ())))
            mark("output.decoded", counts={"tool_calls": len(indices)})
            # 4. 内容完成后在窄事务内核对前提并提交；失败的草稿绝不触发工具。
            try:
                message = await commit(
                    message_id,
                    Output(tuple(parts), "continue" if indices else "complete"),
                    metadata,
                )
            except _Superseded:
                raise asyncio.CancelledError from None
            mark("output.committed", request_id=response.call_record_id or "", parent_operation_id=message.message_id, counts={"seq": message.seq, "tool_calls": len(indices)})
            if not indices:
                return message
        # Loop 保留当前历史，直到下轮真实快照替换它；弱解码缓存才能复用旧消息。
        # 不再使用的准备材料与提交闭包立即释放，结束或取消时释放整个 Loop。
        del commit, prepared


async def apply(ctx: Context) -> None:
    async def owned_react(
        *args: Any, check_start: StartCheck, state: OwnerStore, **kwargs: Any,
    ) -> Message:
        """在来源 scope 中运行；消息提交核对前提，模型请求使用新 key。"""
        async with ctx.runtime_scope():
            return await react(
                *args, check_start=check_start, state=state,
                capture_scope=ctx.capture_runtime_scope, **kwargs,
            )

    _ = await ctx.provide(REACT, owned_react)

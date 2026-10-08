from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, cast
from uuid import uuid4

from core.common.file_io import run_file_io
from core.common.diagnostic_log import log_timing
from agent.plugin_composition import Context, RuntimeScope
from agent.plugin_composition.messages import (
    MessageConflict,
    MessageReader,
    MessageWriter,
    OwnerStore,
    OwnerTransaction,
)
from agent.plugin_composition.models import (
    BoundChatModel,
    ContextLengthError,
    EmptyResponseError,
    OutputLengthError,
    LLMResponse,
    ModelContinuation,
    ModelError,
    ModelRequest,
    ModelUnavailableError,
    StreamCallback,
)
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
    freeze_json,
)
from agent.plugin_contracts.content import (
    ContentView as ContentView,
)
from agent.plugin_contracts.context import (
    ContextBuilder as ContextBuilder,
    SummaryReducer as SummaryReducer,
)
from agent.plugin_contracts.models import (
    MessageProjection as MessageProjection,
)
from agent.plugin_contracts.react import (
    REACT_ORDERED_V2 as REACT,
)
from agent.plugin_contracts.tools import (
    DecodedCall as DecodedCall,
    ToolMenu as ToolMenu,
    StartCheck,
)

Materials = Mapping[str, object]


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


class _History:
    """按新消息推进本来源的执行视图；旧前缀变化时从事实重建。"""

    def __init__(self, source: str) -> None:
        self.source = source
        self.messages: tuple[Message, ...] = ()
        self.head = -1
        self.boundary_id = "initial"
        self.reminder_input_id: str | None = None
        self._boundary = -1
        self._abandoned_upto = -1
        self._calls: dict[CallRef, int] = {}
        self._results: set[CallRef] = set()
        self._succeeded: set[CallRef] = set()
        self._terminal_calls: list[tuple[CallRef, int, str]] = []
        self._outputs: list[int] = []

    def update(self, messages: tuple[Message, ...]) -> None:
        """只解释新增尾部；比较实际 Message 身份，不以相同 id 猜测内容未变。"""
        previous = self.messages
        if len(messages) < len(previous) or any(
            old is not new for old, new in zip(previous, messages)
        ):
            self.__init__(self.source)
            previous = ()
        for message in messages[len(previous):]:
            if message.source != self.source:
                continue
            self.head = message.seq
            body = message.body
            if isinstance(body, Input):
                self.boundary_id = self.reminder_input_id = message.message_id
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
                self.boundary_id = message.message_id
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

    def related(self, messages: tuple[Message, ...]) -> frozenset[CallRef]:
        """正常提交复用当前视图；恢复旧冻结请求时只读取其原有前缀。"""
        history = self
        if messages is not self.messages:
            history = _History(self.source)
            history.update(messages)
        pending, abandoned = history.open_calls()
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
    if isinstance(body, (Input, Control)):
        return True
    if isinstance(body, Output):
        # 任何同来源 Output 都占据输出前驱位置；本代草稿的前提已被取代。
        return True
    return isinstance(body, ToolResult) and body.call_ref in related


def _encode_request(request: ModelRequest) -> Mapping[str, object]:
    """生成准备只冻结模型可见字段；on_delta/request_key 是执行细节。"""
    continuation = request.continuation
    return {
        "messages": request.messages,
        "tools": request.tools,
        "max_output_tokens": request.max_output_tokens,
        "system_prompt": request.system_prompt,
        "tool_choice": request.tool_choice,
        "prompt_cache_key": request.prompt_cache_key,
        "disable_reasoning": request.disable_reasoning,
        "content_refs": request.content_refs,
        "content_transformed": request.content_transformed,
        "continuation": (
            None
            if continuation is None
            else {
                "binding_id": continuation.binding_id,
                "payload": continuation.payload,
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
        content_refs=cast(tuple[tuple[str, int], ...], value.get("content_refs", ())),
        content_transformed=cast(bool, value.get("content_transformed", False)),
        continuation=(
            None
            if continuation is None
            else ModelContinuation(
                cast(str, cast(Mapping[str, object], continuation)["binding_id"]),
                cast(Mapping[str, Any], cast(Mapping[str, object], continuation)["payload"]),
            )
        ),
    )


def _load_entry(entry: Mapping[str, object], state: OwnerStore) -> tuple[ModelRequest, Materials]:
    """在同一快照内恢复准确请求；引用缺失或损坏不能改用当前材料。"""
    def load() -> tuple[ModelRequest, Materials]:
        # 1. 沿原 owner 的冻结记录读取，旧版内联格式仍作为完整起点。
        current = entry
        deltas: list[Mapping[str, object]] = []
        while "encoding" in current:
            if (current.get("encoding") != "request-delta-v1" or len(deltas) >= 15
                    or set(current) != {"encoding", "base_key", "base_attempt", "request", "materials"}):
                raise ValueError("冻结请求的增量格式或引用长度无效")
            key, attempt = current["base_key"], current["base_attempt"]
            if not isinstance(key, str) or not key or type(attempt) is not int or attempt < 0:
                raise ValueError("冻结请求的前驱身份无效")
            record = state.read(key)
            if record is None:
                raise ValueError("冻结请求的前驱记录缺失")
            attempts = record.value.get("attempts")
            if not isinstance(attempts, tuple) or attempt >= len(attempts):
                raise ValueError("冻结请求的前驱 attempt 缺失")
            parent = attempts[attempt]
            if not isinstance(parent, Mapping):
                raise ValueError("冻结请求的前驱内容损坏")
            deltas.append(current)
            current = cast(Mapping[str, object], parent)
        request, materials = current["request"], current["materials"]
        if not isinstance(request, Mapping) or not isinstance(materials, Mapping):
            raise ValueError("冻结请求的完整起点损坏")
        # 2. 从完整起点按原顺序应用差异；数组长度和尾部都来自持久事实。
        for delta in reversed(deltas):
            changed = delta["request"]
            if not isinstance(changed, Mapping):
                raise ValueError("冻结请求的增量内容损坏")
            fields = dict(changed)
            for name in ("messages", "tools", "content_refs"):
                part, old = fields[name], request[name]
                if not isinstance(part, Mapping) or set(part) != {"prefix", "tail"}:
                    raise ValueError("冻结请求的数组增量损坏")
                prefix, tail = part["prefix"], part["tail"]
                if (not isinstance(old, tuple) or type(prefix) is not int
                        or not 0 <= prefix <= len(old) or not isinstance(tail, tuple)):
                    raise ValueError("冻结请求的数组范围损坏")
                fields[name] = old[:prefix] + tail
            request = fields
            if delta["materials"] is not None:
                materials = delta["materials"]
                if not isinstance(materials, Mapping):
                    raise ValueError("冻结请求的材料损坏")
        return _decode_request(request), cast(Materials, materials)

    return state.snapshot(load)


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
    snapshot: tuple[Message, ...], prepared: Materials, *, source: str,
    context: ContextBuilder, model: BoundChatModel, projection: MessageProjection,
    tools: ToolMenu, max_output_tokens: int, reduce: SummaryReducer | None,
    preview: Preview | None,
    reminder_input_id: str | None,
    fallback_key: str | None = None,
    prepare: Callable[[int, ModelRequest, Materials], Awaitable[tuple[ModelRequest, Materials, str, str]]] | None = None,
    resumed: Mapping[int, tuple[ModelRequest, Materials]] | None = None,
    start_at: int = 0,
    resume_rejected: bool = False,
    operation_id: str = "",
) -> AsyncGenerator[tuple[LLMResponse, Materials, str, ModelRequest]]:
    """缩减只更新已取得材料中的摘要；provider 容量拒绝最多重试一次。

    每个 attempt 的请求、材料和启动身份一次提交；恢复时按原字节精确重放，
    ContextLength 缩减后的第二次请求同样不重建不漂移。
    """
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

    prepared_attempt = prepared
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
            summary = await reduce(snapshot, mats, request, model, projection,
                                   source=source, force=force, on_status=report)
            mark("context.reduce.end")
        return ({**mats, "notices": tuple(notices)} if notices else mats), summary

    if resumed is not None and start_at in resumed:
        # 恢复直接命中已冻结的当前请求与材料，不重建、不重跑缩减。
        request, prepared_attempt = resumed[start_at]
    else:
        request, rejection = build(prepared_attempt)
        if rejection is not None and reduce is None:
            raise ContextLengthError(rejection)
        if reduce is not None:
            prepared_attempt, summary = await reduce_request(prepared_attempt, request, force=rejection is not None)
            if summary is not None and summary != prepared_attempt.get("summary"):
                prepared_attempt = {**prepared_attempt, "summary": summary}
                request, rejection = build(prepared_attempt)
            if rejection is not None:
                raise ContextLengthError(rejection)
    prepared = prepared_attempt
    # 2. 请求与启动身份由同一 owner 原子提交，再打开流式预览并调用模型。
    with ExitStack() as previews:
        async def begin(
            attempt: int, request: ModelRequest, mats: Materials,
        ) -> tuple[ModelRequest, Materials, str, str | None, StreamCallback | None]:
            """固定本次请求与启动身份，再签发本地预览。"""
            if prepare is None:
                message_id = uuid4().hex
                request_key = (
                    None if fallback_key is None else f"{fallback_key}:{attempt}"
                )
            else:
                mark("request.freeze.begin")
                request, mats, message_id, request_key = await prepare(attempt, request, mats)
                mark("request.freeze.end")
            callback = None if preview is None else previews.enter_context(preview(message_id))
            mark("request.claimed", request_id=request_key or "")
            return request, mats, message_id, request_key, callback

        attempt = start_at
        request, prepared, message_id, request_key, callback = await begin(attempt, request, prepared)
        try:
            if resume_rejected:
                # 该 attempt 的 key 已有耐久的 provider 容量拒绝结算：
                # 不重发已失败的原请求，直接续跑已批准的本地缩减阶段。
                rejected = ContextLengthError("provider 容量拒绝已耐久结算")
                rejected.send_evidence = "rejected"
                raise rejected
            mark("model.begin", request_id=request_key or "")
            response = await model.complete(replace(request, on_delta=callback, request_key=request_key))
            mark("model.end", request_id=request_key or "")
        except ContextLengthError as error:
            previews.close()
            # 强制缩减重试每代至多一次，且只适用于可证明的容量拒绝——
            # send_evidence="rejected" 是 provider HTTP 拒绝应答的正面证据；
            # HTTP 200 流内失败无论是否观察到 delta 都不得缩减后重发同一
            # 请求；恢复续发 attempt>=1 的冻结请求再遭拒绝同样终结。
            if reduce is None or attempt != 0 or getattr(error, "send_evidence", None) != "rejected":
                raise
            prepared, summary = await reduce_request(prepared, request, force=True)
            if summary is None or summary == prepared.get("summary"):
                raise
            attempt += 1
            if resumed is not None and attempt in resumed:
                request, prepared = resumed[attempt]
            else:
                prepared = {**prepared, "summary": summary}
                request, rejection = build(prepared)
                if rejection is not None:
                    raise ContextLengthError(rejection)
            request, prepared, message_id, request_key, callback = await begin(attempt, request, prepared)
            mark("model.begin", request_id=request_key or "")
            response = await model.complete(replace(request, on_delta=callback, request_key=request_key))
            mark("model.end", request_id=request_key or "")
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
        head_before = reader.head()
        initial = await reader.snapshot_async(through_seq=head_before)
        history.update(initial)
        pending, abandoned = history.open_calls()
        for call in abandoned:
            # 已放弃调用的结算故障必须先阻断本来源：缺回执的调用不能带着未知效果进入新请求。
            await tools.settle_abandoned(call)
        if abandoned:
            # 放弃结算已追加事实，重新读取；普通路径复用同一次扫描的待执行调用。
            history.update(await reader.snapshot_async(through_seq=reader.head()))
            pending, _ = history.open_calls()
        await _settle_pending(
            reader, tools, pending, max_parallel_calls, capture_scope,
        )
        mark("tools.settled")
        if pending or abandoned or reader.head() != head_before:
            snapshot = await reader.snapshot_async(through_seq=reader.head())
        else:
            # 无结算写入且 head 未动：进入结算前的固定前缀仍然有效。
            snapshot = initial
        mark("history.loaded", counts={"messages": len(snapshot)})
        history.update(snapshot)
        head = history.head
        boundary_id = history.boundary_id
        reminder_input_id = history.reminder_input_id
        frozen = snapshot

        async def commit(message_id: str, body: Output, metadata: Mapping[str, object] | None = None) -> Message:
            """检查与追加同事务；竞争 Output、新边界或读集内结果都取代旧草稿。"""
            related = history.related(frozen)
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
            raise StepLimit(f"本来源未完成工作已达到 {max_steps} 个模型输出")

        # 2. 生成准备冻结请求、材料、binding 与 Output 身份；恢复不重建不漂移。
        prepare: Callable[[int, ModelRequest, Materials], Awaitable[tuple[ModelRequest, Materials, str, str]]] | None = None
        resumed: dict[int, tuple[ModelRequest, Materials]] | None = None
        start_at = 0
        resume_rejection = False
        mark("preparation.begin")
        if state is not None:
            # 输出前驱位置用该来源已有 Output 计数；与边界身份共同固定本代。
            prep_base = (
                f"reply:{reader.session_id}:{writer.source}"
                f":{boundary_id}:{history.steps}"
            )
            base_seq = reader.head()

            def prep_attempts(
                prep: Mapping[str, object]
            ) -> list[Mapping[str, object] | None]:
                entries: list[Mapping[str, object] | None] = list(
                    cast(Sequence[Mapping[str, object] | None], prep.get("attempts") or ())
                )
                if not entries and "request" in prep:
                    # v2 记录的首个请求/材料视为 attempt 0。
                    entries = [{"request": prep["request"], "materials": prep["materials"]}]
                return entries

            # 逐代核对：最后冻结 attempt 的 key 已终结失败时，同一代不得借
            # 重启/新随机 key 重获预算。只有真实耐久来源事实——该代冻结之后
            # 本 lane 内提交的 Input 或 resume Control——才构成新业务边界，
            # 允许开启新一代准备；否则如实停摆。冻结但未 claim 的后续
            # attempt（崩溃于 freeze/claim 之间）不丢弃，直接续用。
            prep_key = prep_base
            existing = state.snapshot(lambda: state.read(prep_key))
            generation = 0
            resume_rejection = False
            while existing is not None:
                keys = list(cast(Sequence[str], existing.value.get("request_keys", ())))
                current = len(prep_attempts(existing.value)) - 1
                if not (0 <= current < len(keys) and keys):
                    break
                recovery = model.key_recovery(keys[current])
                if recovery == "open":
                    break
                if reduce is not None and current == 0 and recovery == "rejected":
                    # 可证明的 provider 容量拒绝且本代尚未缩减过：进程死于
                    # 缩减/冻结之间时留在本代续跑本地缩减阶段——不重发已失败的
                    # 原请求，不开新代，也不需要新来源事实。attempt>=1 的再次
                    # 拒绝即终态：强制缩减每代至多一次，跨重启也不重获额度。
                    resume_rejection = True
                    break
                base = cast(int, existing.value["base_seq"])
                newer = tuple(
                    message
                    for message in reader.snapshot(after_seq=base)
                    if message.seq > base and message.source == writer.source
                )
                has_input = any(isinstance(message.body, Input) for message in newer)
                has_resume = any(
                    isinstance(message.body, Control) and message.body.action == "resume"
                    for message in newer
                )
                # 显式 resume 允许重新生成模型响应；工具效果仍查原 key 与回执。
                # 自动恢复只沿 Models 已保存的 next_attempt_at，不靠新 key 绕过终态。
                qualified = has_input or has_resume
                if not qualified:
                    raise ModelUnavailableError(
                        f"该来源边界的生成准备已终结失败；需要新的来源事实才能恢复 "
                        f"(prep={prep_key} source={writer.source} base={base} key={keys[current]})"
                    )
                generation += 1
                prep_key = f"{prep_base}#{generation}"
                existing = state.snapshot(lambda: state.read(prep_key))
            if existing is not None:
                prep = dict(existing.value)
                if prep.get("binding_id") != model.descriptor.binding_id:
                    raise ModelUnavailableError("生成准备记录的 binding 已失效")
                frozen = await reader.snapshot_async(through_seq=cast(int, prep["base_seq"]))
                attempts = prep_attempts(prep)
                resumed = {
                    index: _load_entry(entry, state)
                    for index, entry in enumerate(attempts)
                    if entry is not None
                }
                # 恢复从最后冻结的 attempt 继续，不重放旧失败请求、不重跑 reduce。
                start_at = max(resumed, default=0)
                prepared = (
                    cast(Materials, resumed[start_at][1])
                    if resumed
                    else await materials(frozen)
                )
            else:
                prepared = await materials(frozen)

            async def prepare_request(
                attempt: int, request: ModelRequest, built: Materials
            ) -> tuple[ModelRequest, Materials, str, str]:
                """在同一来源检查下提交请求、材料和稳定启动身份。"""
                # Context 与模型句柄只在当前 scope 读取，worker 接收已冻结的请求字段。
                # 冻结请求恒完整保存：增量基线曾按 (session, source) 进程级缓存且用
                # 普通相等比较前缀（评审 #1122/#1123），准确重放合同优先于写入压缩。
                binding_id = model.descriptor.binding_id
                encoded_request = cast(Mapping[str, object], freeze_json(_encode_request(request)))
                fixed_materials = cast(Materials, freeze_json(built))
                new_entry: Mapping[str, object] = {"request": encoded_request, "materials": fixed_materials}

                def open_prep(transaction: OwnerTransaction) -> tuple[Mapping[str, object], bool, str, str]:
                    """新准备只保存一次；旧准备复用原请求并补齐尚未领取的身份。"""
                    if check_start is not None:
                        check_start(transaction)
                    record = transaction.read(prep_key)
                    if record is None:
                        value: dict[str, object] = {
                            "version": 3, "output_id": uuid4().hex,
                            "request_keys": [uuid4().hex], "base_seq": base_seq,
                            "binding_id": binding_id,
                            "attempts": [],
                        }
                    else:
                        value = dict(record.value)
                    entries: list[Mapping[str, object] | None] = list(
                        cast(Sequence[Mapping[str, object] | None], value.get("attempts") or ())
                    )
                    if not entries and "request" in value:
                        entries = [{"request": value["request"], "materials": value["materials"]}]
                    while len(entries) <= attempt:
                        entries.append(None)
                    created = entries[attempt] is None
                    if created:
                        entries[attempt] = new_entry
                    keys = tuple(cast(Sequence[str], value["request_keys"]))
                    while len(keys) <= attempt:
                        keys += (uuid4().hex,)
                    started = tuple(cast(Sequence[int], value.get("started_attempts", ())))
                    needs_start = attempt not in started
                    if created or needs_start or keys != tuple(cast(Sequence[str], value["request_keys"])):
                        record = transaction.save(
                            prep_key, {**value, "attempts": entries, "request_keys": keys,
                                       "started_attempts": (*started, attempt) if needs_start else started},
                            expected_version=None if record is None else record.version,
                        )
                        value = dict(record.value)
                        entries = list(cast(Sequence[Mapping[str, object]], value["attempts"]))
                    return (cast(Mapping[str, object], entries[attempt]), created,
                            cast(str, value["output_id"]), keys[attempt])

                entry, created, output_id, request_key = await state.transact_async(open_prep)
                # 刚提交的请求已不可变；只有恢复旧记录时才重新解码。
                if created:
                    saved_request, saved_materials = replace(request, on_delta=None, request_key=None), fixed_materials
                else:
                    # 恢复读取支持旧版增量格式；损坏引用报错，不重新生成请求。
                    saved_request, saved_materials = _load_entry(entry, state)
                return saved_request, saved_materials, output_id, request_key

            prepare = prepare_request
        else:
            # 3. 取得材料与组装请求分开，Context 不获得模型调用或检索权。
            prepared = await materials(frozen)
        mark("preparation.end")
        async with _complete(
            frozen, prepared, source=writer.source, context=context, model=model,
            projection=projection, tools=tools, max_output_tokens=max_output_tokens, reduce=reduce, preview=preview,
            reminder_input_id=reminder_input_id,
            prepare=prepare,
            resumed=resumed,
            start_at=start_at,
            resume_rejected=resume_rejection,
            operation_id=operation_id,
            fallback_key=(
                f"reply:{reader.session_id}:{writer.source}"
                f":{boundary_id}:{history.steps}"
            ),
        ) as (response, prepared, message_id, request):
            # 先检查完整性：即使截断参数恰好是合法 JSON，也不能执行该批工具。
            if response.finish_reason == "length":
                raise OutputLengthError(
                    f"模型生成达到长度限制（输出预算 {request.max_output_tokens} tokens，含推理）；"
                    "回复未完成，本次返回的工具调用未执行。请检查输出预算与模型上下文容量后，用新输入继续。"
                )
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
                raise EmptyResponseError("模型没有产生内容或工具调用；空响应不是 quiet")
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
        del commit, prepared, resumed, prepare


async def apply(ctx: Context) -> None:
    async def owned_react(
        *args: Any, check_start: StartCheck, state: OwnerStore, **kwargs: Any,
    ) -> Message:
        """来源前提和首次生成 intent 共同提交，再交给 Models 的固定 key。"""
        async with ctx.runtime_scope():
            return await react(
                *args, check_start=check_start, state=state,
                capture_scope=ctx.capture_runtime_scope, **kwargs,
            )

    _ = await ctx.provide(REACT, owned_react)

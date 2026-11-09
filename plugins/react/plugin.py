from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
from types import TracebackType
from typing import Any, cast
from uuid import uuid4

from core.common.diagnostic_log import log_timing
from core.common.file_io import run_file_io
from agent.plugin_composition import Context, RuntimeScope
from agent.plugin_composition.tasks import interrupt_requested
from plugins.ledger.contract import (
    HistoryWindow,
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
    ContentSafetyError,
    ContextLengthError,
    EmptyResponseError,
    InputReductionRequired,
    OutputLengthError,
    ModelError,
    RetryWaitInterrupted,
    content_rejection_reason,
    is_content_rejection,
)
from plugins.models.contract import BoundChatModel
from plugins.ledger.contract import (
    CallRef,
    ContentPart,
    Control,
    Input,
    Message,
    Output,
    Part,
    ToolCall,
    ToolResult,
    json_value,
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
from plugins.tools.contract import (
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
    """本次程序达到明确的模型请求上限，保留日志供来源继续控制。

    只在未完成区段含未回执工具调用时抛出：这些调用可能已有外部效果，
    引擎不能替用户作废它们，只能停下并如实报告。
    """


# 引擎放弃未完成区段的持久原因前缀；来源与界面按它区分人工与自动放弃。
STEP_LIMIT_ABANDON = "步数上限放弃"
# 撞线但本段还有未回执调用时的失败前缀：来源只记 failure，界面按它提供放弃出口。
# 这类失败重试不会生效——计数与未回执调用都还在，下一次仍在请求之前撞同一条门槛。
STEP_LIMIT_FAILURE = "步数上限失败"


def step_limit_abandon_reason(max_steps: int) -> str:
    return (
        f"{STEP_LIMIT_ABANDON}：本轮连续 {max_steps} 次模型请求都没有收尾，"
        "本轮到此结束，未完成的工作不再继续；接下来的一句话将重新开始计数。"
    )


def step_limit_failure_reason(max_steps: int, unfinished: int) -> str:
    return (
        f"{STEP_LIMIT_FAILURE}：本轮连续 {max_steps} 次模型请求都没有收尾，已经停下。"
        f"其中 {unfinished} 个工具调用没有拿到回执，状态未知（可能已部分执行），"
        "所以没有自动作废。请先检查当前状态；重试不会生效，发新消息会重新开始计数并接着结算这些调用。"
        "确实要丢掉这一轮，请点「放弃本轮未完成的工作」或发送 /abandon。"
    )


class _Superseded(Exception):
    """提交前提检查发现同来源新事实；被替代的旧草稿不写 failure。"""


TOOL_FAILURE_LIMIT = 3
TOOL_FAILURE_HINT_LIMIT = 2
# 已提交拒绝的原因前缀；历史重放据此区分保护提示与原工具错误。
TOOL_FAILURE_HINT = "重复失败保护："


def _tool_key(name: str, arguments: Mapping[str, object] | str) -> tuple[str, str]:
    return name, json.dumps(json_value(arguments), sort_keys=True, separators=(",", ":"))


@dataclass(slots=True)
class _ToolAttempt:
    key: tuple[str, str]
    output_id: str
    name: str
    ready: bool = False
    error: str | None = None
    blocked: bool = False


class _ToolFailures:
    """按调用顺序折叠本轮已提交失败；只保留连续失败和未结算批次。"""

    def __init__(self, tool_name: Callable[[str], str]) -> None:
        self._name = tool_name
        self.reset()

    # 新输入、停止、收尾或放弃开启新的失败计数，迟到的旧结果（含「未执行」回执）不参与。
    def reset(self) -> None:
        self._pending: dict[CallRef, _ToolAttempt] = {}
        self._queue: deque[_ToolAttempt] = deque()
        self._key: tuple[str, str] | None = None
        self._count = 0
        self._blocked = 0
        self._blocked_output: str | None = None
        self._failed_name = ""
        self._error = ""

    # 回执可能晚于同批协议拒绝；等前驱就绪后再按调用位置计数。
    def update(self, message: Message) -> None:
        body = message.body
        if isinstance(body, Input) or isinstance(body, Control) and body.action in {"abandon", "pause"}:
            self.reset()
        elif isinstance(body, Output):
            if body.finish != "continue":
                self.reset()
                return
            self._add_output(message.message_id, body)
        elif isinstance(body, ToolResult):
            attempt = self._pending.pop(body.call_ref, None)
            if attempt is not None:
                attempt.ready = True
                if body.outcome in {"error", "denied"}:
                    detail = " ".join(str(part.value)[:500] for part in body.parts[:3])
                    attempt.error = f"{body.outcome}: {detail}"[:500]
        self._drain()

    # 使用原模型调用名与参数；间接调用的 wire 事实已由投影 owner 保存。
    def _add_output(self, message_id: str, output: Output) -> None:
        wire: Mapping[str, Mapping[str, object]] = {}
        for part in output.parts:
            if isinstance(part, ContentPart) and part.kind == "model.facts":
                facts = cast(Mapping[str, object], part.value)
                wire = cast(Mapping[str, Mapping[str, object]], facts.get("wire_tool_calls", {}))
        for index, part in enumerate(output.parts):
            if isinstance(part, ToolCall):
                name = self._name(part.binding_id)
                raw = wire.get(str(index))
                key = (_tool_key(name, part.arguments) if raw is None
                       else _tool_key(cast(str, raw["name"]), cast(Mapping[str, object], raw["arguments"])))
                attempt = _ToolAttempt(key, message_id, name)
                self._pending[CallRef(message_id, index)] = attempt
            elif isinstance(part, ContentPart) and part.kind == "model.tool_rejection":
                rejected = cast(Mapping[str, object], part.value)
                error = cast(str, rejected["error"])
                key = _tool_key(cast(str, rejected["name"]), cast(Mapping[str, object] | str, rejected["arguments"]))
                attempt = _ToolAttempt(key, message_id, key[0], ready=True,
                                       error=error[:500], blocked=error.startswith(TOOL_FAILURE_HINT))
            else:
                continue
            self._queue.append(attempt)

    # success/interrupted 或不同调用打断连续失败；保护提示不覆盖原始错误。
    def _drain(self) -> None:
        while self._queue and self._queue[0].ready:
            attempt = self._queue.popleft()
            if attempt.key != self._key or attempt.error is None:
                self._key = attempt.key
                self._count = self._blocked = 0
                self._blocked_output = None
                self._failed_name = ""
                self._error = ""
            if attempt.error is None:
                continue
            if attempt.blocked:
                if attempt.output_id != self._blocked_output:
                    self._blocked += 1
                    self._blocked_output = attempt.output_id
            else:
                self._count += 1
                self._failed_name = attempt.name
                self._error = attempt.error

    # 达到失败门槛后，下一次相同请求只保存可行动的协议拒绝。
    def rejection(self, name: str, arguments: Mapping[str, object] | str) -> str | None:
        if self._key != _tool_key(name, arguments) or self._count < TOOL_FAILURE_LIMIT:
            return None
        return (
            f"{TOOL_FAILURE_HINT}工具 {self._failed_name} 已用相同参数连续失败 {self._count} 次，"
            f"本次未执行。最后错误：{self._error}。请换方案或向用户说明；"
            "若仍重复，本轮将停止。先检查现场，失败不证明之前没有产生外部效果。"
        )

    # 提示之后仍重复才结束，批内多个真实失败不能跳过提示直接终止。
    def failure(self) -> str | None:
        if self._blocked < TOOL_FAILURE_HINT_LIMIT:
            return None
        assert self._key is not None
        return (
            f"工具重复失败：{self._failed_name} 使用相同参数连续失败 {self._count} 次，"
            f"收到纠正提示后仍重复请求，本轮已停止。最后错误：{self._error}。"
            "请检查工具状态或权限后发送新输入继续；新输入会重新计数。"
        )


class _History:
    """按新消息推进本来源的执行视图；旧前缀变化时从事实重建。"""

    def __init__(self, source: str, tool_name: Callable[[str], str]) -> None:
        self.source = source
        self._tool_name = tool_name
        self.failures = _ToolFailures(tool_name)
        self.messages: Sequence[Message] = ()
        self.head = -1
        self.reminder_input_id: str | None = None
        self._boundary = -1
        self._abandoned_upto = -1
        # 停止覆盖它之前已提交的全部调用；resume 只撤销新输入到来之前的停止。
        self._stopped_upto = -1
        self._stop_floor = -1
        self._calls: dict[CallRef, int] = {}
        self._results: set[CallRef] = set()
        self._succeeded: set[CallRef] = set()
        self._terminal_calls: list[tuple[CallRef, int, str]] = []
        self._outputs: list[int] = []
        # Own Input/resume grants a new budget; complete/abandon closes the old one.
        self.generation_rejections: list[tuple[int, str]] = []

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
            self.__init__(self.source, self._tool_name)
            previous = ()
        for message in messages[len(previous):]:
            if message.source != self.source:
                continue
            self.head = message.seq
            self.failures.update(message)
            body = message.body
            if isinstance(body, Input):
                self.reminder_input_id = message.message_id
                self.generation_rejections.clear()
                self._stop_floor = self._stopped_upto
                # 同一回合内只有用户插话会写 Input，后台来源的 Input 各有自己的回合预算。
                self._outputs.clear()
            elif isinstance(body, Output):
                if body.finish != "continue":
                    self._boundary = message.seq
                    self._outputs.clear()
                    self.generation_rejections.clear()
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
                    elif isinstance(part, ContentPart) and part.kind == "model.facts" and body.finish == "continue":
                        facts = cast(Mapping[str, object], part.value)
                        if "generation_rejection" in facts:
                            self.generation_rejections.append((message.seq, cast(str, facts["generation_rejection"])))
            elif isinstance(body, Control) and body.action == "abandon":
                self._boundary = max(self._boundary, body.through_seq)
                self._abandoned_upto = max(self._abandoned_upto, body.through_seq)
                self._outputs = [seq for seq in self._outputs if seq > body.through_seq]
                self.generation_rejections = [(seq, kind) for seq, kind in self.generation_rejections if seq > body.through_seq]
            elif isinstance(body, Control) and body.action == "pause":
                self._stopped_upto = message.seq
            elif isinstance(body, Control) and body.action == "resume":
                self._stopped_upto = self._stop_floor
                self.generation_rejections.clear()
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
        """保留调用顺序；已关闭但未结算的旧调用仍可由之后的 abandon 结算，停止只结算未闭段。"""
        pending: list[CallRef] = []
        abandoned: list[CallRef] = []
        for ref, seq in self._calls.items():
            if seq <= self._abandoned_upto or self._boundary < seq < self._stopped_upto:
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
    """普通撤权等工具自然结束；用户停止的中断才传给工具，再等待 Tools 保存终态并释放资源。"""
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
            # 中断只传给工具一次；之后的取消不能打断 Tools 的回执提交和 owner 收尾。
            interrupted = interrupt_requested()
            if interrupted:
                _ = work.cancel()
            while not work.done():
                try:
                    _ = await asyncio.shield(work)
                except asyncio.CancelledError:
                    # 新输入撤权后用户又点停止：等待中升级为中断。
                    if not interrupted and interrupt_requested():
                        interrupted = True
                        _ = work.cancel()
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
    return reader.result_committed(call)


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
    except asyncio.CancelledError:
        # 同组调用各自读取当前 Task 的中断状态：撤权时等结果，中断时停下。
        for task in tasks:
            _ = task.cancel()
        if tasks:
            await _drain_group(tasks)
        raise
    finally:
        # 普通失败仍收集同批已启动调用。
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)


# 同组调用排空期间又收到中断时继续转给它们；排空完成前不离开。
async def _drain_group(tasks: Sequence[asyncio.Task[None]]) -> None:
    while not all(task.done() for task in tasks):
        try:
            _ = await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
        except asyncio.CancelledError:
            if interrupt_requested():
                for task in tasks:
                    _ = task.cancel()


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


class _StepAttempts:
    """退避被叫醒时结束本步且不提交，未用的尝试额度带给下一步；生成成功后清空。"""

    def __init__(self) -> None:
        self.remaining: int | None = None

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc is None:
            self.remaining = None
            return False
        failure = ModelError.read(exc, RetryWaitInterrupted)
        if not isinstance(failure, RetryWaitInterrupted):
            return False
        self.remaining = failure.remaining_attempts
        return True


@asynccontextmanager
async def _complete(
    snapshot: Sequence[Message], prepared: Materials, *, source: str,
    context: ContextBuilder, model: BoundChatModel, projection: MessageProjection,
    tools: ToolMenu, max_output_tokens: int, reduce: SummaryReducer | None,
    preview: Preview | None,
    reminder_input_id: str | None,
    operation_id: str = "",
    max_retry_wait: float | None = None,
    retry_wake: asyncio.Event | None = None,
    max_attempts: int | None = None,
) -> AsyncGenerator[tuple[LLMResponse, Materials, str, ModelRequest]]:
    """缩减只更新已取得材料中的摘要；provider 容量拒绝最多重试一次。"""
    mark = partial(log_timing, session_id=snapshot[-1].session_id if snapshot else "",
                   source=source, operation_id=operation_id)
    # 1. 本地容量与软水位先交给同一摘要 owner，其他材料不重新获取。
    # 组装要折叠全史并核对调用账，冷读可达数秒，放进 worker 不占事件循环。
    def build(mats: Materials) -> tuple[ModelRequest, str | None]:
        mark("context.build.begin")
        result = context.build_attempt(snapshot, materials=mats, model=projection,
                                     tools=tools.schemas, max_output_tokens=max_output_tokens,
                                     current_reminder_input_id=reminder_input_id)
        mark("context.build.end")
        return result

    async def reduce_request(
        mats: Materials, request: ModelRequest, *, force: bool, required: bool = True,
    ) -> tuple[Materials, Mapping[str, object] | None]:
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
            reducer = reduce if required else partial(reduce, required=False)
            summary = await reducer(tuple(snapshot), mats, request, model, projection,
                                    source=source, force=force, on_status=report)
            mark("context.reduce.end")
        return ({**mats, "notices": tuple(notices)} if notices else mats), summary

    request, rejection = await run_file_io(lambda: build(prepared))
    if rejection is not None and reduce is None:
        raise ContextLengthError(rejection).exception()
    if reduce is not None:
        prepared, summary = await reduce_request(prepared, request, force=rejection is not None)
        if summary is not None and summary != prepared.get("summary"):
            prepared = {**prepared, "summary": summary}
            request, rejection = await run_file_io(lambda: build(prepared))
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
            *, max_attempts: int | None = None, reduce_on_empty: bool = False,
        ) -> LLMResponse:
            mark("model.begin", request_id=request_key)
            try:
                response = await model.complete(replace(
                    request, on_delta=callback, request_key=request_key, max_retry_wait=max_retry_wait,
                    retry_wake=retry_wake, max_attempts=max_attempts, reduce_on_empty=reduce_on_empty,
                ))
            except (RuntimeError, TimeoutError) as error:
                if not is_content_rejection(error):
                    raise
                raise ModelError.change(error, message=content_rejection_reason(
                    str(error), may_be_config=not ModelError.matches(error, ContentSafetyError),
                )) from error
            mark("model.end", request_id=request_key)
            return response

        message_id, request_key, callback = begin()
        try:
            response = await generate(
                request, request_key, callback, max_attempts=max_attempts, reduce_on_empty=reduce is not None,
            )
            # 没有摘要 owner 时缩减不可用，保持长度拒绝的反馈恢复。
            if reduce is not None:
                _check_output_room(response)
        except (RuntimeError, TimeoutError) as error:
            failure = ModelError.read(error, ContextLengthError, InputReductionRequired)
            if failure is None:
                raise
            previews.close()
            # A capacity refusal requires a real reduction; empty output only asks for an optimization.
            # The follow-up key spends the remaining budget and cannot request another empty-output reduction.
            if reduce is None or isinstance(failure, ContextLengthError) and failure.response_delta_seen:
                raise
            required = isinstance(failure, ContextLengthError)
            prepared, summary = await reduce_request(prepared, request, force=True, required=required)
            if summary is not None and summary != prepared.get("summary"):
                prepared = {**prepared, "summary": summary}
                request, rejection = await run_file_io(lambda: build(prepared))
                if rejection is not None:
                    raise ContextLengthError(rejection).exception()
            elif required:
                raise
            else:
                notices = cast(Sequence[str], prepared.get("notices", ()))
                prepared = {**prepared, "notices": (*notices, "本次压缩优化未缩减输入，保留原请求并继续本步剩余尝试额度。")}
            message_id, request_key, callback = begin()
            # 第二次调用在 except 内；再次拒绝直接上抛，不产生第三次请求。
            response = await generate(
                request, request_key, callback,
                max_attempts=failure.remaining_attempts if isinstance(failure, InputReductionRequired) else max_attempts,
            )
            _check_output_room(response)
        # 3. 草稿持续到调用者完成解码与 CAS；异常和取消也会释放预览。
        yield response, prepared, message_id, request


# 零输出 length 说明输入占满实际窗口，减少输出的反馈帮不上忙；按容量拒绝走单次强制缩减，有正文、推理或被丢弃片段的不算。
def _check_output_room(response: LLMResponse) -> None:
    usage = response.usage
    if (
        response.finish_reason != "length"
        or (response.content and response.content.strip())
        or response.thinking
        or response.tool_calls
        or response.output_discarded
        or (usage is not None and (usage.output_tokens or usage.reasoning_output_tokens))
    ):
        return
    raise ContextLengthError(
        "模型没有产生任何输出就达到了长度限制，输入可能已占满模型实际可用的上下文窗口。"
        "压缩后仍无法生成；请减少上下文或换用窗口更大的模型后继续。"
    ).exception()


# Rejected generations become durable feedback, never executable tool calls.
def _response_rejection(reason: str | None, budget: int) -> str | None:
    if reason == "length":
        return (
            f"上次模型生成达到长度限制（输出预算 {budget} tokens，含推理）；"
            "回复未完成，全部工具调用均未执行。请减少推理和单次输出，分段完成任务。"
        )
    if reason == "malformed_function_call":
        return "上次模型生成了无法解析的工具调用，全部调用均未执行。请按工具 schema 重新生成完整的 JSON 参数。"
    return None


# Enforce the same limit after a commit and before resuming a crashed generation.
def _check_generation_recovery(count: int, reason: str | None, budget: int) -> None:
    if count <= 3:
        return
    rejection = _response_rejection(reason, budget)
    assert rejection is not None
    failure = OutputLengthError if reason == "length" else ModelError
    raise failure(rejection + " 已尝试 3 次恢复，现已停止；请调整输出预算或任务要求后继续。").exception()


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
    max_retry_wait: float | None = None,
    reduce: SummaryReducer | None = None,
    preview: Preview | None = None,
    terminal_tools: frozenset[str] = frozenset(),
    capture_scope: Callable[[], RuntimeScope] | None = None,
    state: OwnerStore | None = None,
    max_parallel_calls: int = 1,
    check_start: StartCheck | None = None,
    before_step: Callable[[], Awaitable[None]] | None = None,
    retry_wake: asyncio.Event | None = None,
    aborted: Callable[[str], Awaitable[Message]] | None = None,
    window: HistoryWindow | None = None,
) -> Message:
    """先结算已提交调用，再读日志推理并逐条提交；没有 Turn、Attempt 或历史副本。

    给出 window 时只读摘要覆盖之后的窗口；窗口失效（前缀版本变化、迟到回执）时账本改给完整历史，
    组装按快照自身的起点理解。
    """
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("模型请求上限必须是非负整数")
    if type(max_parallel_calls) is not int or max_parallel_calls < 1:
        raise ValueError("并行工具上限必须是正整数")
    if reader.session_id != writer.session_id:
        raise ValueError("ReAct reader 与 writer 必须属于同一 Session")
    history = _History(writer.source, tools.name)
    attempts = _StepAttempts()
    while True:
        operation_id = uuid4().hex
        mark = partial(log_timing, session_id=reader.session_id, source=writer.source, operation_id=operation_id)
        mark("react.begin")
        # 1. 放弃与停止区保持串行结算；未闭段里连续的 parallel 调用才重叠。
        initial = await reader.committed_window_async(window)
        head_before = initial.through_seq
        history.update(initial)
        pending, abandoned = history.open_calls()
        for call in abandoned:
            # 已放弃调用的结算故障必须先阻断本来源：缺回执的调用不能带着未知效果进入新请求。
            await tools.settle_abandoned(call)
        if abandoned:
            # 放弃结算已追加事实，重新读取；普通路径复用同一次扫描的待执行调用。
            history.update(await reader.committed_window_async(window))
            pending, _ = history.open_calls()
        await _settle_pending(
            reader, tools, pending, max_parallel_calls, capture_scope,
        )
        mark("tools.settled")
        # 工具结果都已提交、下一次请求之前：调用方在这里写入插话，下面的重读会带上它。
        if before_step is not None:
            await before_step()
        if pending or abandoned or reader.head() != head_before:
            snapshot = await reader.committed_window_async(window)
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
        failure = history.failures.failure()
        if failure is not None:
            raise ModelError(failure).exception()
        if history.generation_rejections:
            _check_generation_recovery(
                len(history.generation_rejections), history.generation_rejections[-1][1], max_output_tokens,
            )
        if max_steps > 0 and history.steps >= max_steps:
            # 门槛落在请求之前：本轮没有任何模型输出，未完成区段全部来自更早的轮次。
            # 只有「从未落过回执」的调用会被放弃结算，它们的真实效果未知，
            # 不能由引擎替用户作废（那会把整段已执行工作从模型历史和记忆里抹掉）。
            unfinished, _ = history.open_calls()
            if unfinished:
                raise StepLimit(
                    step_limit_failure_reason(max_steps, len(unfinished)),
                ).exception()
            if aborted is not None:
                return await aborted(step_limit_abandon_reason(max_steps))
            # 没有写控制能力的调用方只能得到明确失败，来源仍会记录 failure；
            # 重试不重置计数，与修复前一致；新输入才会。
            raise StepLimit(f"本来源未完成工作已达到 {max_steps} 个模型输出").exception()

        # 2. 重启和显式重试都用当前材料发新请求，不恢复旧请求身份。
        mark("preparation.begin")
        prepared = await materials(tuple(snapshot))
        mark("preparation.end")
        # 退避被叫醒时跳过下面的提交，回到循环顶部写入插话再发新请求。
        async with attempts, _complete(
            snapshot, prepared, source=writer.source, context=context, model=model,
            projection=projection, tools=tools, max_output_tokens=max_output_tokens, reduce=reduce, preview=preview,
            reminder_input_id=reminder_input_id,
            operation_id=operation_id, max_retry_wait=max_retry_wait,
            retry_wake=retry_wake, max_attempts=attempts.remaining,
        ) as (response, prepared, message_id, request):
            # 3. Keep rejection feedback in history before another model request.
            rejection = _response_rejection(response.finish_reason, request.max_output_tokens)
            # 没有工具调用的截断保留正文供用户阅读和模型续写；其余拒绝仍以提示替换，半截调用绝不落账。
            partial_text = (response.finish_reason == "length" and not response.tool_calls
                            and bool((response.content or "").strip()))
            if rejection is not None:
                response = replace(response, tool_calls=(), continuation=None, provider_metadata=None)
            decoded, metadata = await content.decode(
                (response.content or "") if partial_text else rejection or response.content or "",
                cast(tuple[Mapping[str, object], ...], prepared.get("references", ())),
            )
            parts: list[Part] = list(decoded)
            indices: list[int] = []
            actual_calls: list[ToolCall | ContentPart] = []
            calls = () if rejection is not None else response.tool_calls
            decoded_calls = tuple(tools.decode(call) for call in calls)
            exclusive_names = tuple(
                tools.name(decoded.binding_id)
                for decoded in decoded_calls
                if decoded.binding_id is not None and decoded.exclusive_batch
            )
            mixed_exclusive = len(decoded_calls) > 1 and bool(exclusive_names)
            for call, decoded_call in zip(calls, decoded_calls, strict=True):
                indices.append(len(parts))
                repeated = history.failures.rejection(call.name, call.arguments)
                if repeated is not None:
                    actual = ContentPart("model.tool_rejection", {
                        "name": call.name, "arguments": call.arguments, "error": repeated,
                    })
                elif mixed_exclusive:
                    actual = ContentPart("model.tool_rejection", {
                        "name": call.name, "arguments": call.arguments,
                        "error": f"此批次包含要求独占的工具 {', '.join(exclusive_names)}；整批未执行，请分别单独调用这些工具。",
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
                **({"partial_text": True} if partial_text else {}),
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
                    Output(tuple(parts), "continue" if indices or rejection is not None else "complete"),
                    metadata,
                )
            except _Superseded:
                raise asyncio.CancelledError from None
            mark("output.committed", request_id=response.call_record_id or "", parent_operation_id=message.message_id, counts={"seq": message.seq, "tool_calls": len(indices)})
            if rejection is not None:
                _check_generation_recovery(
                    len(history.generation_rejections) + 1, response.finish_reason, request.max_output_tokens,
                )
            elif not indices:
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

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol, cast
from uuid import uuid4

from agent.plugin_composition import (
    Context,
    RUNTIME_STARTED,
    RUNTIME_STARTING,
    RUNTIME_STOPPING,
)
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.restart import RESTART_GATE, RestartRejectedError
from plugins.delivery.api import FINAL_OUTPUT_DELIVERY, FinalOutputWaiter
from plugins.tools.api import BoundTool, CallSource, ContentPart, Result, durable_call_key
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION, Turn, TurnProjection
from session.log import Message, MessageCatalog, MessageReader
from session.message import CallRef, Output, ToolCall, ToolResult, freeze_json


api_version = 3
name = "agent_restart"
version = "1.0.0"
desc = "在当前最终回复写入并送达后请求 supervisor 重启"
inject = (
    TOOLS,
    BINDINGS,
    MESSAGE_CATALOG,
    TURN_PROJECTION,
    FINAL_OUTPUT_DELIVERY,
    RESTART_GATE,
)


@dataclass(frozen=True, slots=True)
class PendingRestart:
    call_ref: CallRef
    effect_key: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RestartRequest:
    request_id: str
    session_id: str
    source: str
    call_ref: CallRef


class RestartTool(BoundTool):
    """只把一次工具调用的参数固定到其已提交 ToolResult。"""

    @property
    def idempotent(self) -> bool:
        return False

    def __init__(self) -> None:
        self._prepared: PendingRestart | None = None

    async def prepare(
        self,
        arguments: Mapping[str, object],
        source: CallSource | None = None,
    ) -> Mapping[str, object]:
        if source is None:
            raise ValueError("agent_restart 必须引用当前 Turn 的 ToolCall")
        if set(arguments) != {"reason"}:
            raise ValueError("agent_restart 参数只能包含 reason")
        reason = arguments.get("reason")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 300:
            raise ValueError("reason 长度必须为 1..300")
        call_message = next(
            (message for message in source.messages if message.message_id == source.call_ref.message_id),
            None,
        )
        if call_message is None:
            raise ValueError("agent_restart CallRef 不在当前消息前缀")
        final_arguments = freeze_json({"reason": reason.strip()})
        if not isinstance(final_arguments, Mapping):
            raise TypeError("agent_restart 参数必须是对象")
        pending = PendingRestart(
            source.call_ref,
            durable_call_key(source.call_ref),
            cast(Mapping[str, object], final_arguments),
        )
        current = self._prepared
        if current is not None and current != pending:
            if current.call_ref == pending.call_ref:
                raise RestartRejectedError("同一 restart ToolCall 的 binding 或参数不一致")
            raise RestartRejectedError("同一工具 binding 不能准备多个 restart ToolCall")
        if current is None:
            self._prepared = pending
            current = pending
        return current.arguments

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        pending = self._prepared
        if pending is None:
            raise RestartRejectedError("agent_restart 当前 binding 没有可恢复的 prepare")
        if key != pending.effect_key:
            raise RestartRejectedError("agent_restart durable key 不属于当前 ToolCall")
        final_arguments = freeze_json(arguments)
        if not isinstance(final_arguments, Mapping) or final_arguments != pending.arguments:
            raise RestartRejectedError("agent_restart 参数不属于当前 ToolCall")
        return Result("success", (ContentPart("text", "已安排在本轮最终回复送达后重启。"),))

    async def query(self, key: str) -> Result | None:
        return None


class _ClosableHeads(Protocol):
    def __aiter__(self) -> AsyncIterator[Mapping[str, int]]: ...
    async def __anext__(self) -> Mapping[str, int]: ...
    async def aclose(self) -> None: ...


class RestartWatcher:
    """只在正式 Root 观察新 ToolResult，并拥有一次重启请求的等待生命周期。"""

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._gate = None
        self._baseline_task: asyncio.Task[tuple[_ClosableHeads, Mapping[str, int]]] | None = None
        self._stream: _ClosableHeads | None = None
        self._watcher: asyncio.Task[None] | None = None
        self._active: str | None = None

    def prepare(self, _event: object) -> None:
        """在正式接纳开放前订阅日志并固定旧消息 heads。"""
        self._gate = self._ctx.require(RESTART_GATE)
        catalog = self._ctx.require(MESSAGE_CATALOG)

        async def prime() -> tuple[_ClosableHeads, Mapping[str, int]]:
            stream = cast(_ClosableHeads, catalog.follow())
            try:
                heads = await stream.__anext__()
            except BaseException:
                await stream.aclose()
                raise
            return stream, heads

        self._baseline_task = asyncio.create_task(prime(), name="agent-restart-baseline")

    async def start(self, _event: object) -> None:
        baseline_task = self._baseline_task
        if baseline_task is None:
            raise RuntimeError("agent_restart watcher 缺少启动基线")
        self._baseline_task = None
        self._stream, baseline = await baseline_task
        catalog = self._ctx.require(MESSAGE_CATALOG)
        projection = self._ctx.require(TURN_PROJECTION)
        delivery = self._ctx.require(FINAL_OUTPUT_DELIVERY)
        self._watcher = await self._ctx.spawn(
            self._watch(self._stream, baseline, catalog, projection, delivery),
            name="agent-restart-watcher",
        )

    async def stop(self, _event: object) -> None:
        baseline_task = self._baseline_task
        self._baseline_task = None
        if baseline_task is not None and not baseline_task.done():
            _ = baseline_task.cancel()
            _ = await asyncio.gather(baseline_task, return_exceptions=True)
        watcher = self._watcher
        self._watcher = None
        if watcher is not None and not watcher.done():
            _ = watcher.cancel()
            _ = await asyncio.gather(watcher, return_exceptions=True)
        if self._active is not None and self._gate is not None:
            self._gate.abort(self._active)
            self._active = None
        stream = self._stream
        self._stream = None
        if stream is not None:
            await stream.aclose()

    async def _watch(
        self,
        stream: _ClosableHeads,
        baseline: Mapping[str, int],
        catalog: MessageCatalog,
        projection: TurnProjection,
        delivery: FinalOutputWaiter,
    ) -> None:
        cursors = dict(baseline)
        try:
            async for heads in stream:
                for session_id, head in sorted(heads.items()):
                    after = cursors.get(session_id, -1)
                    if head <= after:
                        continue
                    reader = catalog.reader(session_id)
                    messages = reader.read(after_seq=after, limit=100)
                    while messages:
                        for message in messages:
                            if self._active is not None:
                                return
                            if not isinstance(message.body, ToolResult) or message.body.outcome != "success":
                                continue
                            request = await self._request(reader, message)
                            if request is None:
                                continue
                            self._active = request.request_id
                            await self._wait_for_request(request, reader, projection, delivery)
                            self._active = None
                            return
                        after = messages[-1].seq
                        if after >= head:
                            break
                        messages = reader.read(after_seq=after, limit=100)
                    cursors[session_id] = head
        finally:
            if self._active is not None and self._gate is not None:
                self._gate.abort(self._active)
                self._active = None
            await stream.aclose()

    async def _request(self, reader: MessageReader, message: Message) -> RestartRequest | None:
        result = message.body
        if not isinstance(result, ToolResult):
            return None
        call_message = reader.get(result.call_ref.message_id)
        if call_message is None or not isinstance(call_message.body, Output):
            raise RestartRejectedError("agent_restart ToolResult 引用缺失")
        if result.call_ref.part_index >= len(call_message.body.parts):
            raise RestartRejectedError("agent_restart ToolResult CallRef 无效")
        call = call_message.body.parts[result.call_ref.part_index]
        if not isinstance(call, ToolCall):
            raise RestartRejectedError("agent_restart ToolResult 未指向 ToolCall")
        async with self._ctx.runtime_scope():
            bindings = self._ctx.require(BINDINGS)
            metadata = bindings.describe(call.binding_id, TOOLS)
            descriptor = cast(Mapping[str, object], metadata).get("tool")
            if not isinstance(descriptor, Mapping):
                raise ValueError("工具 binding 描述无效")
            descriptor = cast(Mapping[str, object], descriptor)
            if descriptor.get("name") != name or descriptor.get("owner") != self._ctx.runtime.plugin_id:
                return None
        if call_message.source != message.source:
            raise RestartRejectedError("agent_restart ToolResult 来源不一致")
        return RestartRequest(
            "restart_" + uuid4().hex,
            message.session_id,
            message.source,
            result.call_ref,
        )

    async def _wait_for_request(
        self,
        request: RestartRequest,
        reader: MessageReader,
        projection: TurnProjection,
        delivery: FinalOutputWaiter,
    ) -> None:
        gate = self._gate
        if gate is None:
            raise RuntimeError("agent_restart watcher 缺少 RestartGate")
        gate.prepare(request.request_id)
        try:
            async with asyncio.timeout(15.0):
                async for _message in reader.follow():
                    turn = _find_turn(reader, projection, request)
                    if turn is None or turn.status == "open":
                        continue
                    if turn.status != "complete" or turn.ending_message_id is None:
                        raise RestartRejectedError("restart ToolCall 所属 Turn 未正常完成")
                    await delivery.wait(reader, turn)
                    await gate.commit(request.request_id)
                    return
            raise RestartRejectedError("等待最终 Output 超时")
        except BaseException:
            gate.abort(request.request_id)
            raise


def _find_turn(reader: MessageReader, projection: TurnProjection, request: RestartRequest) -> Turn | None:
    """从现有 TurnProjection 找出精确 CallRef 所属 Turn。"""
    call_message = reader.get(request.call_ref.message_id)
    if call_message is None or call_message.source != request.source or not isinstance(call_message.body, Output):
        raise RestartRejectedError("restart ToolCall 引用缺失")
    if request.call_ref.part_index >= len(call_message.body.parts):
        raise RestartRejectedError("restart CallRef 位置无效")
    if not isinstance(call_message.body.parts[request.call_ref.part_index], ToolCall):
        raise RestartRejectedError("restart CallRef 未指向 ToolCall")
    for turn in projection.project(reader.snapshot(), request.source):
        if request.call_ref.message_id in turn.message_ids:
            return turn
    return None


async def apply(ctx: Context, config: object) -> None:
    gate = ctx.require(RESTART_GATE)
    if not gate.supervised:
        return
    watcher = RestartWatcher(ctx)

    @asynccontextmanager
    async def open_tool(_state: Mapping[str, object]) -> AsyncGenerator[BoundTool, None]:
        yield RestartTool()

    _ = await ctx.require(TOOLS).register(
        ctx,
        name="agent_restart",
        description="在当前最终回复写入并送达后安全重启 Agent。",
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "minLength": 1, "maxLength": 300},
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
        open=open_tool,
        risk="external-side-effect",
        always_on=False,
        preloadable=False,
        requires_search=True,
        search_hint="重启 reload restart 核心代码",
    )
    _ = await ctx.on(RUNTIME_STARTING, watcher.prepare)
    _ = await ctx.on(RUNTIME_STARTED, watcher.start)
    _ = await ctx.on(RUNTIME_STOPPING, watcher.stop)

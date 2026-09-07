from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from agent.plugin_composition import Context
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.restart import RESTART_GATE, RestartRejectedError
from plugins.delivery.api import FINAL_OUTPUT_DELIVERY
from plugins.tools.api import BoundTool, CallSource, ContentPart, Result, durable_call_key
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION, Turn, TurnProjection
from session.log import MessageReader
from session.message import Output, ToolCall, CallRef, freeze_json


api_version = 3
name = "agent_restart"
version = "1.0.0"
desc = "在当前最终回复写入并送达后请求 supervisor 重启"
inject = (TOOLS, MESSAGE_CATALOG, TURN_PROJECTION, FINAL_OUTPUT_DELIVERY, RESTART_GATE)


@dataclass(frozen=True, slots=True)
class PendingRestart:
    call_ref: CallRef
    request_id: str
    session_id: str
    source: str
    effect_key: str
    arguments: Mapping[str, object]


class RestartTool(BoundTool):
    @property
    def idempotent(self) -> bool:
        # pending 只在本 boot 内存中存在；跨 boot 没有可查询的 commit 回执。
        return False

    def __init__(self, runtime: "RestartRuntime") -> None:
        self._runtime = runtime
        self._prepared: PendingRestart | None = None

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
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
            "restart_" + uuid4().hex,
            call_message.session_id,
            call_message.source,
            durable_call_key(source.call_ref),
            final_arguments,
        )
        current = self._prepared
        if current is not None and (
            current.call_ref != pending.call_ref
            or current.effect_key != pending.effect_key
            or current.arguments != pending.arguments
        ):
            if current.call_ref == pending.call_ref:
                raise RestartRejectedError("同一 restart ToolCall 的 binding 或参数不一致")
            raise RestartRejectedError("同一工具 binding 不能准备多个 restart ToolCall")
        self._prepared = pending if current is None else current
        return self._prepared.arguments

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        pending = self._prepared
        if pending is None:
            raise RestartRejectedError("agent_restart 当前 boot 没有可恢复的 prepare")
        self._runtime.prepare(pending)
        pending = self._runtime.match(key, arguments)
        await self._runtime.start(pending)
        return Result("success", (ContentPart("text", "已安排在本轮最终回复送达后重启。"),))

    async def query(self, key: str) -> Result | None:
        return None


class RestartRuntime:
    """持有一个 boot 内 pending restart；不跨 boot 恢复业务请求。"""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.pending: PendingRestart | None = None
        self._task: asyncio.Task[None] | None = None

    def prepare(self, pending: PendingRestart) -> None:
        current = self.pending
        if current is None:
            self.pending = pending
            return
        if current.call_ref != pending.call_ref:
            raise RestartRejectedError("已有另一个 restart ToolCall")
        if current.effect_key != pending.effect_key or current.arguments != pending.arguments:
            raise RestartRejectedError("同一 restart ToolCall 的 binding 或参数不一致")

    def match(self, key: str, arguments: Mapping[str, object]) -> PendingRestart:
        """只把 invoke 绑定到 prepare 已接纳的同一调用和最终参数。"""
        pending = self.pending
        if pending is None:
            raise RestartRejectedError("agent_restart 没有当前 boot 的待处理 ToolCall")
        if key != pending.effect_key:
            raise RestartRejectedError("agent_restart durable key 不属于当前 ToolCall")
        final_arguments = freeze_json(arguments)
        if not isinstance(final_arguments, Mapping) or final_arguments != pending.arguments:
            raise RestartRejectedError("agent_restart 参数不属于当前 ToolCall")
        return pending

    async def start(self, pending: PendingRestart) -> None:
        if self.pending is not pending:
            raise RestartRejectedError("agent_restart pending 已失效")
        if self._task is not None and not self._task.done():
            return
        gate = self.ctx.require(RESTART_GATE)
        try:
            gate.prepare(pending.request_id)
            self._task = await self.ctx.spawn(
                self._wait_for_final_output(pending),
                name="agent-restart-final-output",
            )
        except BaseException:
            gate.abort(pending.request_id)
            self._clear_pending(pending)
            raise

    async def _wait_for_final_output(self, pending: PendingRestart) -> None:
        gate = self.ctx.require(RESTART_GATE)
        try:
            catalog = self.ctx.require(MESSAGE_CATALOG)
            reader = catalog.reader(pending.session_id)
            projection = self.ctx.require(TURN_PROJECTION)
            delivery = self.ctx.require(FINAL_OUTPUT_DELIVERY)
            async with asyncio.timeout(15.0):
                async for _message in reader.follow():
                    turn = _find_turn(reader, projection, pending)
                    if turn is None:
                        continue
                    if turn.status == "open":
                        continue
                    if turn.status != "complete" or turn.ending_message_id is None:
                        raise RestartRejectedError("restart ToolCall 所属 Turn 未正常完成")
                    await delivery.wait(reader, turn)
                    await gate.commit(pending.request_id)
                    return
            raise RestartRejectedError("等待最终 Output 超时")
        except BaseException:
            gate.abort(pending.request_id)
            raise
        finally:
            self._clear_pending(pending)

    def _clear_pending(self, pending: PendingRestart) -> None:
        """只清理仍属于本 owner 的 pending，避免旧任务抹掉新调用。"""
        if self.pending is pending:
            self.pending = None


def _find_turn(reader: MessageReader, projection: TurnProjection, pending: PendingRestart) -> Turn | None:
    """从现有 TurnProjection 找出精确 CallRef 所属 Turn。"""
    call_message = reader.get(pending.call_ref.message_id)
    if call_message is None or call_message.source != pending.source or not isinstance(call_message.body, Output):
        raise RestartRejectedError("restart ToolCall 引用缺失")
    if pending.call_ref.part_index >= len(call_message.body.parts):
        raise RestartRejectedError("restart ToolCall 位置无效")
    if not isinstance(call_message.body.parts[pending.call_ref.part_index], ToolCall):
        raise RestartRejectedError("restart CallRef 未指向 ToolCall")
    for turn in projection.project(reader.snapshot(), pending.source):
        if pending.call_ref.message_id in turn.message_ids:
            return turn
    return None


async def apply(ctx: Context, config: object) -> None:
    runtime = RestartRuntime(ctx)

    @asynccontextmanager
    async def open_tool(_state: Mapping[str, object]) -> AsyncIterator[BoundTool]:
        yield RestartTool(runtime)

    await ctx.require(TOOLS).register(
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

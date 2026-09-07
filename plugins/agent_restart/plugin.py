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
from plugins.tools.api import BoundTool, CallSource, ContentPart, Result
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION, Turn, TurnProjection
from session.log import MessageReader
from session.message import Output, ToolCall, CallRef


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


class RestartTool(BoundTool):
    @property
    def idempotent(self) -> bool:
        return True

    def __init__(self, runtime: "RestartRuntime") -> None:
        self._runtime = runtime

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        if source is None:
            raise ValueError("agent_restart 必须引用当前 Turn 的 ToolCall")
        reason = arguments.get("reason")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 300:
            raise ValueError("reason 长度必须为 1..300")
        call_message = next(
            (message for message in source.messages if message.message_id == source.call_ref.message_id),
            None,
        )
        if call_message is None:
            raise ValueError("agent_restart CallRef 不在当前消息前缀")
        pending = PendingRestart(
            source.call_ref,
            "restart_" + uuid4().hex,
            call_message.session_id,
            call_message.source,
        )
        self._runtime.prepare(pending)
        return {"reason": reason.strip()}

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        pending = self._runtime.pending
        if pending is None:
            raise RestartRejectedError("agent_restart 没有待处理的 ToolCall")
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
        if self.pending is not None and self.pending.call_ref != pending.call_ref:
            raise RestartRejectedError("已有另一个 restart ToolCall")
        self.pending = pending

    async def start(self, pending: PendingRestart) -> None:
        if self._task is not None and not self._task.done():
            return
        gate = self.ctx.require(RESTART_GATE)
        gate.prepare(pending.request_id)
        self._task = await self.ctx.spawn(
            self._wait_for_final_output(pending),
            name="agent-restart-final-output",
        )

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

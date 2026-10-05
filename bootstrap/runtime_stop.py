"""宿主停止准备：正常回合、送达与工作排空分别核对。"""

from __future__ import annotations

import asyncio
from pydantic import Field

from agent.control.protocol.models import StrictModel
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugin_contracts import CallRef, Input, Output, ToolCall, ToolResult
from agent.plugin_contracts.delivery import FINAL_OUTPUT_DELIVERY
from agent.plugin_contracts.turns import TURN_PROJECTION
from bootstrap.tools import CoreRuntime


class StopParams(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    boot_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=512)
    call_message_id: str = Field(min_length=1, max_length=256)
    call_part_index: int = Field(ge=0)
    timeout_s: int = Field(default=600, ge=1, le=3600)
    arm_only: bool = False


class CancelStopParams(StopParams):
    pass


async def prepare_stop(core: CoreRuntime, params: StopParams) -> dict[str, object]:
    """固定原 boot/Root，等待真实结束；失败时恢复尚未开始关闭的准入。"""
    gate = core.restart_gate
    if gate is None or gate.boot_id != params.boot_id:
        raise ValueError("停止请求不属于当前 boot")
    root = core.plugin_manager.live_root
    if root is None:
        raise RuntimeError("停止请求没有 live Root")
    ref = CallRef(params.call_message_id, params.call_part_index)
    reader = root.service_value(MESSAGE_CATALOG).reader(params.session_id)
    call = await reader.read_async(lambda snapshot: snapshot.get(ref.message_id))
    if (
        call is None
        or not isinstance(call.body, Output)
        or ref.part_index >= len(call.body.parts)
    ):
        raise ValueError("停止请求没有原 ToolCall")
    if not isinstance(call.body.parts[ref.part_index], ToolCall):
        raise ValueError("停止请求引用的不是 ToolCall")
    claim = core.control_frames.claim_for(params.session_id, ref)
    if params.arm_only:
        if call.source == "programmatic" and claim is None:
            messages = await reader.read_async(lambda snapshot: snapshot.snapshot())
            origin = next(
                message
                for message in reversed(messages[: messages.index(call)])
                if message.source == call.source and isinstance(message.body, Input)
            )
            claim = core.control_frames.arm_claim(
                params.session_id, origin.message_id, ref
            )
            asyncio.get_running_loop().call_later(params.timeout_s, claim.abort)
        return {"bootId": gate.boot_id, "state": "armed"}
    projection = root.service_value(TURN_PROJECTION)
    delivery_context, delivery = root._service_provider(FINAL_OUTPUT_DELIVERY)
    # 同一停止只允许一个 waiter；重放接单由宿主任务记录处理。
    gate.check_open()
    gate.prepare_stop(params.request_id)
    try:
        async with asyncio.timeout(params.timeout_s), delivery_context.runtime_scope():
            # 1. follow 包含当前快照，不需要下一条消息来唤醒已完成的回合。
            async for _ in reader.follow():
                messages = await reader.read_async(lambda snapshot: snapshot.snapshot())
                turn = next(
                    (
                        item
                        for item in projection.project(messages, call.source)
                        if call.message_id in item.message_ids
                    ),
                    None,
                )
                if turn is None or turn.status == "open":
                    continue
                if turn.status != "complete" or turn.ending_message_id is None:
                    raise RuntimeError("发起部署的回合未正常完成")
                result = next(
                    (
                        message.body
                        for message in messages
                        if isinstance(message.body, ToolResult)
                        and message.body.call_ref == ref
                    ),
                    None,
                )
                if result is None or result.outcome != "success":
                    raise RuntimeError("提交部署的 Shell 调用未成功结算")
                # 2. 完成正文不代替原渠道的最终送达证据。
                if call.source == "programmatic":
                    if claim is None:
                        raise RuntimeError("停止请求缺少提交时保留的最终送达凭据")
                    await claim.wait_output()
                    if claim.ending_message_id != turn.ending_message_id:
                        raise RuntimeError("最终送达凭据与原回合不一致")
                else:
                    await delivery.wait(reader, turn)
                # 3. 当前控制请求不持有业务许可，不会等待自己。
                await gate.wait_drained(params.timeout_s)
                await core.plugin_manager.wait_idle()
                if core.plugin_manager.live_root is not root:
                    raise RuntimeError("停止准备期间 Root 已被替换")
                gate.check_stop(params.request_id)
                return {
                    "bootId": gate.boot_id,
                    "rootIdentity": root.generation_id,
                    "endingMessageId": turn.ending_message_id,
                    "state": "drained",
                }
            raise RuntimeError("消息读取在停止准备期间结束")
    except BaseException:
        gate.abort(params.request_id)
        raise

    finally:
        if claim is not None:
            claim.consume()

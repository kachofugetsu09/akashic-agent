from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Hashable, Mapping
from dataclasses import replace
from typing import cast

from agent.plugin_composition.tasks import TaskAdmission, TaskSlot
from agent.plugin_contracts.tool_api import Denied, Result, durable_call_key
from plugins.tools.api import MessageReply
from plugins.tools.execution import _fingerprint, finish
from agent.plugin_composition.messages import MessageCatalog, MessageReader, OwnerStore
from agent.plugin_contracts import CallRef, ContentPart, Control, Message, Output, ToolCall, ToolResult


class LegacyReplyIdentityUnavailable(ValueError):
    """旧回执使用自定义结果身份，但没有保存可恢复的明文身份。"""


async def abandon_call(
    state: OwnerStore, tasks: TaskAdmission, reply: MessageReply, *, task_key: Hashable,
) -> Result:
    """Tools 先提交终态再协作取消；真实效果与清理仍由原 Task 持有。"""
    state.check_access(reply.reader, reply.writer)
    if not reply.abandoned():
        raise ValueError("工具没有已接纳的放弃控制")
    key = durable_call_key(reply.call_ref)

    def settle(slot: TaskSlot) -> Result:
        record = state.read(key)
        if record is None:
            for message in reply.reader.snapshot():
                body = message.body
                if isinstance(body, ToolResult) and body.call_ref == reply.call_ref:
                    if slot.current is not None:
                        slot.current.cancel()
                    return Result(body.outcome, body.parts)
        if record is not None and "reply_id" in record.value:
            identity = record.value["reply_id"]
            if not isinstance(identity, str):
                raise ValueError("消息工具回执缺少结果身份")
            target = replace(reply, message_id=identity)
        else:
            target = reply
            if record is not None and record.value["phase"] == "done":
                result = record.value.get("result")
                if isinstance(result, Mapping) and isinstance(result.get("message_id"), str):
                    target = replace(reply, message_id=cast(str, result["message_id"]))
        call = target.request()
        fingerprint = _fingerprint(call.binding_id, call.arguments, target)
        if record is not None:
            if record.value["request"] != fingerprint:
                if "reply_id" not in record.value:
                    raise LegacyReplyIdentityUnavailable(
                        f"旧工具回执缺少可恢复的结果身份 call={reply.call_ref.message_id}:{reply.call_ref.part_index}"
                    )
                raise ValueError("放弃的工具回执与原请求不一致")
            if record.value["phase"] == "done":
                if slot.current is not None:
                    slot.current.cancel()
                return target.read(record.value["result"])
            if record.value["phase"] not in {"requested", "prepared", "started"}:
                raise ValueError("工具回执阶段无效")
        started = record is not None and record.value["phase"] == "started"
        result = finish(
            state, key, record,
            Result("interrupted" if started else "denied", (ContentPart(
                "text", "用户已放弃此工作；调用已中断，外部效果可能已经发生，不能据此重跑。"
                if started else "用户已放弃此工作，工具未启动。",
            ),)), target,
            initial={"version": 1, "request": fingerprint, "binding": call.binding_id,
                     "reply_id": target.message_id, "arguments": call.arguments},
        )
        if slot.current is not None:
            slot.current.cancel()
        return result

    return await tasks.admit((task_key, key), settle)


async def follow_abandon(
    catalog: MessageCatalog, state: OwnerStore, tasks: TaskAdmission,
    reply: Callable[[MessageReader, str, CallRef], Awaitable[MessageReply]], *, task_key: Hashable,
    report_incident: Callable[[str, str], object],
) -> None:
    """只消费持久 abandon；启动追赶也结算未开始或进程中断后的调用。"""
    seen: dict[str, int] = {}
    async for heads in catalog.follow():
        for session_id, head in heads.items():
            previous = seen.get(session_id, -1)
            if head == previous:
                continue
            reader = catalog.reader(session_id)
            changed = await asyncio.to_thread(reader.snapshot, after_seq=previous, through_seq=head)
            controls = [message for message in changed
                        if isinstance(message.body, Control) and message.body.action == "abandon"]
            messages = await asyncio.to_thread(reader.snapshot, through_seq=head) if controls else ()
            for control in controls:
                for ref in abandoned_calls(messages, control):
                    target = await reply(reader, control.source, ref)
                    try:
                        try:
                            _ = await abandon_call(state, tasks, target, task_key=task_key)
                        except LegacyReplyIdentityUnavailable as error:
                            _ = report_incident("legacy_tool_reply_identity", str(error))
                    finally:
                        target.writer.expire()
            seen[session_id] = head


def abandoned_calls(messages: tuple[Message, ...], control: Message) -> tuple[CallRef, ...]:
    """只选同来源的被放弃调用；已有结果也可能仍有资源等待清理。"""
    body = cast(Control, control.body)
    calls: dict[CallRef, int] = {}
    for message in messages:
        if message.source != control.source:
            continue
        if isinstance(message.body, Output) and message.seq <= body.through_seq:
            if message.body.finish != "continue":
                calls.clear()
            else:
                calls.update((CallRef(message.message_id, index), message.seq)
                             for index, part in enumerate(message.body.parts) if isinstance(part, ToolCall))
    return tuple(calls)


def reject_start() -> None:
    raise Denied("放弃消费者没有启动工具的权限")

from __future__ import annotations

import asyncio
from collections.abc import Callable, Hashable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from typing import cast

from agent.plugin_composition.tasks import TaskAdmission, TaskSlot
from core.common.file_io import run_file_io
from .api import Denied, MessageReply, Result, durable_call_key
from .execution import _fingerprint, finish
from plugins.ledger.contract import MessageCatalog, MessageReader, OwnerStore, OwnerTransaction
from plugins.ledger.contract import CallRef, ContentPart, Control, Message, Output, ToolCall, ToolResult


# 结果只可能在调用消息之后；流式扫描命中即停，不为每个被放弃的调用解码整段会话。
def _existing_result(reader: MessageReader, call_ref: CallRef) -> ToolResult | None:
    call = reader.get(call_ref.message_id)

    def first(rows: Iterable[Message]) -> ToolResult | None:
        for message in rows:
            body = message.body
            if isinstance(body, ToolResult) and body.call_ref == call_ref:
                return body
        return None

    return reader.scan(first, after_seq=-1 if call is None else call.seq)


class LegacyReplyIdentityUnavailable(ValueError):
    """旧回执使用自定义结果身份，但没有保存可恢复的明文身份。"""


async def abandon_call(
    state: OwnerStore, tasks: TaskAdmission, reply: MessageReply, *, task_key: Hashable,
) -> Result:
    """Tools 先提交终态再协作取消；真实效果与清理仍由原 Task 持有。"""
    state.check_access(reply.reader, reply.writer)
    # 放弃判定会扫调用之后的整段前缀，与下方回执读取一样放进 worker。
    if not await run_file_io(reply.abandoned):
        raise ValueError("工具没有已接纳的放弃控制")
    key = durable_call_key(reply.call_ref)

    async def settle(slot: TaskSlot) -> Result:
        # 回执与已有结果的读取放进 worker：启动扫描时机械盘冷读曾让事件循环停顿 2.2s。
        record = await run_file_io(lambda: state.read(key))
        if record is None:
            existing = await run_file_io(lambda: _existing_result(reply.reader, reply.call_ref))
            if existing is not None:
                if slot.current is not None:
                    slot.current.cancel()
                return Result(existing.outcome, existing.parts)
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
        call = await run_file_io(target.request)
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
                done = record.value["result"]
                return await run_file_io(lambda: target.read(done))
            if record.value["phase"] not in {"requested", "prepared", "started"}:
                raise ValueError("工具回执阶段无效")
        # 无阶段回执不能证明未执行；重启后内存 Task 已丢失，仍须保留效果不确定性。
        may_have_started = record is None or record.value["phase"] == "started"
        def cancel_after_commit(_result: Result) -> None:
            if slot.current is not None:
                slot.current.cancel()

        result = await finish(
            state, key, record,
            Result("interrupted" if may_have_started else "denied", (ContentPart(
                "text", "用户已放弃此工作；工具可能已部分执行。请先检查当前状态，再决定是否重试。"
                if may_have_started else "用户已放弃此工作，工具未启动。",
            ),)), target,
            initial={"version": 1, "request": fingerprint, "binding": call.binding_id,
                     "reply_id": target.message_id, "arguments": call.arguments},
            on_commit=cancel_after_commit,
        )
        return result

    return await tasks.admit_async((task_key, key), settle)


# 区间内任一来源出现 abandon Control；按来源走 kind 索引，只解码 Control。
# 只解码 abandon 控制及其所需窗口：被放弃的调用只来自同来源、through_seq 前最后一次结束输出之后，
# 不必为一条历史控制解码整段会话（启动时 seen 为空，大会话会被整段解码两次）。
def _abandon_targets(
    reader: MessageReader, after_seq: int, through_seq: int,
) -> list[tuple[Message, tuple[CallRef, ...]]]:
    targets: list[tuple[Message, tuple[CallRef, ...]]] = []
    for source in reader.source_names():
        seqs = reader.scan_controls(
            lambda rows: [seq for seq, body in rows if body.action == "abandon"],
            source=source, after_seq=after_seq, through_seq=through_seq,
        )
        for seq in seqs:
            control = reader.read(after_seq=seq - 1, through_seq=seq, limit=1)[0]
            bound = cast(Control, control.body).through_seq
            finished = reader.latest_finished_output_seq(source, after_seq=-1, through_seq=bound)
            window = reader.scan(
                tuple, after_seq=-1 if finished is None else finished, through_seq=bound, source=source,
            )
            targets.append((control, abandoned_calls(window, control)))
    targets.sort(key=lambda target: target[0].seq)
    return targets


async def follow_abandon(
    catalog: MessageCatalog, state: OwnerStore, tasks: TaskAdmission,
    reply: Callable[[MessageReader, str, CallRef], AbstractAsyncContextManager[MessageReply]], *, task_key: Hashable,
    report_incident: Callable[[str, str], object],
    retry_delay: float = 5.0,
) -> None:
    """只消费持久 abandon；坏 item 只影响本 Session 的保守 cursor，其余继续推进。"""
    seen: dict[str, int] = {}
    failed: set[str] = set()
    stream = catalog.follow(wake_on=Control).__aiter__()
    pending_next: asyncio.Task[Mapping[str, int]] | None = None
    try:
        while True:
            if pending_next is None:
                pending_next = asyncio.ensure_future(stream.__anext__())
            try:
                if failed:
                    # 失败区间有界退避重扫；cursor 未推进，幂等回执保证重读安全。
                    try:
                        heads = await asyncio.wait_for(asyncio.shield(pending_next), retry_delay)
                    except TimeoutError:
                        heads = None
                else:
                    heads = await pending_next
            except StopAsyncIteration:
                return
            if heads is not None:
                pending_next = None
            if heads is None:
                sessions = set(failed)
            else:
                sessions = {
                    session_id
                    for session_id, head in heads.items()
                    if seen.get(session_id, -1) != head
                } | failed
            for session_id in sorted(sessions):
                reader = catalog.reader(session_id)
                try:
                    head = await asyncio.to_thread(reader.head)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    _ = report_incident("tool_abandon_scan", str(error))
                    failed.add(session_id)
                    continue
                previous = seen.get(session_id, -1)
                if head == previous and session_id not in failed:
                    continue
                ok = True
                try:
                    # 启动时 seen 为空，窗口是整段历史；走 Control 索引找 abandon，只解码各自的尾部窗口。
                    targets = await asyncio.to_thread(_abandon_targets, reader, previous, head)
                    for control, refs in targets:
                        for ref in refs:
                            try:
                                async with reply(reader, control.source, ref) as target:
                                    try:
                                        _ = await abandon_call(state, tasks, target, task_key=task_key)
                                    except LegacyReplyIdentityUnavailable as error:
                                        _ = report_incident("legacy_tool_reply_identity", str(error))
                            except asyncio.CancelledError:
                                raise
                            except Exception as error:
                                _ = report_incident("tool_abandon_item", str(error))
                                ok = False
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    _ = report_incident("tool_abandon_scan", str(error))
                    ok = False
                if ok:
                    seen[session_id] = head
                    failed.discard(session_id)
                else:
                    failed.add(session_id)
    finally:
        await stream.aclose()


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


def reject_start(_transaction: OwnerTransaction | None) -> None:
    raise Denied("放弃消费者没有启动工具的权限")

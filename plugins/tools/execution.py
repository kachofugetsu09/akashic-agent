from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Hashable, Mapping
from typing import cast

from agent.plugin_composition.tasks import Task, TaskAdmission, TaskSlot
from session.log import (
    OwnerRecord,
    OwnerStore,
    OwnerTransaction,
)
from session.message import (
    ContentPart,
    ToolResult,
    freeze_json,
)
from session.message_codec import json_value

from plugins.tools.api import (
    Authorize, Denied, InvalidArguments, MessageReply, OpenTool, Outcome, Result,
    durable_call_key,
)


class ToolExecution:
    """拥有单次工具效果回执；只使用已获授的状态、Task 和实际绑定。"""

    def __init__(
        self,
        state: OwnerStore,
        tasks: TaskAdmission,
        open_tool: OpenTool,
        authorize: Authorize,
        *,
        task_key: Hashable,
    ):
        self._state = state
        self._tasks = tasks
        self._open_tool = open_tool
        self._authorize = authorize
        self._task_key = task_key

    async def execute(
        self, key: str, binding_id: str, arguments: Mapping[str, object]
    ) -> Result:
        """独立程序使用自身持久 key，无需创建 Session 或伪造工具调用消息。"""
        if not isinstance(key, str) or not key:
            raise ValueError("工具调用必须有稳定 key")
        return await self._execute("program:" + key, binding_id, arguments, None)

    async def execute_call(self, reply: MessageReply) -> Result:
        """同一已提交调用只有一个效果身份，与等待者及结果展示位置无关。"""
        self._state.check_access(reply.reader, reply.writer)
        call = reply.request()
        key = durable_call_key(reply.call_ref)
        return await self._execute(key, call.binding_id, call.arguments, reply)

    async def deny_call(self, reply: MessageReply, reason: str) -> Result:
        """结算不再获准启动的调用；已有执行先排空，崩溃后的 start 不能伪称未发生。"""
        self._state.check_access(reply.reader, reply.writer)
        call = reply.request()
        key = durable_call_key(reply.call_ref)
        fingerprint = _fingerprint(call.binding_id, call.arguments, reply)

        async def deny(task: Task) -> Result:
            record = self._record(key, fingerprint)
            if record is not None:
                if record.value["phase"] == "done":
                    return reply.read(record.value["result"])
            reply.check(self._state)
            if record is None:
                record = self._save(key, None, {
                    "version": 1, "request": fingerprint, "binding": call.binding_id,
                    "phase": "prepared", "arguments": call.arguments,
                })
            outcome: Outcome = "unknown" if record.value["phase"] == "started" else "denied"
            return self._finish(key, record, Result(outcome, (ContentPart("text", reason),)), reply)

        def admit(slot: TaskSlot) -> Task:
            return slot.current if slot.current is not None else slot.start(deny)

        task = await self._tasks.admit((self._task_key, key), admit)
        # 已经跨过 start 的 owner 不因放弃而被强杀，真实结果仍由原调用结算。
        result = cast(Result, await task.join())
        if self._record(key, fingerprint) is None:
            raise RuntimeError("工具结算缺少回执")
        return result

    async def _execute(
        self,
        key: str,
        binding_id: str,
        arguments: Mapping[str, object],
        reply: MessageReply | None,
    ) -> Result:
        if not isinstance(binding_id, str) or not binding_id:
            raise ValueError("工具调用必须固定 binding")
        if not isinstance(arguments, Mapping):
            raise TypeError("工具参数必须是对象")
        arguments = cast(Mapping[str, object], freeze_json(arguments))
        fingerprint = _fingerprint(binding_id, arguments, reply)

        async def run(task: Task) -> Result:
            return await self._run(task, key, binding_id, arguments, fingerprint, reply)

        # 1. 不同插件 scope 共用获授的 Task key，热更不能把活调用当成崩溃。
        def admit(slot: TaskSlot) -> tuple[Task, bool]:
            current = slot.current
            return (current, False) if current is not None else (slot.start(run), True)

        task, owned = await self._tasks.admit((self._task_key, key), admit)
        try:
            result = cast(Result, await task.join())
        except asyncio.CancelledError:
            if owned:
                task.cancel()
                await _drain(task)
            raise
        record = self._state.read(key)
        if record is None or record.value["request"] != fingerprint:
            raise ValueError("同一工具 key 的 binding 或参数不一致")
        return result

    async def _run(
        self,
        task: Task,
        key: str,
        binding_id: str,
        arguments: Mapping[str, object],
        fingerprint: str,
        reply: MessageReply | None,
    ) -> Result:
        """恢复先查回执；最终授权后先落盘 start，再进入真实工具。"""
        record = self._record(key, fingerprint)
        if record is not None:
            if record.value["phase"] == "done":
                return (
                    _read_result(record.value["result"])
                    if reply is None
                    else reply.read(record.value["result"])
                )

        try:
            if reply is not None:
                reply.check(self._state)
            async with self._open_tool(binding_id) as tool:
                # 2. prepare 的最终参数只固定一次，恢复不重新随机化或改写。
                if record is None:
                    source = None if reply is None else reply.source()
                    try:
                        final = freeze_json(await tool.prepare(arguments, source))
                    except InvalidArguments as error:
                        return self._finish(
                            key, None, Result("error", (ContentPart("text", str(error)),)), reply,
                            initial={"version": 1, "request": fingerprint, "binding": binding_id},
                        )
                    if not isinstance(final, Mapping):
                        raise TypeError("工具 prepare 必须返回参数对象")
                    value: dict[str, object] = {
                        "version": 1,
                        "request": fingerprint,
                        "binding": binding_id,
                        "phase": "prepared",
                        "arguments": final,
                    }
                    record = self._save(key, None, value)
                final_arguments = cast(Mapping[str, object], record.value["arguments"])
                started = record.value["phase"] == "started"
                # 3. 已跨过 start 的调用先 query；只有同 key 幂等协议才允许重发。
                if started:
                    found = await tool.query(key)
                    if found is not None:
                        return self._finish(key, record, found, reply)
                    if not tool.idempotent:
                        return self._finish(
                            key,
                            record,
                            Result(
                                "unknown", (ContentPart("text", "原工具效果无法确定"),)
                            ),
                            reply,
                        )
                # 4. 只为即将发生的调用授权；撤权不能抹掉可查询的历史结果。
                try:
                    permission = await self._authorize(binding_id, final_arguments)
                    if reply is not None:
                        reply.check_start()
                except Denied as error:
                    return self._finish(
                        key,
                        record,
                        Result(
                            "unknown" if started else "denied",
                            (ContentPart("text", str(error)),),
                        ),
                        reply,
                    )
                if not task.active:
                    raise asyncio.CancelledError
                if reply is not None:
                    reply.check(self._state)
                record = self._save(
                    key,
                    record,
                    {**record.value, "phase": "started", "permission": permission},
                )
                try:
                    result = await tool.invoke(key, final_arguments)
                except BaseException as failure:
                    # start intent 已耐久；内部异常或取消都不能证明远端没有效果。
                    try:
                        _ = self._finish(
                            key,
                            record,
                            Result(
                                "unknown",
                                (ContentPart("text", "工具调用未取得可确认结果"),),
                            ),
                            reply,
                        )
                    except Exception as record_failure:
                        raise failure from record_failure
                    raise
                return self._finish(key, record, result, reply)
        except asyncio.CancelledError as failure:
            # 恢复期间取消也终结原 started intent，不能稍后借重试重新发起效果。
            try:
                current = self._state.read(key)
                if current is not None and current.value["phase"] == "started":
                    _ = self._finish(
                        key,
                        current,
                        Result(
                            "unknown", (ContentPart("text", "原工具调用在恢复时取消"),)
                        ),
                        reply,
                    )
            except Exception as record_failure:
                raise failure from record_failure
            raise

    def _record(self, key: str, fingerprint: str) -> OwnerRecord | None:
        """在工具回执读取边界验证格式和请求身份。"""
        record = self._state.read(key)
        if record is not None:
            if record.value["version"] != 1 or record.value["phase"] not in {"prepared", "started", "done"}:
                raise ValueError("工具回执版本或阶段无效")
            if record.value["request"] != fingerprint:
                raise ValueError("同一工具 key 的 binding 或参数不一致")
        return record

    def _save(
        self, key: str, previous: OwnerRecord | None, value: Mapping[str, object]
    ) -> OwnerRecord:
        return self._state.transact(
            lambda transaction: transaction.save(
                key,
                value,
                expected_version=None if previous is None else previous.version,
            )
        )

    def _finish(
        self,
        key: str,
        record: OwnerRecord | None,
        result: Result,
        reply: MessageReply | None,
        *, initial: Mapping[str, object] | None = None,
    ) -> Result:
        """结果正文与 receipt 指针同事务提交；独立调用直接保存结果。"""

        value = initial if record is None else record.value
        if value is None:
            raise RuntimeError("工具结果缺少原请求身份")

        def commit(transaction: OwnerTransaction) -> None:
            saved: Mapping[str, object]
            if reply is None:
                saved = _result_value(result)
            else:
                message = transaction.append(
                    reply.writer,
                    reply.message_id,
                    ToolResult(reply.call_ref, result.outcome, result.parts),
                )
                saved = {"message_id": message.message_id, "seq": message.seq}
            _ = transaction.save(
                key,
                {**value, "phase": "done", "result": saved},
                expected_version=None if record is None else record.version,
            )

        self._state.transact(commit)
        return result


async def _drain(task: Task) -> None:
    """取消后继续等待已有 owner 结算，重复取消不再次截断清理。"""
    while not task.done:
        try:
            _ = await task.join()
        except asyncio.CancelledError:
            continue


def _fingerprint(
    binding_id: str, arguments: Mapping[str, object], reply: MessageReply | None
) -> str:
    value = json.dumps(
        [
            binding_id,
            json_value(arguments),
            (
                None
                if reply is None
                else [
                    reply.message_id,
                    reply.call_ref.message_id,
                    reply.call_ref.part_index,
                ]
            ),
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _result_value(result: Result) -> Mapping[str, object]:
    return {
        "outcome": result.outcome,
        "parts": tuple(
            {"kind": part.kind, "value": part.value} for part in result.parts
        ),
    }


def _read_result(value: object) -> Result:
    """只从 owner 存储边界重建已记录的工具结果。"""
    if not isinstance(value, Mapping):
        raise ValueError("工具结果回执损坏")
    data = cast(Mapping[str, object], value)
    if set(data) != {"outcome", "parts"}:
        raise ValueError("工具结果回执损坏")
    parts = data["parts"]
    if not isinstance(parts, tuple):
        raise ValueError("工具结果内容损坏")
    content: list[ContentPart] = []
    for part in cast(tuple[object, ...], parts):
        if not isinstance(part, Mapping):
            raise ValueError("工具结果内容块损坏")
        item = cast(Mapping[str, object], part)
        if set(item) != {"kind", "value"}:
            raise ValueError("工具结果内容块损坏")
        content.append(ContentPart(cast(str, item["kind"]), item["value"]))
    return Result(cast(Outcome, data["outcome"]), tuple(content))

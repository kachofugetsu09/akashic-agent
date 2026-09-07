from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Hashable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from agent.plugin_composition.tasks import Task, Tasks, TaskSlot
from session.log import (
    MessageReader,
    MessageWriter,
    OwnerRecord,
    OwnerStore,
    OwnerTransaction,
)
from session.message import (
    CallRef,
    ContentPart,
    Output,
    ToolCall,
    ToolResult,
    freeze_json,
)
from session.message_codec import json_value

Outcome = Literal["success", "denied", "error", "unknown"]


@dataclass(frozen=True, slots=True)
class Result:
    outcome: Outcome
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "denied", "error", "unknown"}:
            raise ValueError("工具结果状态无效")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("工具结果必须是内容块")
        object.__setattr__(self, "parts", parts)


@dataclass(frozen=True, slots=True)
class MessageReply:
    """已获授的调用结果写入位置；独立程序调用不需要它。"""

    message_id: str
    call_ref: CallRef
    reader: MessageReader
    writer: MessageWriter

    def __post_init__(self) -> None:
        if not isinstance(self.message_id, str) or not self.message_id:
            raise ValueError("工具结果 message_id 不能为空")
        if not isinstance(self.call_ref, CallRef):
            raise TypeError("工具结果必须引用 CallRef")

    def request(self) -> ToolCall:
        """直接读取已提交请求，调用者不能另外指定 binding、参数或执行 key。"""
        if self.reader.session_id != self.writer.session_id:
            raise ValueError("结果 reader 与 writer 的 Session 不一致")
        message = self.reader.get(self.call_ref.message_id)
        if message is None or not isinstance(message.body, Output):
            raise ValueError("工具调用消息缺失")
        if self.call_ref.part_index >= len(message.body.parts):
            raise ValueError("工具调用位置不存在")
        call = message.body.parts[self.call_ref.part_index]
        if not isinstance(call, ToolCall):
            raise ValueError("调用引用不指向 ToolCall")
        return call

    def check(self, state: OwnerStore) -> None:
        state.check_access(self.reader, self.writer)
        self.writer.check(ToolResult(self.call_ref, "unknown", ()))

    def read(self, pointer: object) -> Result:
        """按持久指针读取正文，不在工具回执中保留第二份结果。"""
        if not isinstance(pointer, Mapping):
            raise ValueError("工具结果指针损坏")
        value = cast(Mapping[str, object], pointer)
        if (
            set(value) != {"message_id", "seq"}
            or value["message_id"] != self.message_id
        ):
            raise ValueError("工具结果指针不匹配")
        seq = value["seq"]
        if type(seq) is not int or seq < 0:
            raise ValueError("工具结果序号无效")
        messages = self.reader.read(after_seq=seq - 1, through_seq=seq, limit=1)
        if len(messages) != 1 or messages[0].message_id != self.message_id:
            raise ValueError("工具结果消息缺失")
        body = messages[0].body
        if not isinstance(body, ToolResult) or body.call_ref != self.call_ref:
            raise ValueError("工具结果不属于原调用")
        return Result(body.outcome, body.parts)


class Denied(Exception):
    """授权 owner 明确拒绝当前最终参数；没有发生本次调用。"""


class BoundTool(Protocol):
    @property
    def idempotent(self) -> bool: ...

    async def prepare(
        self, arguments: Mapping[str, object]
    ) -> Mapping[str, object]: ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result: ...

    async def query(self, key: str) -> Result | None:
        """查询原调用；None 只表示无法确定，不能解释为没有效果。"""
        ...


OpenTool = Callable[[str], AbstractAsyncContextManager[BoundTool]]
Authorize = Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]


class ToolExecution:
    """拥有单次工具效果回执；只使用已获授的状态、Task 和实际绑定。"""

    def __init__(
        self,
        state: OwnerStore,
        tasks: Tasks,
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
        key = "message:" + json.dumps(
            [reply.call_ref.message_id, reply.call_ref.part_index],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return await self._execute(key, call.binding_id, call.arguments, reply)

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
        record = self._state.read(key)
        if record is not None:
            if record.value["version"] != 1 or record.value["phase"] not in {
                "prepared",
                "started",
                "done",
            }:
                raise ValueError("工具回执版本或阶段无效")
            if record.value["request"] != fingerprint:
                raise ValueError("同一工具 key 的 binding 或参数不一致")
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
                    final = freeze_json(await tool.prepare(arguments))
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
        record: OwnerRecord,
        result: Result,
        reply: MessageReply | None,
    ) -> Result:
        """结果正文与 receipt 指针同事务提交；独立调用直接保存结果。"""

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
                {**record.value, "phase": "done", "result": saved},
                expected_version=record.version,
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

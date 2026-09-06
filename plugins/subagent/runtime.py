from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal, cast

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS, Task, TaskSlot
from plugins.content.plugin import check_text
from plugins.conversation.plugin import CONVERSATION
from plugins.delivery.plugin import DELIVERY
from plugins.reply.api import REPLY_PROGRAM
from session.log import MessageReader, OwnerRecord, OwnerTransaction, SessionAttributes
from session.message import ContentPart, Control, Input, Message, Output
from session.message_codec import json_value

from .request import Request, check_request

logger = logging.getLogger(__name__)
SUBAGENT_PROGRAM = ServiceKey[Callable[[Task, MessageReader, Request], Awaitable[Message]]]("subagent.program.v1")
_MAX_ACTIVE = 3


class SubagentBusy(RuntimeError):
    """现有任务仍占准入名额，当前请求没有被接纳。"""


class Subagents:
    """来源拥有准入和回传；Task、模型、工具与消息分别由既有能力执行。"""

    def __init__(self, ctx: Context):
        self.ctx = ctx

    def read(self, key: str) -> tuple[OwnerRecord, Request, MessageReader] | None:
        record = self.ctx.require(OWNER_STATE).open(self.ctx).read(key)
        if record is None:
            return None
        value = record.value
        if (set(value) != {"session_id", "input_id", "settled"}
                or not isinstance(value["session_id"], str) or not isinstance(value["input_id"], str)
                or type(value["settled"]) is not bool):
            raise ValueError("子任务恢复指针损坏")
        reader = self.ctx.require(MESSAGE_CATALOG).reader(value["session_id"])
        message = reader.get(value["input_id"])
        if message is None or not isinstance(message.body, Input):
            raise ValueError("子任务原输入缺失")
        requests = [part for part in message.body.parts if part.kind == "subagent.request"]
        if len(requests) != 1:
            raise ValueError("子任务原输入缺少唯一请求")
        request = Request.model_validate(json_value(requests[0].value))
        if (request.session_id, request.input_id) != (reader.session_id, message.message_id):
            raise ValueError("子任务请求与恢复指针不一致")
        return record, request, reader

    def accept(self, key: str, request: Request, text: str) -> None:
        """同一输入与来源回执原子提交；重放不重新占额或创建任务目录。"""
        ctx = self.ctx
        state = ctx.require(OWNER_STATE).open(ctx)
        existing = self.read(key)
        if existing is None and len(self.jobs()) >= _MAX_ACTIVE:
            raise SubagentBusy("subagent capacity reached: max=3; current spawn rejected")
        _ = ctx.require(SESSION_ADMISSION).ensure(ctx, request.session_id,
            SessionAttributes(visibility="internal", learning="excluded"))
        writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="subagent", source="subagent",
            body_types=(Input,), content={"text": check_text, "subagent.request": check_request})(request.session_id)
        body = Input((ContentPart("text", text), ContentPart("subagent.request", request.model_dump())))
        def commit(tx: OwnerTransaction) -> None:
            previous = tx.read(key)
            if previous is None:
                _ = tx.save(key, {"session_id": request.session_id, "input_id": request.input_id,
                                  "settled": False}, expected_version=None)
            elif (previous.value["session_id"], previous.value["input_id"]) != (request.session_id, request.input_id):
                raise ValueError("同一子任务效果 key 已用于另一请求")
            _ = tx.append(writer, request.input_id, body)
        try:
            state.transact(commit)
        finally:
            writer.expire()
        if existing is None:
            self._trace(request, "started")

    def _trace(self, request: Request, phase: str) -> None:
        path = self.ctx.workspace_file("memory/spawn_trace.jsonl")
        value: dict[str, object] = {"version": 2, "job_id": request.job_id, "phase": phase,
                 "parent_session_id": request.parent_session_id, "parent_message_id": request.parent_message_id,
                 "parent_part_index": request.parent_part_index, "profile": request.profile,
                 "task_dir": str(self.ctx.workspace_root("subagent-runs") / request.job_id),
                 "timestamp": datetime.now(UTC).isoformat()}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                _ = stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        except OSError:
            # 诊断文件不是接纳或结算 owner；磁盘错误不能推翻已提交事实。
            logger.exception("子任务诊断写入失败 job=%s phase=%s", request.job_id, phase)

    async def start(self, key: str) -> Task | None:
        """同步工具与正式订阅者共用准入 key；程序归档覆盖整个工作生命期。"""
        def admit(slot: TaskSlot) -> Task | None:
            if slot.current is not None:
                return slot.current
            found = self.read(key)
            if found is None:
                raise KeyError(key)
            if found[0].value["settled"]:
                return None
            return slot.start(lambda task: self._run(task, key))
        return await self.ctx.require(TASKS).open(self.ctx).admit(("job", key), admit)

    async def _run(self, task: Task, key: str) -> None:
        """先恢复已提交调用，完成或明确取消后再回传；停机留下原输入。"""
        found = self.read(key)
        assert found is not None
        _, request, reader = found
        try:
            # 1. 只有未关闭输入才进入原程序；最终消息或控制足以决定恢复方向。
            if self.outcome(reader) is None:
                async with self.ctx.require(BINDINGS).open(request.program_binding, SUBAGENT_PROGRAM) as (program, _):
                    _ = await program(task, reader, request)
        except asyncio.CancelledError:
            if self.outcome(reader) is None:
                raise
        except Exception as error:
            if task.active:
                self._control(request, "failure", str(error) or type(error).__name__)
            else:
                raise
        # 2. 已开始的工具与原程序资源此时已经排空，回传不复制内部推理。
        outcome = self.outcome(reader)
        if outcome is None:
            raise RuntimeError("子任务程序没有保存终态")
        if not request.background:
            self._settle(key)
        self._trace(request, outcome[0])

    def _settle(self, key: str) -> None:
        state = self.ctx.require(OWNER_STATE).open(self.ctx)
        def commit(tx: OwnerTransaction) -> None:
            record = tx.read(key)
            assert record is not None
            _ = tx.save(key, {**record.value, "settled": True}, expected_version=record.version)
        state.transact(commit)

    @staticmethod
    def outcome(reader: MessageReader) -> tuple[str, str] | None:
        """只从持久正文和控制判断结果，不保存第二份回答或虚构模型成功。"""
        for message in reversed(reader.snapshot()):
            if message.source != "subagent":
                continue
            if isinstance(message.body, Output) and message.body.finish != "continue":
                return "completed", "\n".join(cast(str, part.value) for part in message.body.parts
                    if isinstance(part, ContentPart) and part.kind == "text")
            if isinstance(message.body, Control) and message.body.action in {"pause", "failure"}:
                return ("cancelled", "子任务已按请求取消。") if message.body.action == "pause" else (
                    "failed", message.body.reason or "子任务执行失败")
        return None

    def _control(self, request: Request, action: Literal["pause", "failure"], reason: str) -> None:
        reader = self.ctx.require(MESSAGE_CATALOG).reader(request.session_id)
        writer = self.ctx.require(MESSAGE_WRITERS).bind(self.ctx, author="subagent", source="subagent",
            body_types=(Control,), content={})(request.session_id)
        try:
            _ = writer.append(request.input_id + ":" + action, Control(action, reader.head(source="subagent"), reason))
        finally:
            writer.expire()

    async def cancel(self, job_id: str) -> bool:
        """取消意图先保存，再撤销新决策并等待原效果排空。"""
        for key, _ in self.ctx.require(OWNER_STATE).open(self.ctx).list():
            found = self.read(key)
            assert found is not None
            record, request, reader = found
            if request.job_id != job_id:
                continue
            outcome = self.outcome(reader)
            if outcome is not None:
                return outcome[0] == "cancelled"
            if record.value["settled"]:
                raise ValueError("子任务结算记录缺少终态")
            self._control(request, "pause", "用户取消子任务")
            def cancel(slot: TaskSlot) -> Task | None:
                current = slot.current
                if current is not None:
                    current.cancel()
                return current
            task = await self.ctx.require(TASKS).open(self.ctx).admit(("job", key), cancel)
            if task is not None:
                await drain(task)
            task = await self.start(key)
            if task is not None:
                await drain(task)
            return True
        return False

    async def _announce(self, request: Request, reader: MessageReader, outcome: tuple[str, str]) -> bool:
        """主程序读取低信任结果；原最终消息和发送回执共同承担崩溃恢复。"""
        ctx = self.ctx
        parent = ctx.require(MESSAGE_CATALOG).reader(request.parent_session_id)
        source = request.session_id
        def finished() -> Message | None:
            return next((item for item in reversed(parent.snapshot())
                         if item.source == source and isinstance(item.body, Output)
                         and item.body.finish in {"complete", "quiet"}), None)

        # 1. 原 job 独占来源；已完成的主回复不因发送失败再调用模型。
        message = finished()
        if message is None:
            original = reader.get(request.input_id)
            assert original is not None and isinstance(original.body, Input)
            task_text = "\n".join(cast(str, part.value) for part in original.body.parts if part.kind == "text")
            extra = (ContentPart("text", json.dumps({"kind": "background_task_result",
                "job_id": request.job_id, "label": request.label, "task": task_text,
                "status": outcome[0], "result": outcome[1][:12_000],
                "truncated": len(outcome[1]) > 12_000}, ensure_ascii=False)),)
            async def report(task: Task, current: MessageReader) -> Message:
                message = finished()
                if message is not None:
                    return message
                return await ctx.require(REPLY_PROGRAM)(task, current, source, extra)
            message = await ctx.require(CONVERSATION)(parent.session_id).complete(report)

        # 2. 主回复保存后，只恢复原目标的发送；unknown 不冒称已结算。
        assert request.sink is not None
        delivery = ctx.require(DELIVERY).open(ctx)
        selected = delivery.prepare(parent, message, (request.sink,))
        receipts = [await delivery.send(message.message_id, sink) for sink in selected.sinks]
        return all(receipt.status != "unknown" for receipt in receipts)

    def jobs(self) -> tuple[Mapping[str, object], ...]:
        result: list[Mapping[str, object]] = []
        for key, _ in self.ctx.require(OWNER_STATE).open(self.ctx).list():
            found = self.read(key)
            assert found is not None
            record, request, _ = found
            if not record.value["settled"]:
                result.append({"job_id": request.job_id, "label": request.label, "profile": request.profile})
        return tuple(result)

    async def follow(self) -> None:
        """正式生命周期追赶持久输入；短通知不持有工具归档或运行 generation。"""
        attempted: dict[str, tuple[int, int]] = {}
        active: dict[str, Task] = {}
        async def wait(key: str, task: Task) -> None:
            try:
                try:
                    _ = await task.join()
                except asyncio.CancelledError:
                    caller = asyncio.current_task()
                    if caller is not None and caller.cancelling():
                        raise
                async with self.ctx.runtime_scope():
                    found = self.read(key)
                    assert found is not None
                    record, request, reader = found
                    if request.background and not record.value["settled"]:
                        outcome = self.outcome(reader)
                        if outcome is None:
                            raise RuntimeError("子任务没有可回传的终态")
                        if await self._announce(request, reader, outcome):
                            self._settle(key)
            except asyncio.CancelledError:
                task.cancel()
                await drain(task)
                raise
            except Exception:
                logger.exception("子任务未结算，保留原消息和恢复指针 key=%s", key)
            finally:
                del active[key]
        try:
            async with asyncio.TaskGroup() as group:
                async for _ in self.ctx.require(MESSAGE_CATALOG).follow():
                    async with self.ctx.runtime_scope():
                        for key, _record in self.ctx.require(OWNER_STATE).open(self.ctx).list():
                            if key in active:
                                continue
                            found = self.read(key)
                            assert found is not None
                            record, request, reader = found
                            if record.value["settled"]:
                                continue
                            parent = self.ctx.require(MESSAGE_CATALOG).reader(request.parent_session_id)
                            heads = (reader.head(), parent.head())
                            if attempted.get(key) == heads:
                                continue
                            task = await self.start(key)
                            attempted[key] = heads
                            if task is not None:
                                active[key] = task
                                _ = group.create_task(wait(key, task))
        finally:
            tasks = tuple(active.values())
            for task in tasks:
                task.cancel()
            for task in tasks:
                await drain(task)


async def drain(task: Task) -> None:
    """等待者取消不截断已经开始的工具结算。"""
    while True:
        try:
            _ = await task.join()
            return
        except asyncio.CancelledError:
            if task.done:
                return


def completion(request: Request, outcome: tuple[str, str], *, limit: int) -> str:
    status, text = outcome
    if len(text) > limit:
        text = text[:limit] + f"\n...[结果已截断，原始长度 {len(text)}]"
    guidance = "最多重试一次；已有完整结果则直接向用户汇报。" if request.retry_count == 0 else "已经重试一次，请直接汇报已有结果。"
    return f"[子任务「{request.label}」结果]\n退出原因: {status}\n\n{text}\n\n{guidance}"

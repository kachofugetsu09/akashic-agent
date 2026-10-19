"""后台任务完成通知：忙时注入当前回合，闲时唤醒原会话并回传渠道。"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import CompositionError, Context
from agent.plugin_composition.tasks import Task, independent_task_context
from plugins.context.contract import MATERIALS_V4 as MATERIALS
from plugins.conversation.contract import CONVERSATION_COMPLETE_V2 as CONVERSATION_COMPLETE
from plugins.delivery.contract import DELIVERY_GUARDED_START as DELIVERY
from plugins.delivery.contract import DELIVERY_SENDERS
from plugins.ledger.contract import BINDINGS, MESSAGE_CATALOG, Input, Message, MessageReader, json_value
from plugins.reply.contract import REPLY_PROGRAM_V3 as REPLY_PROGRAM
from plugins.sources.contract import SourceGuard
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest

from .board import JobBoard, JobMirror, Notice
from .contract import JOB_NOTICES, JobOutcome, JobSnapshot, JobWatch

logger = logging.getLogger(__name__)
_STOPPED_BY_USER = "用户在界面上停止了此任务"
api_version = 3
name = "jobs"
version = "1.0.0"
desc = "后台任务完成通知：忙时注入当前回合，闲时唤醒原会话"
inject = (MESSAGE_CATALOG, MATERIALS, CONVERSATION_COMPLETE, REPLY_PROGRAM, DELIVERY, DELIVERY_SENDERS, BINDINGS)

_CONVERSATION = "conversation"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    completion_delivery: Literal["wakeup", "quiet"] = "wakeup"
    max_consecutive_wakes: int | None = Field(default=None, ge=1)


def _latest_input(messages: Sequence[Message], source: str) -> Message | None:
    return next((message for message in reversed(messages)
                 if message.source == source and isinstance(message.body, Input)), None)


def _origin(message: Message | None) -> tuple[str, str] | None:
    if message is None or not isinstance(message.body, Input):
        return None
    for part in message.body.parts:
        if part.kind == "channel.origin":
            value = cast(Mapping[str, str], json_value(part.value))
            return value["channel"], value["chat_id"]
    return None


class Jobs:
    """持有等待任务与通知状态；结算顺序由生产方的 settled 决定。"""

    def __init__(self, ctx: Context, config: Config) -> None:
        self._ctx = ctx
        self._config = config
        self._board = JobBoard(config.max_consecutive_wakes)
        self._mirror = JobMirror()
        self._queue: asyncio.Queue[JobWatch] = asyncio.Queue()
        self._running: dict[str, JobWatch] = {}
        self._stopped_by_user: set[str] = set()

    def watch(self, job: JobWatch) -> None:
        self._running[job.job_id] = job
        self._mirror.start(job)
        self._queue.put_nowait(job)

    def origin_source(self, session_id: str, source: str) -> str:
        return self._board.origin_source(session_id, source)

    # 1. 只停本会话登记、且生产方给了停止入口的运行中任务；2. 先进入停止中，再交给生产方终止；
    # 3. 结算时据 _stopped_by_user 注明是用户停止，通知照常送达模型，不让它以为任务还在跑。
    async def stop_job(self, session_id: str, job_id: str) -> bool:
        job = self._running.get(job_id)
        if job is None or job.session_id != session_id or job.stop is None:
            return False
        if not self._mirror.set_stopping(session_id, job_id, True):
            return False
        self._stopped_by_user.add(job_id)
        try:
            return await job.stop()
        except Exception:
            logger.warning("用户停止后台任务失败 job=%s", job_id, exc_info=True)
            self._stopped_by_user.discard(job_id)
            _ = self._mirror.set_stopping(session_id, job_id, False)
            return False

    # activation 关闭时结束全部镜像订阅；消费方据此失效旧快照并向新一代重新订阅。
    def close(self) -> None:
        self._mirror.close()

    def follow_jobs(self, session_id: str) -> AsyncGenerator[tuple[JobSnapshot, ...], None]:
        return self._mirror.follow(session_id)

    async def run(self) -> None:
        """插件任务持有全部等待者；卸载先取消并排空，再释放运行许可。"""
        async with asyncio.TaskGroup() as tasks:
            while True:
                job = await self._queue.get()
                _ = tasks.create_task(
                    self._follow(job), name=f"job-notice:{job.job_id}", context=independent_task_context(),
                )

    # 每一步准备材料时调用：当前输入领取未读结算，并在该输入的后续步骤中保持可回放。
    async def materials(self, snapshot: tuple[Message, ...], source: str) -> Mapping[str, object]:
        latest = _latest_input(snapshot, source)
        if latest is None:
            return {}
        notices = self._board.claim(latest.session_id, source, latest.message_id)
        if not notices:
            return {}
        return {"reminders": tuple(
            {"name": "job-" + notice.job.job_id, "text": notice.text(), "priority": 450, "replay": True}
            for notice in notices
        )}

    async def _follow(self, job: JobWatch) -> None:
        # 1. 生产方等待失败只丢弃这一条通知，不影响其他任务；无论结果如何都让镜像离开运行中。
        outcome: JobOutcome | None = None
        try:
            outcome = await job.settled()
        except Exception:
            logger.warning("后台任务结算等待失败 job=%s", job.job_id, exc_info=True)
        finally:
            _ = self._running.pop(job.job_id, None)
            if job.job_id in self._stopped_by_user:
                self._stopped_by_user.discard(job.job_id)
                outcome = _stopped_outcome(outcome)
            self._mirror.finish(job, outcome)
        if outcome is None:
            return
        notice = self._board.settle(job, outcome)
        if self._config.completion_delivery == "quiet" or job.source != _CONVERSATION:
            return
        # 2. 唤醒失败撤回领取，下一次用户输入仍会注入这条结算。
        try:
            async with self._ctx.runtime_scope():
                await self._wake(notice)
        except CompositionError:
            return
        except Exception:
            logger.exception("后台任务完成唤醒失败 job=%s", job.job_id)

    # 在原会话槽空闲后运行一轮；槽内再次核对，忙时已被回合领取的结算不再唤醒。
    async def _wake(self, notice: Notice) -> None:
        ctx, board, job = self._ctx, self._board, notice.job
        source = "job:" + job.job_id

        async def report(task: Task, current: MessageReader, check_admission: SourceGuard) -> Message:
            latest = current.latest_input(_CONVERSATION, through_seq=current.head())
            if latest is None:
                raise RuntimeError("后台任务的原会话没有用户输入")
            if not board.take_wake(notice, latest.message_id):
                return latest
            reminders: tuple[Mapping[str, object], ...] = (
                {"name": "background_job", "text": notice.text(), "priority": 500},
            )
            try:
                return await ctx.require(REPLY_PROGRAM)(task, current, source, reminders, check_admission=check_admission)
            except (Exception, asyncio.CancelledError):
                # complete 会在新输入抢占后重试；领取必须在本次执行退出时释放。
                board.release_wake(notice)
                raise

        message = await ctx.require(CONVERSATION_COMPLETE)(job.session_id, report)
        if message.source != source:
            return
        board.finish_wake(notice)
        await self._deliver(message)

    # 唤醒回复不属于 conversation 来源，按原会话最近输入的渠道显式回传。
    async def _deliver(self, message: Message) -> None:
        ctx = self._ctx
        reader = ctx.require(MESSAGE_CATALOG).reader(message.session_id)
        if reader.attributes.visibility == "internal":
            return
        origin = _origin(reader.latest_input(_CONVERSATION, through_seq=message.seq))
        if origin is None:
            return
        channel, address = origin
        try:
            binding = ctx.require(DELIVERY_SENDERS).bind(channel, ctx.require(BINDINGS))
        except KeyError:
            logger.warning("后台任务回传渠道没有发送能力 channel=%s", channel)
            return
        delivery = ctx.require(DELIVERY).open(ctx)
        selected = await delivery.prepare_async(
            reader, message, ({"name": channel, "binding_id": binding, "address": address},),
        )
        for sink in selected.sinks:
            receipt = await delivery.send(message.message_id, sink)
            if receipt.status != "delivered":
                logger.error("后台任务通知发送失败 message=%s reason=%s", message.message_id,
                             receipt.error or receipt.status)


# 用户停止后生产方可能已把条目回收（settled 返回 None），也可能先给出 killed 终态；两者都要让模型知道是谁停的。
def _stopped_outcome(outcome: JobOutcome | None) -> JobOutcome | None:
    if outcome is None:
        return JobOutcome("killed", _STOPPED_BY_USER)
    if outcome.status == "killed":
        return JobOutcome("killed", f"{outcome.detail}；{_STOPPED_BY_USER}")
    return outcome


async def apply(ctx: Context) -> None:
    config = Config.model_validate(ctx.config)
    jobs = Jobs(ctx, config)
    _ = await ctx.spawn(jobs.run(), name="job-notices")
    _ = await ctx.effect(lambda: jobs.close, label="job-mirror")
    _ = await ctx.provide(JOB_NOTICES, jobs)
    _ = await ctx.require(MATERIALS).register(ctx, name="job-notices", kind="context", prepare=jobs.materials)
    _ = await ctx.inject((UI_SLOTS,), lambda child: register_ui(child, jobs), name="plugin-ui")


# 后台任务列表是本插件自己的界面：会话快照经 follow 推送，停止经界面查询，宿主不认识任务。
async def register_ui(ctx: Context, jobs: Jobs) -> None:
    async def query(method: str, payload: dict[str, object], *,
                    session_id: str | None, turn_id: str | None) -> dict[str, object]:
        job_id = payload.get("job_id")
        if method != "job.stop" or set(payload) != {"job_id"} or not isinstance(job_id, str) or not job_id or not session_id:
            raise PluginUiRpcInvalidRequest("停止任务需要会话与 job_id")
        return {"accepted": await jobs.stop_job(session_id, job_id)}

    async def follow(session_id: str) -> AsyncGenerator[Mapping[str, object], None]:
        async with aclosing(jobs.follow_jobs(session_id)) as snapshots:
            async for snapshot in snapshots:
                yield {"jobs": [job.to_wire() for job in snapshot]}

    _ = await ctx.require(UI_SLOTS).register_plugin_ui(
        ctx, PluginUiDefinition(module="plugin_ui.js", stylesheet="plugin_ui.css", slots=("header.end",)),
        query=query, follow=follow,
    )

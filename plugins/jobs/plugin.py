"""后台任务完成通知：忙时注入当前回合，闲时唤醒原会话并回传渠道。"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from importlib import import_module
from html import escape
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import replace
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import CompositionError, Context, ServiceKey
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.tasks import Task, independent_task_context
from plugins.content.contract import CONTENT
from plugins.context.contract import MATERIALS_V4 as MATERIALS
from plugins.conversation.contract import CONVERSATION_COMPLETE_V2 as CONVERSATION_COMPLETE
from plugins.delivery.contract import DELIVERY_GUARDED_START as DELIVERY
from plugins.delivery.contract import DELIVERY_SENDERS
from plugins.gateway.contract import RpcMethod
from plugins.ledger.contract import (
    BINDINGS, MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, ContentPart, Control,
    Input, Message, MessageReader, Output, OwnerTransaction, is_user_input, json_value,
)
from plugins.reply.contract import REPLY_PROGRAM_V3 as REPLY_PROGRAM
from plugins.sources.contract import SourceGuard
from plugins.ui.contract import UI_SLOTS, Configuration, ModelSettings, PluginUiDefinition, PluginUiRpcInvalidRequest, mount_settings

from core.common.file_io import run_file_io

from .board import DEFAULT_MAX_WAKES, JobBoard, JobMirror, Notice
from .contract import JOB_NOTICES, JobOutcome, JobSnapshot, JobWatch
from .host_notices import HostNoticeParams, SavedNotice, read_failure

logger = logging.getLogger(__name__)
_STOPPED_BY_USER = "用户在界面上停止了此任务"
api_version = 3
name = "jobs"
version = "1.0.0"
desc = "后台任务完成通知：忙时注入当前回合，闲时唤醒原会话"
inject = (PLUGIN_CONFIG, MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, CONTENT, MATERIALS,
          CONVERSATION_COMPLETE, REPLY_PROGRAM, DELIVERY, DELIVERY_SENDERS, BINDINGS)

_CONVERSATION = "conversation"
_LIMIT_SOURCE = "jobs:limit"


SETTINGS = ServiceKey[Configuration]("jobs.settings.v1")
SETTING_FIELDS = ("completion_delivery", "max_consecutive_wakes")


# 宿主进程（如自部署 worker）在 Core 之外结束的任务，经 Gateway 补登一条已结算的通知。
HOST_NOTICE_METHOD = "jobs/host-notice"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    completion_delivery: Literal["wakeup", "quiet"] = "wakeup"
    max_consecutive_wakes: int | None = Field(default=DEFAULT_MAX_WAKES, ge=1)


def _latest_input(messages: Sequence[Message], source: str) -> Message | None:
    return next((message for message in reversed(messages)
                 if message.source == source and isinstance(message.body, Input)), None)


# Automatic goal inputs do not reset a user's budget or replace their channel.
def _latest_user_input(reader: MessageReader, through_seq: int) -> Message | None:
    def latest(messages: Iterable[Message]) -> Message | None:
        found = None
        for message in messages:
            if is_user_input(message):
                found = message
        return found
    return reader.scan(latest, source=_CONVERSATION, through_seq=through_seq)


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
        self._host_store = ctx.require(OWNER_STATE).open_scoped(ctx, "host-notices")
        self._host_scheduled: set[str] = set()
        self._host_received: set[str] = set()
        self._host_writer = ctx.entrypoint(ctx.require(MESSAGE_WRITERS).bind(
            ctx, author="jobs", source="jobs:host", body_types=(Output,),
            content={"text": ctx.require(CONTENT).check_text}))

    def watch(self, job: JobWatch) -> None:
        job = replace(job, source=self.origin_source(job.session_id, job.source))
        self._running[job.job_id] = job
        self._mirror.start(job)
        self._queue.put_nowait(job)

    def origin_source(self, session_id: str, source: str) -> str:
        source = self._board.origin_source(session_id, source)
        if source != _CONVERSATION:
            reader = self._ctx.require(MESSAGE_CATALOG).reader(session_id)
            if reader.latest_input(_CONVERSATION, through_seq=reader.head()) is not None:
                return _CONVERSATION
        return source

    # 宿主任务已在 Core 外结束：只接受有用户输入的对话会话，忙时注入当前回合，闲时按原配置唤醒。
    async def host_notice(self, params: BaseModel) -> object:
        notice = cast(HostNoticeParams, params)
        catalog = self._ctx.require(MESSAGE_CATALOG)
        if not await run_file_io(lambda: catalog.exists(notice.session_id)):
            raise ValueError("目标会话不存在")
        reader = catalog.reader(notice.session_id)
        if await reader.read_async(lambda current: _latest_user_input(current, current.head())) is None:
            raise ValueError("目标会话没有对话输入，无法注入或唤醒")
        saved = await self._save_host_notice(notice)
        # Durable acceptance makes the notice available even if channel delivery fails.
        if not saved.handled:
            self._schedule_host(saved.notice)
        message = reader.get("host-notice:" + notice.job_id)
        assert message is not None
        error = await self._deliver_host(notice.job_id, message)
        if error is None:
            self._host_received.add(notice.job_id)
        else:
            self._host_received.discard(notice.job_id)
        return {"accepted": True, "wake": self._config.completion_delivery == "wakeup",
                "delivery_error": error}

    # Keep one channel failure local to its accepted notice; Delivery owns send retries.
    async def _deliver_host(self, job_id: str, message: Message) -> str | None:
        error = await self._deliver(message)
        def save(tx: OwnerTransaction) -> bool:
            row = tx.read(job_id)
            assert row is not None
            saved = SavedNotice.model_validate(json_value(row.value))
            if saved.delivery_error == error:
                return False
            _ = tx.save(job_id, {**saved.model_dump(mode="json"), "delivery_error": error},
                        expected_version=row.version)
            return True
        changed = await self._host_store.transact_async(save)
        if changed and error is not None:
            _ = self._ctx.report_incident("host-notice-delivery", f"{job_id}: {error}")
        return error

    # Persist the visible failure and pending wake together before acknowledging the host.
    async def _save_host_notice(self, notice: HostNoticeParams) -> SavedNotice:
        writer = self._host_writer(notice.session_id)
        text = f"[宿主任务 {notice.status}] {notice.label}：{notice.detail}\n{notice.collect_hint}"
        try:
            prepared = await writer.prepare_async("host-notice:" + notice.job_id, Output((ContentPart("text", text),), "complete"))
            def commit(tx: OwnerTransaction) -> SavedNotice:
                row = tx.read(notice.job_id)
                saved = SavedNotice(notice=notice) if row is None else SavedNotice.model_validate(json_value(row.value))
                if saved.notice != notice:
                    raise ValueError("同一宿主任务编号不能改写通知")
                if row is None:
                    _ = tx.save(notice.job_id, saved.model_dump(mode="json"), expected_version=None)
                    _ = tx.append_prepared(prepared)
                return saved
            return await self._host_store.transact_async(commit)
        finally:
            writer.expire()

    def _schedule_host(self, notice: HostNoticeParams) -> None:
        if notice.job_id in self._host_scheduled:
            return
        self._host_scheduled.add(notice.job_id)
        async def settled() -> JobOutcome:
            return JobOutcome(notice.status, notice.detail)
        self.watch(JobWatch(notice.session_id, _CONVERSATION, notice.job_id, notice.kind,
                            notice.label, notice.collect_hint, settled))

    # Non-host jobs have no record here; host wakes remain recoverable until observed.
    async def _finish_host(self, job_id: str) -> None:
        if job_id not in self._host_scheduled:
            return
        def finish(tx: OwnerTransaction) -> None:
            row = tx.read(job_id)
            assert row is not None
            saved = SavedNotice.model_validate(json_value(row.value))
            if not saved.handled:
                _ = tx.save(job_id, {**saved.model_dump(mode="json"), "handled": True}, expected_version=row.version)
        await self._host_store.transact_async(finish)

    # The shared host-bridge directory already contains self-deploy records in Docker installs.
    async def _recover_host_notices(self) -> None:
        for job_id, row in await run_file_io(self._host_store.list):
            saved = SavedNotice.model_validate(json_value(row.value))
            if not saved.handled or saved.delivery_error is not None:
                try:
                    _ = await self._ctx.entrypoint(self.host_notice)(saved.notice)
                except (OSError, ValueError) as error:
                    _ = self._ctx.report_incident(
                        "host-notice-recovery", f"{job_id}: {type(error).__name__}: {error}",
                    )
            else:
                self._host_received.add(job_id)
        socket = os.environ.get("AKASHIC_HOST_BRIDGE_SOCKET")
        if not socket:
            return
        directory = Path(socket).parent / "self-deploy"
        errors: dict[Path, str] = {}
        while True:
            for path in await run_file_io(lambda: tuple(sorted(directory.glob("*.json")))):
                if path.stem in self._host_received:
                    continue
                try:
                    notice = await run_file_io(lambda: read_failure(path))
                    if notice is not None:
                        _ = await self._ctx.entrypoint(self.host_notice)(notice)
                    _ = errors.pop(path, None)
                except (OSError, ValueError) as error:
                    # One invalid task does not hide later failures; report changes, then retry.
                    detail = f"{path.name}: {type(error).__name__}: {error}"
                    if errors.get(path) != detail:
                        _ = self._ctx.report_incident("host-notice-recovery", detail)
                        errors[path] = detail
            await asyncio.sleep(5)

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
            _ = tasks.create_task(self._recover_host_notices(), name="host-notice-recovery")
            while True:
                job = await self._queue.get()
                _ = tasks.create_task(
                    self._follow(job), name=f"job-notice:{job.job_id}", context=independent_task_context(),
                )

    # Confirm claims from saved model reminders, including history outside the prompt window.
    async def _confirm_notices(self, session_id: str, source: str, through_seq: int) -> None:
        claims: dict[str, list[Notice]] = {}
        restored: list[Notice] = []
        for notice in self._board.pending(session_id):
            if notice.job.source != source or notice.confirmed:
                continue
            if notice.claimed_input is not None:
                claims.setdefault(notice.claimed_input, []).append(notice)
            elif notice.claimed_by is None and notice.job.job_id in self._host_scheduled:
                restored.append(notice)
        if not claims and not restored:
            return
        reader = self._ctx.require(MESSAGE_CATALOG).reader(session_id)
        def recorded(current: MessageReader) -> tuple[tuple[Notice, str], ...]:
            inputs = tuple(current.get(identity) for identity in claims)
            assert all(message is not None and isinstance(message.body, Input) for message in inputs)
            host_heads: dict[Notice, int] = {}
            for notice in restored:
                saved = current.get("host-notice:" + notice.job.job_id)
                assert saved is not None
                host_heads[notice] = saved.seq
            start = min([message.seq for message in inputs if message is not None] + list(host_heads.values()))
            found: dict[Notice, str] = {}
            def collect(messages: Iterable[Message]) -> None:
                for message in messages:
                    if not isinstance(message.body, Output):
                        continue
                    for part in message.body.parts:
                        if not isinstance(part, ContentPart) or part.kind != "model.facts":
                            continue
                        facts = cast(Mapping[str, object], part.value)
                        identity = cast(str | None, facts.get("reminder_input_id"))
                        reminder = cast(str | None, facts.get("reminder"))
                        if identity is None or reminder is None:
                            continue
                        candidates = (*claims.get(identity, ()), *(notice for notice, head in host_heads.items()
                                                                  if message.seq > head))
                        for notice in candidates:
                            if escape(notice.text(), quote=False) in reminder:
                                found[notice] = identity
            current.scan(collect, source=source, after_seq=start, through_seq=through_seq)
            return tuple(found.items())
        for notice, identity in await reader.read_async(recorded):
            await self._finish_host(notice.job.job_id)
            self._board.confirm(notice, identity)

    # 每一步准备材料时调用：当前输入领取未读结算，并在该输入的后续步骤中保持可回放。
    async def materials(self, snapshot: tuple[Message, ...], source: str) -> Mapping[str, object]:
        latest = _latest_input(snapshot, source)
        if latest is None:
            return {}
        await self._confirm_notices(latest.session_id, source, snapshot[-1].seq)
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
        if self._config.completion_delivery == "quiet":
            logger.info("[skipped] 后台任务完成不唤醒主 Agent：completion_delivery=quiet job=%s", job.job_id)
            return
        if job.source != _CONVERSATION:
            logger.info("[skipped] 后台任务完成不唤醒主 Agent：任务来源不是对话 job=%s source=%s", job.job_id, job.source)
            return
        # 2. 唤醒失败撤回领取，下一次用户输入仍会注入这条结算。
        # 3. 唤醒失败是终态：宿主记录标记已处理，重启不再重试；会话里的宿主通知已可见。
        try:
            async with self._ctx.runtime_scope():
                reader = self._ctx.require(MESSAGE_CATALOG).reader(job.session_id)
                await self._confirm_notices(job.session_id, job.source, reader.head())
                await self._wake(notice)
        except CompositionError as error:
            logger.warning("[skipped] 后台任务完成唤醒没有执行：能力不可用 job=%s 原因=%s", job.job_id, error)
            return
        except Exception:
            logger.exception("后台任务完成唤醒失败，不再重试 job=%s", job.job_id)
            await self._finish_host(job.job_id)

    # 在原会话槽空闲后运行一轮；槽内再次核对，忙时已被回合领取的结算不再唤醒。
    async def _wake(self, notice: Notice) -> None:
        ctx, board, job = self._ctx, self._board, notice.job
        source = "job:" + job.job_id

        async def report(task: Task, current: MessageReader, check_admission: SourceGuard) -> Message:
            latest = await current.read_async(lambda reader: _latest_user_input(reader, reader.head()))
            if latest is None:
                raise RuntimeError("后台任务的原会话没有用户输入")
            if not board.take_wake(notice, latest.message_id):
                if board.wake_limit_reached(notice, latest.message_id):
                    async with ctx.runtime_scope():
                        return await self._limit_message(task, current, latest.message_id, check_admission)
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
        if message.source == _LIMIT_SOURCE:
            await self._deliver(message)
            return
        if message.source != source:
            return
        # 唤醒回合可能以控制消息收尾：引擎撞步数上限放弃本段时，reply program 返回的就是
        # Control("abandon")；用户 /stop 打断时是 Control("pause")。控制没有正文，不是可
        # 回传的答复，照原样投递只会让渠道收到一条空消息并回 rejected。这次唤醒没把结果
        # 告诉用户，所以把结算还给下一次用户输入，而不是悄悄丢掉。
        if isinstance(message.body, Control):
            logger.info("后台任务唤醒没有可回传的正文 message=%s action=%s",
                        message.message_id, message.body.action)
            board.release_wake(notice)
            return
        board.finish_wake(notice)
        await self._finish_host(job.job_id)
        _ = await self._deliver(message)

    # Save one visible notice per user input without another model or tool turn.
    async def _limit_message(self, task: Task, reader: MessageReader, input_id: str,
                             check_admission: SourceGuard) -> Message:
        identity = "job-wake-limit:" + input_id
        saved = reader.get(identity)
        if saved is not None:
            return saved
        pending = sum(notice.claimed_by is None for notice in self._board.pending(reader.session_id))
        text = (f"连续自动唤醒已达上限（{self._config.max_consecutive_wakes} 次），已暂停自动唤醒。"
                f"目前有 {pending} 个后台任务结果待查看；后续结果也会保留到下次对话。"
                "请发送消息决定是否继续，或查看后台任务列表。")
        ctx = self._ctx
        writer = ctx.require(MESSAGE_WRITERS).bind(
            ctx, author="jobs", source=_LIMIT_SOURCE, body_types=(Output,),
            content={"text": ctx.require(CONTENT).check_text})(reader.session_id)
        task.on_close(writer.expire)
        try:
            prepared = await writer.prepare_async(identity, Output((ContentPart("text", text),), "complete"))
            def commit(tx: OwnerTransaction) -> Message:
                check_admission(transaction=tx)
                return tx.append_prepared(prepared)
            return await ctx.require(OWNER_STATE).open(ctx).transact_async(commit)
        finally:
            writer.expire()

    # 唤醒回复不属于 conversation 来源，按原会话最近输入的渠道显式回传。
    async def _deliver(self, message: Message) -> str | None:
        ctx = self._ctx
        reader = ctx.require(MESSAGE_CATALOG).reader(message.session_id)
        if reader.attributes.visibility == "internal":
            logger.info("[skipped] 后台任务结果不回传：原会话是内部会话 source=%s", message.source)
            return
        origin = _origin(await reader.read_async(lambda current: _latest_user_input(current, message.seq)))
        if origin is None:
            logger.warning("[paused] 后台任务结果不回传：找不到来源渠道 source=%s", message.source)
            return
        channel, address = origin
        try:
            binding = ctx.require(DELIVERY_SENDERS).bind(channel, ctx.require(BINDINGS))
        except KeyError:
            logger.warning("后台任务回传渠道没有发送能力 channel=%s", channel)
            return f"渠道没有发送能力: {channel}"
        delivery = ctx.require(DELIVERY).open(ctx)
        selected = await delivery.prepare_async(
            reader, message, ({"name": channel, "binding_id": binding, "address": address},),
        )
        errors: list[str] = []
        for sink in selected.sinks:
            # A sender can fail after its effect; keep Delivery's receipt and continue.
            try:
                receipt = await delivery.send(message.message_id, sink)
            except Exception as failure:
                logger.exception("后台任务通知发送异常 message=%s sink=%s", message.message_id, sink)
                errors.append(f"{sink}: {type(failure).__name__}: {failure}")
                continue
            if receipt.status != "delivered":
                logger.error("后台任务通知发送失败 message=%s reason=%s", message.message_id,
                             receipt.error or receipt.status)
                errors.append(f"{sink}: {receipt.error or receipt.status}")
        return "; ".join(errors)[:2000] if errors else None


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
    await mount_settings(ctx, SETTINGS, ModelSettings(ctx, Config, SETTING_FIELDS),
                         lambda: import_module(".settings_ui", __package__))
    _ = await ctx.spawn(jobs.run(), name="job-notices")
    _ = await ctx.effect(lambda: jobs.close, label="job-mirror")
    _ = await ctx.provide(JOB_NOTICES, jobs)
    _ = await ctx.provide(RpcMethod.key(HOST_NOTICE_METHOD), RpcMethod(HostNoticeParams, ctx.entrypoint(jobs.host_notice)))
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
        ctx, PluginUiDefinition(module="plugin_ui.js", stylesheet="plugin_ui.css", slots=("header.meta",)),
        query=query, follow=follow,
    )

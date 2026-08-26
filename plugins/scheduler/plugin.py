from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.control.scoped_turn import TurnAdmissionRetiredError
from agent.control.timer import TimerHandle, TimerStatus
from agent.plugin_composition import (
    DELIVERIES,
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    SCOPED_TURNS,
    TIMERS,
    TOOL_CATALOG,
    Context,
    PluginDeliveries,
    PluginScopedTurns,
    PluginTimers,
    PluginToolDefinition,
    ServiceView,
    ToolGrant,
    TurnExecutionScope,
)
from agent.scheduler import (
    JobStore,
    LatencyTracker,
    SCHEDULE_MAX_ACTIVE_JOBS,
    ScheduleCapacityError,
    ScheduledJob,
    compute_fire_at,
    compute_actual_trigger,
    is_cron_expr,
    next_cron_fire,
    parse_duration,
)
from agent.turn_effects import PostCommitEffect, TurnStorage

api_version = 3
name = "scheduler"
version = "3.0.0"
desc = "One-shot Timer composition for durable scheduled work"
author = "Akashic Core"
inject = (TIMERS, SCOPED_TURNS, DELIVERIES, TOOL_CATALOG)
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ("schedules.json",)

logger = logging.getLogger(__name__)
_GRACE_SECONDS = 300
_DISABLED_SOFT_TOOLS = (
    "message_push",
    "recall_memory",
    "memorize",
    "forget_memory",
)


@dataclass(slots=True)
class _Wait:
    job: ScheduledJob
    handle: TimerHandle
    task: asyncio.Task[None]


class SchedulerRuntime:
    """Own Scheduler state and compose waits, Turns, delivery, and settlement."""

    def __init__(
        self,
        store_path: Path,
        timers: PluginTimers,
        turns: PluginScopedTurns,
        deliveries: PluginDeliveries,
        *,
        now: Callable[[], datetime] | None = None,
        tracker: LatencyTracker | None = None,
    ) -> None:
        self.store = JobStore(store_path)
        self._timers = timers
        self._turns = turns
        self._deliveries = deliveries
        self._now = now or (lambda: datetime.now(UTC))
        self.tracker = tracker or LatencyTracker()
        self._jobs: dict[str, ScheduledJob] = {}
        self._persisted: dict[str, ScheduledJob] | None = None
        self._waits: dict[str, _Wait] = {}
        self._started = False
        self._closed = False

    @property
    def wait_count(self) -> int:
        return len(self._waits)

    async def start(self) -> None:
        """Load isolated state and arm one Timer per recovered active job."""

        if self._started:
            return
        if self._closed:
            raise RuntimeError("scheduler runtime 已关闭")
        self._ensure_loaded()
        self._started = True
        for job in self._jobs.values():
            self._arm(job)

    async def add_job(self, job: ScheduledJob) -> None:
        """Persist one candidate before publishing its one-shot wait."""

        self._ensure_loaded()
        if job.fire_at.tzinfo is None:
            job = replace(job, fire_at=job.fire_at.replace(tzinfo=UTC))
        current = self._current()
        candidate = dict(current)
        candidate[job.id] = job
        active = sum(1 for item in current.values() if item.enabled)
        candidate_active = sum(1 for item in candidate.values() if item.enabled)
        if candidate_active > SCHEDULE_MAX_ACTIVE_JOBS:
            raise ScheduleCapacityError(
                active_jobs=active,
                max_active_jobs=SCHEDULE_MAX_ACTIVE_JOBS,
            )
        self._commit(candidate)
        if self._started:
            self._arm(job)

    async def cancel_job(self, job_id: str) -> bool:
        self._ensure_loaded()
        candidate = self._current()
        if job_id not in candidate:
            return False
        del candidate[job_id]
        self._commit(candidate)
        wait = self._waits.pop(job_id, None)
        if wait is not None:
            _ = await wait.handle.cancel()
            await wait.handle.cleanup()
            if wait.task is not asyncio.current_task():
                _ = await asyncio.gather(wait.task, return_exceptions=True)
        return True

    async def cancel_job_by_name(self, name: str) -> list[str]:
        self._ensure_loaded()
        cancelled = [job.id for job in self._current().values() if job.name == name]
        for job_id in cancelled:
            _ = await self.cancel_job(job_id)
        return cancelled

    def match_job_ids(self, id_or_prefix: str) -> list[str]:
        self._ensure_loaded()
        return [
            job_id
            for job_id in self._current()
            if job_id == id_or_prefix or job_id.startswith(id_or_prefix)
        ]

    def list_jobs(self) -> list[ScheduledJob]:
        self._ensure_loaded()
        return list(self._jobs.values())

    async def close(self) -> None:
        """Cancel every owned wait/task without changing durable jobs."""

        if self._closed:
            return
        self._closed = True
        waits = tuple(self._waits.values())
        self._waits.clear()
        for wait in waits:
            _ = await wait.handle.cancel()
        if waits:
            _ = await asyncio.gather(
                *(wait.task for wait in waits),
                return_exceptions=True,
            )
        for wait in waits:
            await wait.handle.cleanup()

    def _load_and_recover(self) -> None:
        """Recover current jobs and persist deterministic misfire transitions."""

        now = self._aware_now()
        recovered: dict[str, ScheduledJob] = {}
        persisted: dict[str, ScheduledJob] = {}
        changed = False
        for original in self.store.load():
            job = original
            if not job.enabled:
                persisted[job.id] = job
                continue
            if job.fire_at.tzinfo is None:
                job = replace(job, fire_at=job.fire_at.replace(tzinfo=UTC))
            if job.fire_at <= now:
                age = (now - job.fire_at).total_seconds()
                if job.trigger == "every":
                    job = replace(job, fire_at=self._advance_every(job, now))
                    changed = True
                elif age > _GRACE_SECONDS:
                    persisted[job.id] = replace(job, enabled=False)
                    changed = True
                    continue
            recovered[job.id] = job
            persisted[job.id] = job
        if changed:
            self.store.save(persisted)
        self._persisted = persisted
        self._jobs = recovered

    def _ensure_loaded(self) -> None:
        """Load durable jobs once without claiming runtime lifecycle ownership."""

        if self._persisted is not None:
            return
        if self._closed:
            raise RuntimeError("scheduler runtime 已关闭")
        self._load_and_recover()

    def _arm(self, job: ScheduledJob) -> None:
        if self._closed or not job.enabled:
            return
        if job.id in self._waits:
            raise RuntimeError(f"scheduler job 已有 wait: {job.id}")
        deadline = compute_actual_trigger(job.fire_at, job.tier, self.tracker)
        handle = self._timers.schedule(deadline)
        task = asyncio.create_task(
            self._wait_and_fire(job, handle), name=f"scheduler:{job.id}"
        )
        self._waits[job.id] = _Wait(job, handle, task)

    async def _wait_and_fire(self, job: ScheduledJob, handle: TimerHandle) -> None:
        """Consume one Timer receipt, optionally run work, then settle one job."""

        succeeded = False
        cancelled = False
        handed_off = False
        try:
            receipt = await handle.result()
            if receipt.status is TimerStatus.CANCELLED:
                cancelled = True
                return
            current = self._jobs.get(job.id)
            if current is None or current is not job or not current.enabled:
                return
            await self._execute(job)
            succeeded = True
        except asyncio.CancelledError:
            cancelled = True
            raise
        except TurnAdmissionRetiredError:
            handed_off = True
        except Exception:
            logger.exception("scheduler job failed: %s", job.id)
        finally:
            _ = self._waits.pop(job.id, None)
            await handle.cleanup()
            if not cancelled and not handed_off:
                self._settle(job, succeeded)

    async def _execute(self, job: ScheduledJob) -> None:
        if job.tier == "instant":
            assert job.message is not None
            _ = await self._deliveries.send(
                channel=job.channel,
                chat_id=job.chat_id,
                content=job.message,
            )
            return

        assert job.prompt is not None
        session_id = await self._turns.ensure_session(
            f"scheduler:{job.id}",
            metadata={
                "programmatic": True,
                "ephemeral": True,
                "schedulerJobId": job.id,
            },
        )
        started = time.monotonic()
        handle = await self._turns.start(
            session_id,
            job.prompt,
            scope=TurnExecutionScope(
                tool_grant=ToolGrant.except_names(_DISABLED_SOFT_TOOLS),
                disabled_prompt_sections=frozenset({"memory"}),
                storage=TurnStorage.IN_MEMORY,
                post_commit_effect=PostCommitEffect.SUPPRESS,
                tool_source="scheduler",
            ),
            channel="scheduler",
            chat_id=job.id,
            sender="scheduler",
            busy_session_id=f"{job.channel}:{job.chat_id}",
        )
        try:
            result = await handle.result()
        finally:
            await handle.cleanup()
        if result.status.value != "completed":
            raise RuntimeError(f"scheduler soft Turn 未完成: {result.status.value}")
        content = result.final_response or ""
        if not content:
            raise RuntimeError("scheduler soft Turn 返回空内容")
        self.tracker.record(time.monotonic() - started)
        _ = await self._deliveries.send(
            channel=job.channel,
            chat_id=job.chat_id,
            content=content,
        )

    def _settle(self, job: ScheduledJob, succeeded: bool) -> None:
        candidate = self._current()
        if self._jobs.get(job.id) is not job:
            return
        next_job: ScheduledJob | None = None
        if job.trigger == "every" and not self._closed:
            after = max(self._aware_now(), job.fire_at) + timedelta(microseconds=1)
            next_job = replace(
                job,
                fire_at=self._advance_every(job, after),
                run_count=job.run_count + int(succeeded),
            )
            candidate[job.id] = next_job
        elif job.trigger != "every":
            candidate[job.id] = replace(
                job,
                enabled=False,
                run_count=job.run_count + int(succeeded),
            )
        self._commit(candidate)
        if next_job is not None:
            self._arm(next_job)

    def _commit(self, jobs: dict[str, ScheduledJob]) -> None:
        self.store.save(jobs)
        self._persisted = dict(jobs)
        self._jobs = {job_id: job for job_id, job in jobs.items() if job.enabled}

    def _current(self) -> dict[str, ScheduledJob]:
        return dict(self._persisted if self._persisted is not None else self._jobs)

    def _advance_every(self, job: ScheduledJob, after: datetime) -> datetime:
        if job.cron_expr:
            return next_cron_fire(job.cron_expr, job.timezone, after)
        interval_seconds = job.interval_seconds
        if type(interval_seconds) is not int or interval_seconds <= 0:
            raise ValueError(
                f"every 任务 interval_seconds 必须为正数: {interval_seconds}"
            )
        interval = timedelta(seconds=interval_seconds)
        if after < job.fire_at:
            return job.fire_at + interval
        return job.fire_at + ((after - job.fire_at) // interval + 1) * interval

    def _aware_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            raise ValueError("scheduler clock 必须返回带时区时间")
        return value.astimezone(UTC)


async def _schedule(
    bound: SchedulerRuntime, context: object, arguments: Mapping[str, object]
) -> str:
    """Validate one Tool request, persist its job, and return the first deadline."""

    _ = context
    tier = arguments.get("tier", "")
    trigger = arguments.get("trigger", "")
    when = arguments.get("when", "")
    message = arguments.get("message")
    prompt = arguments.get("prompt")
    channel = arguments.get("channel", "")
    chat_id = arguments.get("chat_id", "")
    timezone = arguments.get("timezone", os.environ.get("TZ", ""))
    job_name = arguments.get("name")
    request_time = arguments.get("request_time")

    error = _schedule_input_error(
        tier=tier,
        trigger=trigger,
        when=when,
        message=message,
        prompt=prompt,
        channel=channel,
        chat_id=chat_id,
        timezone=timezone,
        name=job_name,
        request_time=request_time,
    )
    if error is not None:
        return error
    assert isinstance(tier, str)
    assert isinstance(trigger, str)
    assert isinstance(when, str)
    assert isinstance(channel, str)
    assert isinstance(chat_id, str)
    assert isinstance(timezone, str)
    assert job_name is None or isinstance(job_name, str)
    assert request_time is None or isinstance(request_time, str)
    assert message is None or isinstance(message, str)
    assert prompt is None or isinstance(prompt, str)

    try:
        fire_at = compute_fire_at(trigger, when, timezone, request_time)
        interval_seconds, cron_expr = _recurrence(trigger, when)
    except (TypeError, ValueError, OverflowError) as exc:
        return f"错误：{exc}"
    job = ScheduledJob(
        trigger=trigger,
        tier=tier,
        fire_at=fire_at,
        channel=channel,
        chat_id=chat_id,
        interval_seconds=interval_seconds,
        cron_expr=cron_expr,
        message=message,
        prompt=prompt,
        name=job_name,
        timezone=timezone,
    )
    try:
        await bound.add_job(job)
    except ScheduleCapacityError as exc:
        return (
            f"{exc.code}：当前已有 {exc.active_jobs} 个活动定时任务，"
            f"默认上限为 {exc.max_active_jobs} 个。"
            "请询问用户要移除哪个不再需要的任务后再添加。"
        )
    label = f"「{job_name}」" if job_name else job.id[:8]
    return (
        f"已注册定时任务 {label}，首次触发时间：{_display_time(fire_at, request_time)}"
    )


async def _list_schedules(
    bound: SchedulerRuntime, context: object, arguments: Mapping[str, object]
) -> str:
    _ = context, arguments
    jobs = bound.list_jobs()
    if not jobs:
        return "当前没有待执行的定时任务"
    lines = [f"定时任务列表（共 {len(jobs)} 个）："]
    for job in jobs:
        try:
            display = job.fire_at.astimezone(ZoneInfo(job.timezone)).strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
        except (ZoneInfoNotFoundError, TypeError, ValueError, OverflowError, OSError):
            display = job.fire_at.isoformat()
        label = f"「{job.name}」" if job.name else job.id[:8]
        action = (
            (job.message or "")[:40]
            if job.tier == "instant"
            else f"[AI] {(job.prompt or '')[:40]}"
        )
        lines.append(
            f"• {label}  [{job.tier}/{job.trigger}]  下次: {display}  "
            f"内容: {action}  已运行: {job.run_count}次"
        )
    return "\n".join(lines)


async def _cancel_schedule(
    bound: SchedulerRuntime, context: object, arguments: Mapping[str, object]
) -> str:
    _ = context
    job_id = arguments.get("id", "")
    job_name = arguments.get("name", "")
    if not job_id and not job_name:
        return "错误：id 或 name 至少提供一个"
    if job_id:
        matches = bound.match_job_ids(str(job_id))
        if not matches:
            return f"未找到 ID 为 {job_id!r} 的任务"
        for matched in matches:
            _ = await bound.cancel_job(matched)
        return f"已取消 {len(matches)} 个任务"
    if job_name:
        cancelled = await bound.cancel_job_by_name(str(job_name))
        if not cancelled:
            return f"未找到名称为 {job_name!r} 的任务"
        return f"已取消 {len(cancelled)} 个名为 {job_name!r} 的任务"
    return "未指定有效的取消条件"


async def apply(ctx: Context, config: object) -> None:
    """Register production Tools and bind runtime work to formal lifecycle events."""

    _ = config
    timers = ctx.require(TIMERS)
    turns = ctx.require(SCOPED_TURNS)
    deliveries = ctx.require(DELIVERIES)
    bound = SchedulerRuntime(
        ctx.workspace_file("schedules.json"),
        timers,
        turns,
        deliveries,
    )

    async def schedule_handler(context: object, arguments: Mapping[str, object]) -> str:
        return await _schedule(bound, context, arguments)

    async def list_handler(context: object, arguments: Mapping[str, object]) -> str:
        return await _list_schedules(bound, context, arguments)

    async def cancel_handler(context: object, arguments: Mapping[str, object]) -> str:
        return await _cancel_schedule(bound, context, arguments)

    handlers = {
        "schedule": schedule_handler,
        "list_schedules": list_handler,
        "cancel_schedule": cancel_handler,
    }
    tools = ctx.require(TOOL_CATALOG)
    for definition in _tool_definitions():
        await tools.register(ctx, definition, handlers[definition.name])

    def setup() -> object:
        async def cleanup() -> None:
            await bound.close()

        return cleanup

    _ = await ctx.effect(setup, label="scheduler-runtime")

    async def start(_event: object) -> None:
        await bound.start()

    async def stop(_event: object) -> None:
        await bound.close()

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)


def is_active(services: ServiceView) -> bool:
    timers = services.get(TIMERS)
    turns = services.get(SCOPED_TURNS)
    deliveries = services.get(DELIVERIES)
    return bool(
        timers is not None
        and timers.formal
        and turns is not None
        and turns.formal
        and deliveries is not None
        and deliveries.formal
    )


def _schedule_input_error(**values: object) -> str | None:
    tier = values["tier"]
    trigger = values["trigger"]
    if tier not in ("instant", "soft"):
        return f"错误：tier 须为 instant 或 soft，收到 {tier!r}"
    if trigger not in ("at", "after", "every"):
        return f"错误：trigger 须为 at/after/every，收到 {trigger!r}"
    for field in ("when", "channel", "chat_id", "timezone"):
        if not isinstance(values[field], str):
            return f"错误：{field} 必须是字符串，收到 {values[field]!r}"
    for field in ("message", "prompt", "name"):
        value = values[field]
        if value is not None and not isinstance(value, str):
            return f"错误：{field} 必须是字符串，收到 {value!r}"
    request_time = values["request_time"]
    if (
        trigger == "after"
        and request_time is not None
        and not isinstance(request_time, str)
    ):
        return f"错误：request_time 必须是 ISO 字符串，收到 {request_time!r}"
    if tier == "instant" and not values["message"]:
        return "错误：tier=instant 时 message 为必填项"
    if tier == "soft" and not values["prompt"]:
        return "错误：tier=soft 时 prompt 为必填项"
    if not values["channel"] or not values["chat_id"]:
        return "错误：channel 和 chat_id 为必填项"
    try:
        _ = ZoneInfo(str(values["timezone"]))
    except (ValueError, ZoneInfoNotFoundError):
        return f"错误：无效的时区 {values['timezone']!r}"
    return None


def _recurrence(trigger: str, when: str) -> tuple[int | None, str | None]:
    if trigger != "every":
        return None, None
    if is_cron_expr(when):
        return None, when.strip()
    return int(parse_duration(when).total_seconds()), None


def _display_time(fire_at: datetime, request_time: str | None) -> str:
    try:
        if fire_at.tzinfo is not None and str(fire_at.tzinfo) not in ("UTC", "utc"):
            display = fire_at
        elif request_time:
            parsed = datetime.fromisoformat(request_time)
            display = (
                fire_at.astimezone(parsed.tzinfo)
                if parsed.tzinfo
                else fire_at.astimezone()
            )
        else:
            display = fire_at.astimezone()
        return display.strftime("%Y-%m-%d %H:%M:%S %z")
    except (TypeError, ValueError, OverflowError, OSError):
        return fire_at.isoformat()


def _tool_definitions() -> tuple[PluginToolDefinition, ...]:
    return (
        PluginToolDefinition(
            name="schedule",
            description=(
                "注册定时任务。支持三种触发模式：\n"
                "  at    — 指定绝对时间，如 '14:30' 或 '2025-06-01T09:00'\n"
                "  after — 相对延迟，如 '30s' '5m' '2h'（需传 request_time 补偿延迟）\n"
                "  every — 循环，如 '1h' '30m' '0 9 * * *'（每天9点）\n\n"
                "两种执行模式：\n"
                "  instant — 到时直接推送固定消息，适合喝水提醒等固定文本\n"
                "  soft    — 到时调用 AI 生成实时内容，适合天气/新闻等"
            ),
            parameters=_schedule_schema(),
            handler_export="schedule",
            risk="read-write",
            search_hint="cron timer 延时执行",
        ),
        PluginToolDefinition(
            name="list_schedules",
            description="列出所有待执行的定时任务",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler_export="list_schedules",
            risk="read-only",
            search_hint="提醒列表 已有计划",
        ),
        PluginToolDefinition(
            name="cancel_schedule",
            description="取消定时任务。可按任务 ID 或名称取消",
            parameters={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "任务 ID 或其前缀（至少8位）",
                    },
                    "name": {"type": "string", "description": "任务名称"},
                },
                "required": [],
                "additionalProperties": False,
            },
            handler_export="cancel_schedule",
            risk="read-write",
            search_hint="删除提醒 取消任务",
        ),
    )


def _schedule_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "tier": {
                "type": "string",
                "enum": ["instant", "soft"],
                "description": "instant=直接推消息；soft=触发时调用AI生成内容",
            },
            "trigger": {
                "type": "string",
                "enum": ["at", "after", "every"],
                "description": "触发模式",
            },
            "when": {
                "type": "string",
                "description": "触发时间描述，与 trigger 对应：\n  at    → '14:30' 或 '2025-06-01T09:00'\n  after → '30s' '5m' '2h'\n  every → '1h' '30m' '0 9 * * *'",
            },
            "message": {
                "type": "string",
                "description": "tier=instant 时的消息内容（必填）",
            },
            "prompt": {
                "type": "string",
                "description": "tier=soft 时触发 AI 的提示词（必填）",
            },
            "channel": {"type": "string", "description": "目标渠道，如 telegram、qq"},
            "chat_id": {"type": "string", "description": "目标会话 ID"},
            "timezone": {
                "type": "string",
                "description": "时区，如 Asia/Shanghai，默认使用系统配置",
            },
            "name": {
                "type": "string",
                "description": "任务名，方便后续用 cancel_schedule 取消",
            },
            "request_time": {
                "type": "string",
                "description": "trigger=after 时必填：来自 system prompt 的消息接收时间（ISO 格式）。用于从用户发消息时刻计算延迟，而非从 tool 调用时刻计算。",
            },
        },
        "required": ["tier", "trigger", "when", "channel", "chat_id"],
        "additionalProperties": False,
    }

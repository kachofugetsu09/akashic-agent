from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.control.timer import TimerHandle, TimerStatus
from agent.plugin_composition import (
    DELIVERIES,
    SCOPED_TURNS,
    TIMERS,
    Context,
    PluginDeliveries,
    PluginScopedTurns,
    PluginTimers,
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
    compute_actual_trigger,
    next_cron_fire,
)

api_version = 3
name = "scheduler"
version = "3.0.0"
desc = "One-shot Timer composition for durable scheduled work"
author = "Akashic Core"
inject = (TIMERS, SCOPED_TURNS, DELIVERIES)
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


class SchedulerShadowRuntime:
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
        self._closed = False

    @property
    def wait_count(self) -> int:
        return len(self._waits)

    async def start_shadow(self) -> None:
        """Load isolated state and arm one Timer per recovered active job."""

        if self._persisted is not None:
            raise RuntimeError("scheduler shadow 已启动")
        self._load_and_recover()
        for job in self._jobs.values():
            self._arm(job)

    async def add_job(self, job: ScheduledJob) -> None:
        """Persist one candidate before publishing its one-shot wait."""

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
        self._arm(job)

    async def cancel_job(self, job_id: str) -> bool:
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

    def list_jobs(self) -> list[ScheduledJob]:
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
        except Exception:
            logger.exception("scheduler shadow job failed: %s", job.id)
        finally:
            _ = self._waits.pop(job.id, None)
            await handle.cleanup()
            if not cancelled:
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
                memory_read=False,
                memory_write=False,
                stateless=True,
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


runtime: SchedulerShadowRuntime | None = None


async def apply(ctx: Context, config: object) -> None:
    """Mount a dormant S3 shadow runtime without reading or arming formal jobs."""

    _ = config

    def setup() -> object:
        global runtime
        if runtime is not None:
            raise RuntimeError("scheduler plugin runtime 重复激活")
        bound = SchedulerShadowRuntime(
            ctx.workspace_file("schedules.json"),
            ctx.require(TIMERS),
            ctx.require(SCOPED_TURNS),
            ctx.require(DELIVERIES),
        )
        runtime = bound

        async def cleanup() -> None:
            global runtime
            await bound.close()
            if runtime is bound:
                runtime = None

        return cleanup

    _ = await ctx.effect(setup, label="scheduler-shadow-runtime")


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

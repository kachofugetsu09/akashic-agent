from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

from bus.event_bus import EventSubscription

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshotLease, RuntimeSnapshotStore

logger = logging.getLogger(__name__)


class PluginLlmService(Protocol):
    async def generate_text(
        self,
        *,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class IntervalTrigger:
    seconds: int


@dataclass(frozen=True)
class EventTrigger:
    event_type: type[object]


PluginJobTrigger = IntervalTrigger | EventTrigger


@dataclass(frozen=True)
class PluginJobContext:
    plugin_id: str
    event: object | None
    reason: str
    llm: PluginLlmService
    plugin_context: Any
    triggered_at: datetime


PluginJobHandler = Callable[[PluginJobContext], Awaitable[None]]


@dataclass(frozen=True)
class PluginJobSpec:
    id: str
    triggers: Sequence[PluginJobTrigger]
    handler: PluginJobHandler
    debounce_seconds: int = 0
    coalesce: bool = True


@dataclass(frozen=True)
class RegisteredPluginJob:
    plugin_id: str
    plugin_context: Any
    spec: PluginJobSpec


def plugin_job_key(job: RegisteredPluginJob) -> str:
    return f"{job.plugin_id}:{job.spec.id}"


@dataclass(frozen=True)
class _JobRequest:
    key: str
    reason: str
    event: object | None
    job: RegisteredPluginJob
    snapshot_lease: RuntimeSnapshotLease | None


class ProviderPluginLlmService:
    def __init__(
        self,
        provider: Any,
        *,
        model: str,
        max_tokens: int,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens

    async def generate_text(
        self,
        *,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await self._provider.chat(
            messages=messages,
            tools=[],
            model=model or self._model,
            max_tokens=max_tokens or self._max_tokens,
        )
        return str(resp.content or "").strip()


class PluginJobRuntime:
    def __init__(
        self,
        *,
        event_bus: Any,
        llm: PluginLlmService,
        jobs: list[RegisteredPluginJob] | None = None,
        snapshot_store: RuntimeSnapshotStore | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._llm = llm
        self._jobs = {plugin_job_key(job): job for job in jobs or []}
        self._snapshot_store = snapshot_store
        self._queue: asyncio.Queue[_JobRequest | None] = asyncio.Queue()
        self._queued_keys: set[str] = set()
        self._last_run_at: dict[str, datetime] = {}
        self._interval_task: asyncio.Task[None] | None = None
        self._interval_due: dict[tuple[str, int, int], float] = {}
        self._running = False
        self._bound = False
        self._subscriptions: list[EventSubscription] = []
        self._pending_enqueue_tasks: set[asyncio.Task[None]] = set()
        self._stopped = asyncio.Event()
        self._stopped.set()

    async def run(self) -> None:
        if self._running:
            return
        self._running = True
        self._stopped.clear()
        self._bind_triggers()
        self._interval_task = asyncio.create_task(
            self._interval_loop(),
            name="plugin_job_intervals",
        )
        try:
            while self._running:
                request = await self._queue.get()
                try:
                    if request is None:
                        return
                    self._queued_keys.discard(request.key)
                    await self._run_one(request)
                finally:
                    self._queue.task_done()
        finally:
            if self._interval_task is not None:
                _ = self._interval_task.cancel()
                _ = await asyncio.gather(self._interval_task, return_exceptions=True)
                self._interval_task = None
            self._interval_due.clear()
            for subscription in self._subscriptions:
                subscription.close()
            self._subscriptions.clear()
            for task in self._pending_enqueue_tasks:
                _ = task.cancel()
            if self._pending_enqueue_tasks:
                _ = await asyncio.gather(
                    *self._pending_enqueue_tasks,
                    return_exceptions=True,
                )
            self._pending_enqueue_tasks.clear()
            self._bound = False
            self._running = False
            while not self._queue.empty():
                request = self._queue.get_nowait()
                if request is not None:
                    self._queued_keys.discard(request.key)
                    if request.snapshot_lease is not None:
                        await request.snapshot_lease.release()
                self._queue.task_done()
            self._stopped.set()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        _ = self._queue.put_nowait(None)

    async def wait_stopped(self) -> None:
        _ = await self._stopped.wait()

    def enqueue(
        self,
        key: str,
        *,
        reason: str,
        event: object | None = None,
    ) -> None:
        from agent.plugins.snapshot import get_current_runtime_snapshot

        if (
            self._snapshot_store is not None
            and get_current_runtime_snapshot() is None
            and self._snapshot_store.current is not None
            and not self._snapshot_store.current.accepting_leases
        ):
            task = asyncio.create_task(
                self._enqueue_after_admission(key, reason=reason, event=event),
                name="plugin_job_enqueue_admission",
            )
            self._pending_enqueue_tasks.add(task)
            task.add_done_callback(self._pending_enqueue_tasks.discard)
            return
        job, snapshot_lease = self._resolve_job(key)
        if job is None:
            return
        self._queued_keys.add(key)
        self._queue.put_nowait(
            _JobRequest(
                key=key,
                reason=reason,
                event=event,
                job=job,
                snapshot_lease=snapshot_lease,
            )
        )

    async def _enqueue_after_admission(
        self,
        key: str,
        *,
        reason: str,
        event: object | None,
    ) -> None:
        assert self._snapshot_store is not None
        lease = await self._snapshot_store.acquire()
        job = lease.snapshot.jobs.get(key)
        if job is None or (job.spec.coalesce and key in self._queued_keys):
            await lease.release()
            return
        self._queued_keys.add(key)
        self._queue.put_nowait(
            _JobRequest(
                key=key,
                reason=reason,
                event=event,
                job=job,
                snapshot_lease=lease,
            )
        )

    def _bind_triggers(self) -> None:
        if self._bound:
            return
        self._bound = True
        if self._snapshot_store is not None:
            self._subscriptions.append(self._event_bus.on_any(self._handle_event))
            return
        for key, job in self._jobs.items():
            for trigger in job.spec.triggers:
                if isinstance(trigger, EventTrigger):
                    self._subscriptions.append(
                        self._event_bus.on(
                            trigger.event_type,
                            self._make_event_handler(key),
                        )
                    )

    def _make_event_handler(
        self,
        key: str,
    ) -> Callable[[object], None]:
        def handler(event: object) -> None:
            self.enqueue(key, reason="event", event=event)

        return handler

    def _handle_event(self, event: object) -> None:
        for key, job in self._current_jobs().items():
            if any(
                isinstance(trigger, EventTrigger)
                and type(event) is trigger.event_type
                for trigger in job.spec.triggers
            ):
                self.enqueue(key, reason="event", event=event)

    async def _interval_loop(self) -> None:
        while self._running:
            now = time.monotonic()
            active: set[tuple[str, int, int]] = set()
            for key, job in self._current_jobs().items():
                for index, trigger in enumerate(job.spec.triggers):
                    if not isinstance(trigger, IntervalTrigger):
                        continue
                    interval = max(1, int(trigger.seconds))
                    schedule_key = (key, index, interval)
                    active.add(schedule_key)
                    due = self._interval_due.setdefault(schedule_key, now + interval)
                    if now >= due:
                        self.enqueue(key, reason="interval")
                        self._interval_due[schedule_key] = now + interval
            for schedule_key in self._interval_due.keys() - active:
                del self._interval_due[schedule_key]
            await asyncio.sleep(0.2)

    async def _run_one(
        self,
        request: _JobRequest,
    ) -> None:
        job = request.job
        if self._debounced(request.key, job.spec.debounce_seconds):
            if request.snapshot_lease is not None:
                await request.snapshot_lease.release()
            return
        ctx = PluginJobContext(
            plugin_id=job.plugin_id,
            event=request.event,
            reason=request.reason,
            llm=self._llm,
            plugin_context=job.plugin_context,
            triggered_at=datetime.now(timezone.utc),
        )
        try:
            await self._invoke(request, ctx)
        except Exception:
            logger.exception(
                "插件后台任务失败: plugin=%s job=%s reason=%s",
                job.plugin_id,
                job.spec.id,
                request.reason,
            )
        else:
            self._last_run_at[request.key] = ctx.triggered_at

    async def _invoke(self, request: _JobRequest, ctx: PluginJobContext) -> None:
        lease = request.snapshot_lease
        if lease is None:
            await request.job.spec.handler(ctx)
            return
        from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

        async with lease:
            token = bind_runtime_snapshot(lease)
            try:
                await request.job.spec.handler(ctx)
            finally:
                reset_runtime_snapshot(token)

    def _current_jobs(self) -> dict[str, RegisteredPluginJob]:
        if self._snapshot_store is not None:
            from agent.plugins.snapshot import get_current_runtime_snapshot

            snapshot = get_current_runtime_snapshot()
            if snapshot is not None:
                return dict(snapshot.jobs)
        if self._snapshot_store is None or self._snapshot_store.current is None:
            return self._jobs
        return dict(self._snapshot_store.current.jobs)

    def _resolve_job(
        self,
        key: str,
    ) -> tuple[RegisteredPluginJob | None, RuntimeSnapshotLease | None]:
        if self._snapshot_store is None:
            job = self._jobs.get(key)
            if job is not None and job.spec.coalesce and key in self._queued_keys:
                return None, None
            return job, None
        from agent.plugins.snapshot import (
            get_current_runtime_snapshot,
            lease_current_runtime_snapshot,
        )

        snapshot = get_current_runtime_snapshot() or self._snapshot_store.current
        if snapshot is None:
            return None, None
        job = snapshot.jobs.get(key)
        if job is None:
            return None, None
        if job.spec.coalesce and key in self._queued_keys:
            return None, None
        lease = lease_current_runtime_snapshot()
        if lease is None:
            lease = self._snapshot_store.lease(snapshot.snapshot_id)
        return job, lease

    def _debounced(
        self,
        key: str,
        seconds: int,
    ) -> bool:
        if seconds <= 0:
            return False
        last = self._last_run_at.get(key)
        if last is None:
            return False
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed < seconds

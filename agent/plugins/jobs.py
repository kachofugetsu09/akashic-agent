from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

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
    triggers: list[PluginJobTrigger]
    handler: PluginJobHandler
    debounce_seconds: int = 0
    coalesce: bool = True


@dataclass(frozen=True)
class RegisteredPluginJob:
    plugin_id: str
    plugin_context: Any
    spec: PluginJobSpec


@dataclass(frozen=True)
class _JobRequest:
    key: str
    reason: str
    event: object | None


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
        jobs: list[RegisteredPluginJob],
    ) -> None:
        self._event_bus = event_bus
        self._llm = llm
        self._jobs = {self._job_key(job): job for job in jobs}
        self._queue: asyncio.Queue[_JobRequest | None] = asyncio.Queue()
        self._queued_keys: set[str] = set()
        self._last_run_at: dict[str, datetime] = {}
        self._interval_tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._bound = False

    async def run(self) -> None:
        if self._running:
            return
        self._running = True
        self._bind_triggers()
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
            for task in self._interval_tasks:
                _ = task.cancel()
            _ = await asyncio.gather(*self._interval_tasks, return_exceptions=True)

    def stop(self) -> None:
        self._running = False
        _ = self._queue.put_nowait(None)

    def enqueue(
        self,
        key: str,
        *,
        reason: str,
        event: object | None = None,
    ) -> None:
        job = self._jobs.get(key)
        if job is None:
            return
        if job.spec.coalesce and key in self._queued_keys:
            return
        self._queued_keys.add(key)
        self._queue.put_nowait(_JobRequest(key=key, reason=reason, event=event))

    def _bind_triggers(self) -> None:
        if self._bound:
            return
        self._bound = True
        for key, job in self._jobs.items():
            for trigger in job.spec.triggers:
                if isinstance(trigger, EventTrigger):
                    self._event_bus.on(
                        trigger.event_type,
                        self._make_event_handler(key),
                    )
                elif isinstance(trigger, IntervalTrigger):
                    self._interval_tasks.append(
                        asyncio.create_task(
                            self._interval_loop(key, trigger.seconds),
                            name=f"plugin_job_interval:{key}",
                        )
                    )

    def _make_event_handler(
        self,
        key: str,
    ) -> Callable[[object], None]:
        def handler(event: object) -> None:
            self.enqueue(key, reason="event", event=event)

        return handler

    async def _interval_loop(
        self,
        key: str,
        seconds: int,
    ) -> None:
        interval = max(1, int(seconds))
        while self._running:
            await asyncio.sleep(interval)
            self.enqueue(key, reason="interval")

    async def _run_one(
        self,
        request: _JobRequest,
    ) -> None:
        job = self._jobs.get(request.key)
        if job is None:
            return
        if self._debounced(request.key, job.spec.debounce_seconds):
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
            await job.spec.handler(ctx)
        except Exception:
            logger.exception(
                "插件后台任务失败: plugin=%s job=%s reason=%s",
                job.plugin_id,
                job.spec.id,
                request.reason,
            )
        else:
            self._last_run_at[request.key] = ctx.triggered_at

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

    @staticmethod
    def _job_key(job: RegisteredPluginJob) -> str:
        return f"{job.plugin_id}:{job.spec.id}"

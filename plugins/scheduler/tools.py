from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Literal, Self, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent.plugin_composition.tasks import Task, TaskAdmission, TaskSlot
from plugins.tools.api import CallSource, InvalidArguments, Result
from session.message import ContentPart

from .schedule import ScheduledJob, compute_fire_at, is_cron_expr, parse_duration
from .store import JobStore, Operation


class ScheduleInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tier: Literal["instant", "soft"]
    trigger: Literal["at", "after", "every"]
    when: str = Field(min_length=1, description="at: 14:30 或 ISO 时间；after: 5m；every: 1h 或 cron")
    channel: str = Field(min_length=1)
    chat_id: str = Field(min_length=1)
    timezone: str = Field(default_factory=lambda: os.environ.get("TZ", ""))
    message: str | None = None
    prompt: str | None = None
    name: str | None = None
    request_time: str | None = Field(default=None, description="原消息接收的 ISO 时间；after 从此刻计算延迟")

    @model_validator(mode="after")
    def check_payload(self) -> Self:
        if not (self.message if self.tier == "instant" else self.prompt):
            raise ValueError("instant 需要 message；soft 需要 prompt")
        return self


class CancelInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = ""
    name: str = ""

    @model_validator(mode="after")
    def check_selector(self) -> Self:
        if not self.id and not self.name:
            raise ValueError("id 或 name 至少提供一个")
        return self


def _result(operation: Operation) -> Result:
    return Result(operation.outcome, (ContentPart("text", operation.response),))


async def drain(task: Task) -> None:
    """领域取消等待真实关闭；等待者重复取消不能截断效果结算。"""
    while True:
        try:
            _ = await task.join()
            return
        except asyncio.CancelledError:
            if task.done:
                return


async def cancel_fires(store: JobStore, tasks: TaskAdmission, job_ids: tuple[str, ...]) -> None:
    """持久取消后只撤权；原 fire 自行排空，避免取消工具与所属 fire 互相等待。"""
    def cancel(slot: TaskSlot) -> None:
        task = slot.current
        if task is not None:
            task.cancel()
    for key, fire in store.read().fires.items():
        if fire.job.id in job_ids and fire.status == "cancelled":
            await tasks.admit(("fire", key), cancel)


class ScheduleTool:
    """prepare 固定实际参数，invoke 与 query 只处理原操作和原任务集合。"""

    idempotent = True

    def __init__(self, store: JobStore, tasks: TaskAdmission, kind: Literal["schedule", "cancel"],
                 *, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self._store = store
        self._tasks = tasks
        self._kind = kind
        self._now = now

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        """最终 ID、时间与取消集合在 Tool 的 prepared 回执里固定，不在 invoke 重算。"""
        if self._kind == "cancel":
            try:
                request = CancelInput.model_validate(dict(arguments))
            except ValidationError as error:
                raise InvalidArguments(str(error)) from error
            jobs = self._store.load()
            ids = tuple(job.id for job in jobs if (
                job.id.startswith(request.id) if request.id else job.name == request.name))
            return {"job_ids": ids}
        try:
            request = ScheduleInput.model_validate(dict(arguments))
            _ = ZoneInfo(request.timezone)
            fire_at = compute_fire_at(request.trigger, request.when, request.timezone,
                                      request.request_time, self._now)
            interval, cron = None, None
            if request.trigger == "every":
                if is_cron_expr(request.when):
                    cron = request.when.strip()
                else:
                    interval = int(parse_duration(request.when).total_seconds())
            job = ScheduledJob(trigger=request.trigger, tier=request.tier, fire_at=fire_at,
                channel=request.channel, chat_id=request.chat_id, interval_seconds=interval,
                cron_expr=cron, message=request.message, prompt=request.prompt, name=request.name,
                timezone=request.timezone, created_at=self._now())
            payload = self._store.encode_job(job)
            _ = self._store.decode_job(payload)
        except (ValidationError, ValueError, TypeError, OverflowError, ZoneInfoNotFoundError) as error:
            raise InvalidArguments(str(error)) from error
        label = f"「{job.name}」" if job.name else job.id[:8]
        return {"job": payload, "response": f"已注册定时任务 {label}，首次触发时间：{fire_at.isoformat()}"}

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """只接纳 Tool owner 已固定的执行参数，结果与文件变更同一次提交。"""
        if self._kind == "cancel":
            ids = arguments["job_ids"]
            if set(arguments) != {"job_ids"} or not isinstance(ids, (tuple, list)):
                raise ValueError("原取消参数损坏")
            values = cast(tuple[object, ...] | list[object], ids)
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError("原取消参数损坏")
            ids = tuple(cast(tuple[str, ...], values))
            operation = self._store.cancel(key, ids)
            await cancel_fires(self._store, self._tasks, ids)
        else:
            job = arguments["job"]
            response = arguments["response"]
            if set(arguments) != {"job", "response"} or not isinstance(job, Mapping) or not isinstance(response, str):
                raise ValueError("原调度参数损坏")
            operation = self._store.add(key, self._store.decode_job(dict(cast(Mapping[str, object], job))), response)
        return _result(operation)

    async def query(self, key: str) -> Result | None:
        operation = self._store.read().operations.get(key)
        if operation is None:
            return None
        if operation.kind != self._kind:
            raise ValueError("调度操作 key 属于另一种工具")
        if self._kind == "cancel":
            await cancel_fires(self._store, self._tasks, operation.job_ids)
        return _result(operation)


class ListSchedules:
    idempotent = True

    def __init__(self, store: JobStore):
        self._store = store

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        if arguments:
            raise InvalidArguments("list_schedules 不接收参数")
        return {}

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        jobs = [job for job in self._store.load() if job.enabled]
        lines = [f"定时任务列表（共 {len(jobs)} 个）："] if jobs else ["当前没有待执行的定时任务"]
        for job in jobs:
            action = job.message if job.tier == "instant" else f"[AI] {job.prompt}"
            assert action is not None
            label = f"「{job.name}」" if job.name else job.id[:8]
            display = job.fire_at.astimezone(ZoneInfo(job.timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")
            lines.append(f"• {label} [{job.tier}/{job.trigger}] 下次: {display} 内容: {action[:40]} 已运行: {job.run_count}次")
        return Result("success", (ContentPart("text", "\n".join(lines)),))

    async def query(self, key: str) -> Result | None:
        return None

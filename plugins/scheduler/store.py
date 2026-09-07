from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields as dataclass_fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Self, TypeVar, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.common.timekit import parse_iso as _parse_iso
from infra.persistence.json_store import atomic_save_json
from .schedule import ScheduledJob, SCHEDULE_MAX_ACTIVE_JOBS, is_cron_expr, next_cron_fire

_T = TypeVar("_T")
FireStatus = Literal["pending", "delivered", "failed", "cancelled"]


class Operation(BaseModel):
    """任务变更与工具回执共用一次文件提交；回放不再匹配新的任务。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["schedule", "cancel"]
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    job_ids: tuple[str, ...]
    outcome: Literal["success", "error"]
    response: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_ids(self) -> Self:
        if any(not identity for identity in self.job_ids) or len(set(self.job_ids)) != len(self.job_ids):
            raise ValueError("调度回执任务 ID 必须非空且唯一")
        if self.kind == "schedule" and len(self.job_ids) != int(self.outcome == "success"):
            raise ValueError("新增调度回执与任务数量不一致")
        return self


@dataclass(frozen=True)
class Fire:
    job: ScheduledJob
    status: FireStatus = "pending"
    error: str | None = None

    @property
    def key(self) -> str:
        return fire_key(self.job)

    @property
    def session_id(self) -> str:
        return "scheduler:" + self.key

    @property
    def notification_id(self) -> str:
        return "scheduler-notification:" + self.key


@dataclass
class ScheduleState:
    jobs: dict[str, ScheduledJob]
    operations: dict[str, Operation]
    fires: dict[str, Fire]


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fire_key(job: ScheduledJob) -> str:
    return _digest([job.id, aware(job.fire_at).isoformat()])


def aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def advance(job: ScheduledJob, after: datetime) -> datetime:
    """周期任务只推进到未来；沿既有名义时间计算，不以实际延迟累积漂移。"""
    if job.cron_expr:
        return next_cron_fire(job.cron_expr, job.timezone, after)
    assert job.interval_seconds is not None
    interval = timedelta(seconds=job.interval_seconds)
    start = aware(job.fire_at)
    return start + (max(0, (after - start) // interval) + 1) * interval

class JobStore:
    """Scheduler 独占任务、操作和触发事实；查询不迁移或修改文件。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[ScheduledJob]:
        return list(self.read().jobs.values())

    def read(self) -> ScheduleState:
        """原子 replace 保证读到完整一版；只允许缺失文件表示空状态。"""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ScheduleState({}, {}, {})
        return self.decode(self.parse(text))

    def parse(self, raw: str | bytes) -> object:
        return json.loads(raw, object_pairs_hook=self._object)

    def _object(self, pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"job_store JSON 字段重复：path={self.path} key={key}")
            result[key] = value
        return result

    def decode(self, raw: object) -> ScheduleState:
        """只接纳当前文件 schema；旧数组必须由 yoyo 在停写状态下迁移。"""
        if not isinstance(raw, dict):
            raise ValueError(f"job_store schema 无效，请先迁移：path={self.path}")
        raw = cast(dict[str, Any], raw)
        if (set(raw) != {"version", "jobs", "operations", "fires"}
                or type(raw["version"]) is not int or raw["version"] != 2):
            raise ValueError(f"job_store schema 无效，请先迁移：path={self.path}")
        jobs = self.decode_jobs(raw["jobs"])
        operations: dict[str, Operation] = {}
        if not isinstance(raw["operations"], dict) or not isinstance(raw["fires"], dict):
            raise ValueError(f"job_store 回执必须是对象：path={self.path}")
        for key, value in cast(dict[str, object], raw["operations"]).items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"job_store 操作 key 无效：path={self.path}")
            operations[key] = Operation.model_validate_json(json.dumps(value))
        fires: dict[str, Fire] = {}
        for index, (key, raw_fire) in enumerate(cast(dict[str, object], raw["fires"]).items()):
            if not isinstance(raw_fire, dict):
                raise ValueError(f"job_store 触发记录无效：path={self.path} key={key}")
            value = cast(dict[str, Any], raw_fire)
            if set(value) != {"job", "status", "error"}:
                raise ValueError(f"job_store 触发记录无效：path={self.path} key={key}")
            if not isinstance(value["job"], dict):
                raise ValueError(f"job_store 触发任务无效：path={self.path} key={key}")
            job = self.decode_job(cast(dict[str, Any], value["job"]), index=index)
            status, error = value["status"], value["error"]
            if (status not in ("pending", "delivered", "failed", "cancelled")
                    or (status in {"pending", "delivered"} and error is not None)
                    or (status in {"failed", "cancelled"} and (not isinstance(error, str) or not error))):
                raise ValueError(f"job_store 触发状态无效：path={self.path} key={key}")
            fire = Fire(job, cast(FireStatus, status), error)
            if key != fire.key:
                raise ValueError(f"job_store 触发身份不一致：path={self.path} key={key}")
            fires[key] = fire
        return ScheduleState({job.id: job for job in jobs}, operations, fires)

    def decode_jobs(self, raw: object) -> list[ScheduledJob]:
        """当前文档与显式旧数组迁移共用同一任务 schema owner。"""
        if not isinstance(raw, list):
            raise ValueError(f"job_store 任务必须是 JSON 列表：path={self.path}")

        # 3. 反序列化所有任务，损坏记录直接暴露
        jobs: list[ScheduledJob] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(cast(list[object], raw)):
            if not isinstance(item, dict):
                raise ValueError(
                    f"job_store 任务必须是 JSON 对象："
                    f"path={self.path} index={index} value={item!r}"
                )
            job = self.decode_job(cast(dict[str, Any], item), index=index)
            if job.id in seen_ids:
                raise ValueError(
                    f"job_store 任务 ID 重复：path={self.path} index={index} id={job.id!r}"
                )
            seen_ids.add(job.id)
            jobs.append(job)
        return jobs

    def save(self, jobs: dict[str, ScheduledJob]) -> None:
        def update(state: ScheduleState) -> None:
            state.jobs = dict(jobs)
        self._change(update)

    def encode(self, state: ScheduleState) -> dict[str, object]:
        """校验整个候选后发布，保留已经失效任务的操作与触发恢复证据。"""
        data: list[dict[str, Any]] = []
        for index, (job_id, job) in enumerate(state.jobs.items()):
            if job_id != job.id:
                raise ValueError(
                    f"job_store 任务 ID 与字典键不一致："
                    f"path={self.path} index={index} key={job_id!r} id={job.id!r}"
                )
            self._validate_job(job, index=index)
            data.append(self.encode_job(job))
        value: dict[str, object] = {
            "version": 2, "jobs": data,
            "operations": {key: operation.model_dump(mode="json") for key, operation in state.operations.items()},
            "fires": {key: {"job": self.encode_job(fire.job), "status": fire.status, "error": fire.error}
                      for key, fire in state.fires.items()},
        }
        _ = self.decode(value)
        return value

    def _change(self, change: Callable[[ScheduleState], _T]) -> _T:
        """不同 generation 重读同一文件再提交，不从各自缓存覆盖新事实。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_name(self.path.name + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.read()
            before = self.encode(state)
            result = change(state)
            after = self.encode(state)
            if before != after or not self.path.exists():
                atomic_save_json(self.path, after, domain="job_store")
            return result

    def _previous(self, state: ScheduleState, key: str, kind: str, digest: str) -> Operation | None:
        previous = state.operations.get(key)
        if previous is not None and (previous.kind, previous.digest) != (kind, digest):
            raise ValueError("调度操作 key 已用于不同请求")
        return previous

    def add(self, key: str, job: ScheduledJob, response: str) -> Operation:
        """新任务与操作结果同一次原子写；失败后的查询不会重新计算时间或 ID。"""
        digest = _digest(self.encode_job(job))
        def update(state: ScheduleState) -> Operation:
            previous = self._previous(state, key, "schedule", digest)
            if previous is not None:
                return previous
            if job.id in state.jobs:
                raise ValueError("新增调度任务 ID 已存在")
            active = sum(job.enabled for job in state.jobs.values())
            if active >= SCHEDULE_MAX_ACTIVE_JOBS:
                operation = Operation(kind="schedule", digest=digest, job_ids=(), outcome="error", response=(
                    f"schedule_capacity_reached：当前已有 {active} 个活动定时任务，"
                    f"默认上限为 {SCHEDULE_MAX_ACTIVE_JOBS} 个。请先确认要移除的任务。"))
            else:
                state.jobs[job.id] = job
                operation = Operation(kind="schedule", digest=digest, job_ids=(job.id,),
                                      outcome="success", response=response)
            state.operations[key] = operation
            return operation
        return self._change(update)

    def cancel(self, key: str, job_ids: tuple[str, ...]) -> Operation:
        """只取消 prepare 已固定的 ID；回放不会匹配后来出现的同名任务。"""
        digest = _digest(job_ids)
        def update(state: ScheduleState) -> Operation:
            previous = self._previous(state, key, "cancel", digest)
            if previous is not None:
                return previous
            removed = tuple(identity for identity in job_ids if identity in state.jobs)
            for identity in removed:
                del state.jobs[identity]
            for identity, fire in state.fires.items():
                if fire.job.id in job_ids and fire.status == "pending":
                    state.fires[identity] = replace(fire, status="cancelled", error="任务已被明确取消")
            operation = Operation(kind="cancel", digest=digest, job_ids=removed, outcome="success",
                                  response=f"已取消 {len(removed)} 个任务" if removed else "未找到匹配的任务")
            state.operations[key] = operation
            return operation
        return self._change(update)

    def start_fire(self, job: ScheduledJob) -> Fire | None:
        """计时到达后固定本次任务快照；模型和消息写入在提交成功后开始。"""
        def update(state: ScheduleState) -> Fire | None:
            current = state.jobs.get(job.id)
            if current is None or not current.enabled or fire_key(current) != fire_key(job):
                return None
            previous = state.fires.get(fire_key(job))
            if previous is not None:
                return previous if previous.status == "pending" else None
            fire = Fire(current)
            state.fires[fire.key] = fire
            return fire
        return self._change(update)

    def settle(self, key: str, status: Literal["delivered", "failed"], *, now: datetime,
               error: str | None = None) -> None:
        """送达只增加一次计数；周期与 one-shot 终态在同一候选提交。"""
        def update(state: ScheduleState) -> None:
            fire = state.fires[key]
            if fire.status != "pending":
                return
            state.fires[key] = replace(fire, status=status, error=error)
            current = state.jobs.get(fire.job.id)
            if current is None or fire_key(current) != key:
                return
            count = current.run_count + int(status == "delivered")
            state.jobs[current.id] = (
                replace(current, fire_at=advance(current, max(aware(now), aware(current.fire_at))), run_count=count)
                if current.trigger == "every" else replace(current, enabled=False, run_count=count)
            )
        self._change(update)

    def recover(self, now: datetime) -> ScheduleState:
        """已有触发先恢复；尚未触发的旧周期跳到未来，过期 one-shot 逻辑失效。"""
        now = aware(now)
        def update(state: ScheduleState) -> ScheduleState:
            for identity, job in state.jobs.items():
                if not job.enabled or aware(job.fire_at) > now or fire_key(job) in state.fires:
                    continue
                if job.trigger == "every":
                    state.jobs[identity] = replace(job, fire_at=advance(job, now))
                elif (now - aware(job.fire_at)).total_seconds() > 300:
                    state.jobs[identity] = replace(job, enabled=False)
            return state
        return self._change(update)

    # ── 私有方法 ──

    def encode_job(self, job: ScheduledJob) -> dict[str, Any]:
        d = asdict(job)
        d["fire_at"] = job.fire_at.isoformat()
        d["created_at"] = job.created_at.isoformat()
        return d

    def decode_job(self, d: dict[str, Any], *, index: int = 0) -> ScheduledJob:
        d = dict(d)
        expected = {item.name for item in dataclass_fields(ScheduledJob)}
        missing = sorted(expected.difference(d))
        extra = sorted(set(d).difference(expected))
        if missing or extra:
            raise ValueError(
                f"job_store 任务 schema 无效：path={self.path} index={index} "
                f"missing={missing} extra={extra}"
            )
        for field_name in ("fire_at", "created_at"):
            if field_name not in d:
                raise ValueError(
                    f"job_store 任务缺少必填字段："
                    f"path={self.path} index={index} field={field_name}"
                )
            d[field_name] = self._parse_dt(
                d[field_name], index=index, field_name=field_name
            )
        try:
            job = ScheduledJob(**d)
        except TypeError as e:
            raise ValueError(
                f"job_store 任务结构无效：path={self.path} index={index}"
            ) from e
        self._validate_job(job, index=index)
        return job

    def _validate_job(self, job: ScheduledJob, *, index: int) -> None:
        """校验持久化任务的不变量后再交给调度器使用。"""

        self._validate_job_identity(job, index=index)
        self._validate_job_payload(job, index=index)
        self._validate_job_schedule(job, index=index)

    def _validate_job_identity(self, job: ScheduledJob, *, index: int) -> None:
        """校验任务身份、渠道和 IANA 时区。"""

        # 1. 校验基础字段和枚举
        if job.trigger not in ("at", "after", "every"):
            raise self._schema_error(index, "trigger 无效")
        if job.tier not in ("instant", "soft"):
            raise self._schema_error(index, "tier 无效")
        if type(job.id) is not str or not job.id:
            raise self._schema_error(index, "id 无效")
        if type(job.channel) is not str or not job.channel:
            raise self._schema_error(index, "channel 无效")
        if type(job.chat_id) is not str or not job.chat_id:
            raise self._schema_error(index, "chat_id 无效")
        if type(job.timezone) is not str or not job.timezone:
            raise self._schema_error(index, "timezone 无效")
        try:
            _ = ZoneInfo(job.timezone)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise self._schema_error(
                index, f"timezone 无效 timezone={job.timezone!r}"
            ) from e

    def _validate_job_payload(self, job: ScheduledJob, *, index: int) -> None:
        """校验消息载荷、运行计数和可选字段类型。"""

        # 2. 校验消息、计数和可选字段类型
        if job.tier == "instant" and (type(job.message) is not str or not job.message):
            raise self._schema_error(index, "instant 任务缺少 message")
        if job.tier == "soft" and (type(job.prompt) is not str or not job.prompt):
            raise self._schema_error(index, "soft 任务缺少 prompt")
        if job.message is not None and type(job.message) is not str:
            raise self._schema_error(index, "message 类型无效")
        if job.prompt is not None and type(job.prompt) is not str:
            raise self._schema_error(index, "prompt 类型无效")
        if job.name is not None and type(job.name) is not str:
            raise self._schema_error(index, "name 类型无效")
        if type(job.run_count) is not int or job.run_count < 0:
            raise self._schema_error(index, "run_count 无效")
        if type(job.enabled) is not bool:
            raise self._schema_error(index, "enabled 类型无效")

    def _validate_job_schedule(self, job: ScheduledJob, *, index: int) -> None:
        """校验循环模式的互斥关系和正间隔。"""

        # 3. 校验循环模式的互斥关系和正间隔
        if job.trigger == "every":
            if (job.interval_seconds is None) == (job.cron_expr is None):
                raise self._schema_error(index, "every 任务必须且只能有一种周期")
            if job.interval_seconds is not None and (
                type(job.interval_seconds) is not int or job.interval_seconds <= 0
            ):
                raise self._schema_error(index, "interval_seconds 无效")
            if job.cron_expr is not None and (
                type(job.cron_expr) is not str or not is_cron_expr(job.cron_expr)
            ):
                raise self._schema_error(index, "cron_expr 无效")
        elif job.interval_seconds is not None or job.cron_expr is not None:
            raise self._schema_error(index, "非 every 任务不能带周期字段")

    def _schema_error(self, index: int, detail: str) -> ValueError:
        return ValueError(
            f"job_store 任务 schema 无效：path={self.path} index={index} {detail}"
        )

    def _parse_dt(self, s: str, *, index: int, field_name: str) -> datetime:
        if type(s) is not str:
            raise ValueError(
                f"job_store 时间字段不是字符串："
                f"path={self.path} index={index} field={field_name} value={s!r}"
            )
        parsed = _parse_iso(s)
        if parsed is None:
            raise ValueError(
                f"job_store 时间字段不是有效 ISO 8601："
                f"path={self.path} index={index} field={field_name} value={s!r}"
            )
        return parsed

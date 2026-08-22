"""
Scheduler: 定时任务核心模块

组件：
  LatencyTracker     — 自适应 P90 延迟估算（软实时预触发）
  parse_duration     — "30s" / "5m" / "2h" 等时长解析
  parse_when_at      — "14:30" / ISO datetime 解析
  is_cron_expr       — 判断是否是 cron 表达式
  compute_fire_at    — 计算首次触发时间（含 request_time 延迟补偿）
  compute_actual_trigger — 计算实际触发时间（SOFT 提前 P90）
  ScheduledJob       — 任务数据类
  JobStore           — JSON 持久化
"""

import json
import re
import statistics
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, cast

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.cron import CronTrigger

from core.common.timekit import parse_iso as _parse_iso
from infra.persistence.json_store import atomic_save_json

SCHEDULE_MAX_ACTIVE_JOBS = 10


class ScheduleCapacityError(RuntimeError):
    """表示新增任务会超过 workspace 全局活动任务上限。"""

    code = "schedule_capacity_reached"

    def __init__(self, *, active_jobs: int, max_active_jobs: int) -> None:
        self.active_jobs = active_jobs
        self.max_active_jobs = max_active_jobs
        super().__init__(self.code)


# ── LatencyTracker ───────────────────────────────────────────────


class LatencyTracker:
    """滑动窗口 P90 延迟追踪，用于 SOFT tier 预触发偏移量自适应。"""

    def __init__(self, default: float = 25.0, window: int = 20) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self.default = default

    def record(self, elapsed: float) -> None:
        self._samples.append(elapsed)

    @property
    def lead(self) -> float:
        """返回 P90 估算值；样本不足 3 个时返回 default。"""
        if len(self._samples) < 3:
            return self.default
        return statistics.quantiles(list(self._samples), n=10)[8]


# ── 时间解析 ─────────────────────────────────────────────────

_DURATION_RE = re.compile(r"^(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


def parse_duration(s: str) -> timedelta:
    """解析时长字符串，如 '30s', '5m', '2h', '1h30m', '1d2h'。"""
    s = s.strip()
    m = _DURATION_RE.match(s)
    if not m or not any(m.groups()):
        raise ValueError(f"无效的时间间隔: {s!r}，示例: '30s', '5m', '2h', '1h30m'")
    days, hours, minutes, seconds = (int(x or 0) for x in m.groups())
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def parse_when_at(
    s: str,
    tz: str = "UTC",
    _now_fn: Callable[[], datetime] | None = None,
) -> datetime:
    """解析 'at' 时间：HH:MM（自动判断今天/明天）或 ISO datetime。"""
    tzinfo = ZoneInfo(tz)
    now_fn = _now_fn or (lambda: datetime.now(tzinfo))
    s = s.strip()

    # HH:MM 格式
    if re.match(r"^\d{1,2}:\d{2}$", s):
        now = now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=tzinfo)
        else:
            now = now.astimezone(tzinfo)
        t = datetime.strptime(s, "%H:%M").time()
        dt = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        if dt <= now:
            dt += timedelta(days=1)
        return dt

    # ISO datetime 格式
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo)
        return dt
    except ValueError:
        pass

    raise ValueError(f"无法解析时间: {s!r}，示例: '14:30', '2025-06-01T09:00'")


def is_cron_expr(s: str) -> bool:
    """判断字符串是否是 cron 表达式（5 或 6 字段）。"""
    parts = s.strip().split()
    return len(parts) in (5, 6)


def next_cron_fire(cron_expr: str, tz: str, after: datetime) -> datetime:
    """用 APScheduler CronTrigger 计算 cron 下次触发时间。"""
    parts = cron_expr.strip().split()
    if len(parts) not in (5, 6):
        raise ValueError(f"无效的 cron 表达式: {cron_expr!r}")
    tzinfo = ZoneInfo(tz)

    if len(parts) == 5:
        minute_s, hour_s, dom_s, month_s, dow_s = parts
        trigger = CronTrigger(
            minute=minute_s,
            hour=hour_s,
            day=dom_s,
            month=month_s,
            day_of_week=dow_s,
            timezone=tzinfo,
        )
    else:
        second_s, minute_s, hour_s, dom_s, month_s, dow_s = parts
        trigger = CronTrigger(
            second=second_s,
            minute=minute_s,
            hour=hour_s,
            day=dom_s,
            month=month_s,
            day_of_week=dow_s,
            timezone=tzinfo,
        )
    result = trigger.get_next_fire_time(None, after)
    if result is not None and result <= after:
        result = trigger.get_next_fire_time(result, after)
    if result is None:
        raise ValueError(f"无效的 cron 表达式: {cron_expr!r}")
    # 将结果规范为带 UTC 时区的 datetime
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


# ── fire_at 计算 ──────────────────────────────────────────


def compute_fire_at(
    trigger: str,
    when: str,
    tz: str = "UTC",
    request_time: str | None = None,
    _now_fn: Callable[[], datetime] | None = None,
) -> datetime:
    """
    计算首次触发时间。

    after 模式：以 request_time（用户消息到达时间）为基准，
                补偿 AI 推理延迟，确保 fire_at 从用户视角算起。
    """
    tzinfo = ZoneInfo(tz)
    now_fn = _now_fn or (lambda: datetime.now(tzinfo))

    if trigger == "at":
        return parse_when_at(when, tz, _now_fn)

    if trigger == "after":
        duration = parse_duration(when)
        if request_time:
            base = datetime.fromisoformat(request_time)
            if base.tzinfo is None:
                base = base.replace(tzinfo=tzinfo)
        else:
            base = now_fn()
        return base + duration

    if trigger == "every":
        if is_cron_expr(when):
            return next_cron_fire(when, tz, now_fn())
        interval = parse_duration(when)
        return now_fn() + interval

    raise ValueError(f"未知触发类型: {trigger!r}，须为 at/after/every")


def compute_actual_trigger(
    fire_at: datetime,
    tier: str,
    tracker: LatencyTracker,
) -> datetime:
    """
    计算实际触发时刻。

    INSTANT: 等于 fire_at（直接推送，无 AI 延迟）
    SOFT:    fire_at - P90（提前触发 AI，让 AI 在 fire_at 前完成处理）
    """
    if tier == "instant":
        return fire_at
    return fire_at - timedelta(seconds=tracker.lead)


# ── ScheduledJob ─────────────────────────────────────────────────


@dataclass
class ScheduledJob:
    trigger: str  # "at" | "after" | "every"
    tier: str  # "instant" | "soft"
    fire_at: datetime  # 下次名义触发时间（UTC-aware）
    channel: str
    chat_id: str

    interval_seconds: int | None = None  # every + interval 模式
    cron_expr: str | None = None  # every + cron 模式

    message: str | None = None  # instant 层
    prompt: str | None = None  # soft 层

    name: str | None = None
    timezone: str = "UTC"

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_count: int = 0
    enabled: bool = True
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ── JobStore ─────────────────────────────────────────────────────


class JobStore:
    """JSON 文件持久化，读写 ScheduledJob 列表。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[ScheduledJob]:
        # 1. 未创建的存储按空调度处理
        if not self.path.exists():
            return []

        # 2. 严格读取并解析，保留 I/O 与 JSON 原始异常
        raw: object = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(
                f"job_store 必须是 JSON 列表：path={self.path} value={raw!r}"
            )

        # 3. 反序列化所有任务，损坏记录直接暴露
        jobs: list[ScheduledJob] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(cast(list[object], raw)):
            if not isinstance(item, dict):
                raise ValueError(
                    f"job_store 任务必须是 JSON 对象："
                    f"path={self.path} index={index} value={item!r}"
                )
            job = self._from_dict(cast(dict[str, Any], item), index=index)
            if job.id in seen_ids:
                raise ValueError(
                    f"job_store 任务 ID 重复：path={self.path} index={index} id={job.id!r}"
                )
            seen_ids.add(job.id)
            jobs.append(job)
        return jobs

    def save(self, jobs: dict[str, ScheduledJob]) -> None:
        # 1. 保存前再次确认内存映射仍满足持久化 schema
        data: list[dict[str, Any]] = []
        for index, (job_id, job) in enumerate(jobs.items()):
            if job_id != job.id:
                raise ValueError(
                    f"job_store 任务 ID 与字典键不一致："
                    f"path={self.path} index={index} key={job_id!r} id={job.id!r}"
                )
            self._validate_job(job, index=index)
            data.append(self._to_dict(job))

        # 2. 只在完整校验通过后原子写入
        atomic_save_json(self.path, data, domain="job_store")

    # ── 私有方法 ──

    def _to_dict(self, job: ScheduledJob) -> dict[str, Any]:
        d = asdict(job)
        d["fire_at"] = job.fire_at.isoformat()
        d["created_at"] = job.created_at.isoformat()
        return d

    def _from_dict(self, d: dict[str, Any], *, index: int) -> ScheduledJob:
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

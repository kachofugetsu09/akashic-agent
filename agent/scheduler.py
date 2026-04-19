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
  SchedulerService   — 主调度服务（asyncio tick 循环）
"""

import asyncio
from importlib import import_module
import logging
import re
import statistics
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from zoneinfo import ZoneInfo

from core.common.timekit import parse_iso as _parse_iso
from infra.persistence.json_store import load_json, save_json

logger = logging.getLogger(__name__)


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


# ── Time Parsing ─────────────────────────────────────────────────

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


def _parse_cron_field(field: str, minimum: int, maximum: int) -> set[int]:
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            step = int(step_str)
            if step <= 0:
                raise ValueError(f"无效 cron step: {field!r}")
        if part == "*":
            start, end = minimum, maximum
        elif "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
        else:
            start = end = int(part)
        if start < minimum or end > maximum or start > end:
            raise ValueError(f"无效 cron 字段: {field!r}")
        values.update(range(start, end + 1, step))
    if not values:
        raise ValueError(f"无效 cron 字段: {field!r}")
    return values


def _next_cron_fire_fallback(cron_expr: str, tz: str, after: datetime) -> datetime:
    parts = cron_expr.strip().split()
    if len(parts) == 5:
        second_values = {0}
        minute_s, hour_s, dom_s, month_s, dow_s = parts
        step = timedelta(minutes=1)
        current = after.astimezone(ZoneInfo(tz)).replace(second=0, microsecond=0)
        if current <= after.astimezone(ZoneInfo(tz)):
            current += step
    elif len(parts) == 6:
        second_s, minute_s, hour_s, dom_s, month_s, dow_s = parts
        second_values = _parse_cron_field(second_s, 0, 59)
        step = timedelta(seconds=1)
        current = after.astimezone(ZoneInfo(tz)).replace(microsecond=0) + step
    else:
        raise ValueError(f"无效的 cron 表达式: {cron_expr!r}")

    minute_values = _parse_cron_field(minute_s, 0, 59)
    hour_values = _parse_cron_field(hour_s, 0, 23)
    dom_values = _parse_cron_field(dom_s, 1, 31)
    month_values = _parse_cron_field(month_s, 1, 12)
    dow_values = _parse_cron_field(dow_s.replace("7", "0"), 0, 6)

    for _ in range(366 * 24 * 60 * (60 if len(parts) == 6 else 1)):
        cron_dow = (current.weekday() + 1) % 7
        if (
            current.second in second_values
            and current.minute in minute_values
            and current.hour in hour_values
            and current.day in dom_values
            and current.month in month_values
            and cron_dow in dow_values
        ):
            return current.astimezone(timezone.utc)
        current += step
    raise ValueError(f"无法在合理范围内解析 cron 表达式: {cron_expr!r}")


def next_cron_fire(cron_expr: str, tz: str, after: datetime) -> datetime:
    """用 APScheduler CronTrigger 计算 cron 下次触发时间。"""
    try:
        from apscheduler.triggers.cron import CronTrigger
    except ModuleNotFoundError:
        return _next_cron_fire_fallback(cron_expr, tz, after)

    # APScheduler 3.x 兼容：优先用 pytz，回退到 ZoneInfo
    try:
        pytz = import_module("pytz")
        tzinfo = pytz.timezone(tz)
    except Exception:
        tzinfo = ZoneInfo(tz)

    trigger = CronTrigger.from_crontab(cron_expr, timezone=tzinfo)
    result = trigger.get_next_fire_time(None, after)
    if result is None:
        raise ValueError(f"无效的 cron 表达式: {cron_expr!r}")
    # Normalize to UTC-aware datetime
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


# ── fire_at Computation ──────────────────────────────────────────


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

    message: str | None = None  # instant tier
    prompt: str | None = None  # soft tier

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
        # 1. 读取原始列表
        raw = load_json(self.path, default=[], domain="job_store")

        # 2. 反序列化
        try:
            return [self._from_dict(d) for d in raw]
        except Exception as e:
            logger.warning("[job_store] 反序列化失败: %s", e)
            return []

    def save(self, jobs: dict[str, ScheduledJob]) -> None:
        data = [self._to_dict(j) for j in jobs.values()]
        save_json(self.path, data, domain="job_store")

    # ── private ──

    def _to_dict(self, job: ScheduledJob) -> dict[str, Any]:
        d = asdict(job)
        d["fire_at"] = job.fire_at.isoformat()
        d["created_at"] = job.created_at.isoformat()
        return d

    def _from_dict(self, d: dict[str, Any]) -> ScheduledJob:
        d = dict(d)
        d["fire_at"] = self._parse_dt(d["fire_at"])
        d["created_at"] = self._parse_dt(d["created_at"])
        return ScheduledJob(**d)

    @staticmethod
    def _parse_dt(s: str) -> datetime:
        return _parse_iso(s) or datetime.now(timezone.utc)


# ── SchedulerService ─────────────────────────────────────────────


class SchedulerService:
    """
    asyncio 定时任务服务。

    - 每秒 tick 一次，检查 actual_trigger <= now 的 job
    - INSTANT: 直接 message_push
    - SOFT: process_direct + 记录延迟 + push 响应
    - 持久化到 JSON，重启后自动恢复
    """

    GRACE_SECONDS = 300  # 5分钟内的 misfire 仍执行

    def __init__(
        self,
        store_path: Path,
        push_tool: Any,
        agent_loop: Any = None,
        agent_loop_provider: Callable[[], Any] | None = None,
        tracker: LatencyTracker | None = None,
        _now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = JobStore(store_path)
        self.push_tool = push_tool
        self.agent_loop = agent_loop
        self._agent_loop_provider = agent_loop_provider
        self.tracker = tracker or LatencyTracker()
        self._now = _now_fn or (lambda: datetime.now(timezone.utc))
        self._jobs: dict[str, ScheduledJob] = {}
        self._in_flight: set[str] = set()
        self._running = False

    # ── Public API ───────────────────────────────────────────────

    async def run(self) -> None:
        self.load_and_recover()
        self._running = True
        logger.info("SchedulerService started")
        while self._running:
            await asyncio.sleep(1)
            await self._tick()

    def stop(self) -> None:
        self._running = False

    def add_job(self, job: ScheduledJob) -> None:
        # Ensure fire_at is UTC-aware
        if job.fire_at.tzinfo is None:
            job.fire_at = job.fire_at.replace(tzinfo=timezone.utc)
        self._jobs[job.id] = job
        self.store.save(self._jobs)
        logger.info(
            f"Job added: {job.id[:8]} tier={job.tier} trigger={job.trigger} "
            f"fire_at={job.fire_at.isoformat()}"
        )

    def cancel_job(self, job_id: str) -> bool:
        if job_id not in self._jobs:
            return False
        del self._jobs[job_id]
        self.store.save(self._jobs)
        return True

    def cancel_job_by_name(self, name: str) -> list[str]:
        cancelled = [jid for jid, j in self._jobs.items() if j.name == name]
        for jid in cancelled:
            del self._jobs[jid]
        if cancelled:
            self.store.save(self._jobs)
        return cancelled

    def list_jobs(self) -> list[ScheduledJob]:
        return list(self._jobs.values())

    def load_and_recover(self) -> None:
        """启动时加载持久化 jobs，处理 misfire。"""
        now = self._now()
        jobs = self.store.load()
        count_loaded = 0

        for job in jobs:
            if not job.enabled:
                continue

            if job.fire_at.tzinfo is None:
                job.fire_at = job.fire_at.replace(tzinfo=timezone.utc)

            if job.fire_at <= now:
                age = (now - job.fire_at).total_seconds()
                if job.trigger == "every":
                    # 推进到下一个未来时间
                    job.fire_at = self._advance_every(job, now)
                    self._jobs[job.id] = job
                    count_loaded += 1
                elif age <= self.GRACE_SECONDS:
                    # 在宽限期内，保留（下次 tick 会执行）
                    self._jobs[job.id] = job
                    count_loaded += 1
                else:
                    logger.info(
                        f"Job {job.id[:8]} ({job.name or 'unnamed'}) expired "
                        f"{age:.0f}s ago, beyond grace period — discarded"
                    )
            else:
                self._jobs[job.id] = job
                count_loaded += 1

        logger.info(f"SchedulerService recovered {count_loaded} jobs")

    # ── Internal ────────────────────────────────────────────────

    async def _tick(self) -> None:
        now = self._now()
        for job in list(self._jobs.values()):
            if not job.enabled or job.id in self._in_flight:
                continue
            actual_trigger = compute_actual_trigger(job.fire_at, job.tier, self.tracker)
            if actual_trigger <= now:
                label = job.name or job.id[:8]
                logger.info(
                    f"[scheduler] 触发任务 {label!r}  tier={job.tier}  channel={job.channel}:{job.chat_id}"
                )
                self._in_flight.add(job.id)
                asyncio.create_task(self._execute_and_reschedule(job))

    async def _execute_and_reschedule(self, job: ScheduledJob) -> None:
        try:
            await self._execute(job)
            job.run_count += 1
        except Exception as e:
            logger.error(f"Job {job.id[:8]} execution failed: {e}", exc_info=True)
        finally:
            self._in_flight.discard(job.id)
            now = self._now()
            if job.trigger == "every":
                # SOFT recurring jobs may execute before nominal fire_at.
                # Reschedule strictly after the later of "now" and the nominal
                # boundary, otherwise cron jobs can re-fire the same occurrence
                # repeatedly until wall clock passes fire_at.
                reschedule_after = max(now, job.fire_at) + timedelta(microseconds=1)
                job.fire_at = self._advance_every(job, reschedule_after)
                self._jobs[job.id] = job
            else:
                self._jobs.pop(job.id, None)
            self.store.save(self._jobs)

    async def _execute(self, job: ScheduledJob) -> None:
        label = job.name or job.id[:8]
        if job.tier == "instant":
            result = await self.push_tool.execute(
                channel=job.channel,
                chat_id=job.chat_id,
                message=job.message,
            )
            logger.info(f"[scheduler] instant 推送完成 {label!r}: {result}")
        else:
            loop = self._get_agent_loop()
            t0 = time.monotonic()
            content = await loop.process_direct(
                content=job.prompt,
                channel=job.channel,
                chat_id=job.chat_id,
                session_key=f"scheduler:{job.id}",
                omit_user_turn=True,
            )
            elapsed = time.monotonic() - t0
            self.tracker.record(elapsed)
            logger.info(
                f"[scheduler] soft AI 完成 {label!r}  耗时={elapsed:.1f}s  P90={self.tracker.lead:.1f}s"
            )
            if content:
                result = await self.push_tool.execute(
                    channel=job.channel,
                    chat_id=job.chat_id,
                    message=content,
                )
                logger.info(f"[scheduler] soft 推送完成 {label!r}: {result}")
            else:
                logger.warning(f"[scheduler] soft AI 返回空内容 {label!r}，跳过推送")

    def _get_agent_loop(self) -> Any:
        loop = self._agent_loop_provider() if self._agent_loop_provider else self.agent_loop
        if loop is None:
            raise RuntimeError("scheduler soft job requires agent_loop")
        return loop

    def _advance_every(self, job: ScheduledJob, after: datetime) -> datetime:
        """将 every job 的 fire_at 推进到 after 之后的下一个触发时间。"""
        if job.cron_expr:
            return next_cron_fire(job.cron_expr, job.timezone, after)
        interval = timedelta(seconds=job.interval_seconds or 3600)
        next_fire = job.fire_at + interval
        while next_fire <= after:
            next_fire += interval
        return next_fire

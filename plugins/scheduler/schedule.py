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
"""

import re
import statistics
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger


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



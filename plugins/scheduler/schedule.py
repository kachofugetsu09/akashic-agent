"""
Scheduler: 定时任务核心模块

组件：
  parse_duration     — "30s" / "5m" / "2h" 等时长解析
  parse_when_at      — "14:30" / ISO datetime 解析
  is_cron_expr       — 判断是否是 cron 表达式
  compute_fire_at    — 计算首次触发时间（含 request_time 延迟补偿）
  ScheduledJob       — 任务数据类
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger


SCHEDULE_MAX_ACTIVE_JOBS = 10


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


# ── ScheduledJob ─────────────────────────────────────────────────


@dataclass
class ScheduledJob:
    trigger: str  # "at" | "after" | "every"
    action: str  # "notify" 原文推送 | "run" 在目标会话里执行一轮
    fire_at: datetime  # 下次名义触发时间（UTC-aware）
    channel: str
    chat_id: str
    session_id: str  # 目标会话；run 的过程与结论都写在这里
    text: str  # notify 的推送原文，或 run 的任务指令

    interval_seconds: int | None = None  # every + interval 模式
    cron_expr: str | None = None  # every + cron 模式
    context: str = "none"  # run 是否带会话历史："none" | "session"

    name: str | None = None
    timezone: str = "UTC"

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_count: int = 0
    enabled: bool = True
    id: str = field(default_factory=lambda: str(uuid.uuid4()))



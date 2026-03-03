"""
core.common.timekit — 统一时间工具集。

替代散落在各模块的 _utcnow() / _parse_iso() / datetime.now(...) 调用。
所有内部逻辑以 UTC 存储，展示层按需本地化。
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

__all__ = [
    "utcnow",
    "parse_iso",
    "safe_zone",
    "format_iso",
    "local_now",
]


def utcnow() -> datetime:
    """返回当前 UTC 时间（带 tzinfo）。"""
    return datetime.now(timezone.utc)


def parse_iso(ts: str | None) -> datetime | None:
    """
    解析 ISO 8601 字符串，返回带时区的 datetime。

    - 兼容末尾 'Z' 写法（替换为 +00:00）。
    - 无时区信息时默认视为 UTC。
    - 解析失败返回 None。
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def format_iso(dt: datetime) -> str:
    """将 datetime 序列化为 ISO 8601 字符串（带时区）。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def safe_zone(name: str, *, logger=None) -> ZoneInfo:
    """
    解析时区名称，非法时区回退到 UTC。

    Args:
        name: IANA 时区名，如 "Asia/Shanghai"。
        logger: 可选，传入模块 logger 以输出 warning。
    """
    try:
        return ZoneInfo(name)
    except Exception:
        if logger is not None:
            logger.warning("[timekit] 无效时区 %r，回退到 UTC", name)
        return ZoneInfo("UTC")


def local_now(tz: ZoneInfo | str | None = None) -> datetime:
    """
    返回本地化当前时间。

    Args:
        tz: 时区对象或名称字符串，None 时使用系统本地时区。
    """
    if tz is None:
        return datetime.now().astimezone()
    if isinstance(tz, str):
        tz = safe_zone(tz)
    return datetime.now(tz)

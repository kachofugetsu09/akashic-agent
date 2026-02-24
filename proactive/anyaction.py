from __future__ import annotations

import json
import logging
import math
import random as _random_module
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def _safe_zone(name: str) -> ZoneInfo:
    """解析时区，非法时区回退到 UTC 并记录警告。"""
    try:
        return ZoneInfo(name)
    except Exception:
        logger.warning("[anyaction] 无效时区 %r，回退到 UTC", name)
        return ZoneInfo("UTC")


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


@dataclass
class QuotaSnapshot:
    window_key: str
    next_reset_at: datetime
    used: int
    last_action_at: datetime | None


class QuotaStore:
    """持久化每日动作计数，支持按本地时区 + 指定 reset_hour 刷新。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def snapshot(self, *, now_utc: datetime, reset_hour: int, timezone_name: str) -> QuotaSnapshot:
        tz = _safe_zone(timezone_name)
        window_key, next_reset_at_local = self._window_meta(now_utc, reset_hour, tz)
        self._rollover_if_needed(window_key=window_key, next_reset_at=next_reset_at_local.astimezone(timezone.utc))
        return QuotaSnapshot(
            window_key=self._state["window_key"],
            next_reset_at=_parse_iso(self._state["next_reset_at"]) or now_utc,
            used=int(self._state["used"]),
            last_action_at=_parse_iso(self._state.get("last_action_at")),
        )

    def record_action(self, *, now_utc: datetime, reset_hour: int, timezone_name: str) -> None:
        snap = self.snapshot(now_utc=now_utc, reset_hour=reset_hour, timezone_name=timezone_name)
        self._state["window_key"] = snap.window_key
        self._state["next_reset_at"] = snap.next_reset_at.isoformat()
        self._state["used"] = int(self._state.get("used", 0)) + 1
        self._state["last_action_at"] = now_utc.isoformat()
        self._save()

    def _window_meta(self, now_utc: datetime, reset_hour: int, tz: ZoneInfo) -> tuple[str, datetime]:
        local_now = now_utc.astimezone(tz)
        reset_today = local_now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
        if local_now >= reset_today:
            start = reset_today
            next_reset = reset_today + timedelta(days=1)
        else:
            start = reset_today - timedelta(days=1)
            next_reset = reset_today
        key = f"{start.date().isoformat()}@{reset_hour:02d}@{tz.key}"
        return key, next_reset

    def _rollover_if_needed(self, *, window_key: str, next_reset_at: datetime) -> None:
        if self._state.get("window_key") == window_key:
            return
        self._state = {
            "version": 1,
            "window_key": window_key,
            "next_reset_at": next_reset_at.isoformat(),
            "used": 0,
            "last_action_at": self._state.get("last_action_at"),
        }
        self._save()

    def _load(self) -> dict:
        if not self.path.exists():
            return {
                "version": 1,
                "window_key": "",
                "next_reset_at": "",
                "used": 0,
                "last_action_at": "",
            }
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return {
                "version": int(raw.get("version", 1)),
                "window_key": str(raw.get("window_key", "")),
                "next_reset_at": str(raw.get("next_reset_at", "")),
                "used": int(raw.get("used", 0)),
                "last_action_at": str(raw.get("last_action_at", "")),
            }
        except Exception:
            return {
                "version": 1,
                "window_key": "",
                "next_reset_at": "",
                "used": 0,
                "last_action_at": "",
            }

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


class AnyActionGate:
    """后台 AnyAction 通用层：硬规则 + 概率门。"""

    def __init__(self, *, cfg, quota_store: QuotaStore, rng: _random_module.Random | None = None) -> None:
        self._cfg = cfg
        self._quota = quota_store
        self._rng = rng

    def should_act(self, *, now_utc: datetime, last_user_at: datetime | None) -> tuple[bool, dict[str, float | int | str]]:
        snap = self._quota.snapshot(
            now_utc=now_utc,
            reset_hour=self._cfg.anyaction_reset_hour_local,
            timezone_name=self._cfg.anyaction_timezone,
        )
        remaining = max(0, self._cfg.anyaction_daily_max_actions - snap.used)
        if remaining <= 0:
            return False, {
                "reason": "quota_exhausted",
                "used_today": snap.used,
                "remaining_today": remaining,
            }

        if snap.last_action_at is not None:
            since_last = (now_utc - snap.last_action_at).total_seconds()
            if since_last < self._cfg.anyaction_min_interval_seconds:
                return False, {
                    "reason": "min_interval",
                    "used_today": snap.used,
                    "remaining_today": remaining,
                    "seconds_since_last_action": max(0.0, since_last),
                }

        idle_min = (
            max(0.0, (now_utc - last_user_at).total_seconds() / 60.0)
            if last_user_at is not None
            else self._cfg.anyaction_idle_scale_minutes * 2.0
        )
        idle_factor = 1.0 - math.exp(-idle_min / max(self._cfg.anyaction_idle_scale_minutes, 1.0))
        local_hour = now_utc.astimezone(_safe_zone(self._cfg.anyaction_timezone)).hour
        time_factor = self._time_factor(local_hour)
        p = self._cfg.anyaction_probability_min + (
            self._cfg.anyaction_probability_max - self._cfg.anyaction_probability_min
        ) * idle_factor * time_factor
        p = max(0.0, min(1.0, p))
        draw = (self._rng or _random_module).random()
        return draw < p, {
            "reason": "probability",
            "used_today": snap.used,
            "remaining_today": remaining,
            "idle_minutes": idle_min,
            "time_factor": time_factor,
            "p_act": p,
            "draw": draw,
        }

    def record_action(self, *, now_utc: datetime) -> None:
        self._quota.record_action(
            now_utc=now_utc,
            reset_hour=self._cfg.anyaction_reset_hour_local,
            timezone_name=self._cfg.anyaction_timezone,
        )

    @staticmethod
    def _time_factor(local_hour: int) -> float:
        if 9 <= local_hour < 23:
            return 1.0
        if 7 <= local_hour < 9:
            return 0.75
        if 23 <= local_hour or local_hour < 2:
            return 0.55
        return 0.35

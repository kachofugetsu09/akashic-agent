from __future__ import annotations

import json
import logging
import math
import random as _random_module
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Never, TypedDict, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.common.timekit import parse_iso as _parse_iso, safe_zone as _safe_zone
from infra.persistence.json_store import atomic_save_json

logger = logging.getLogger(__name__)


@dataclass
class QuotaSnapshot:
    window_key: str
    next_reset_at: datetime
    used: int
    last_action_at: datetime | None


class QuotaState(TypedDict):
    version: int
    window_key: str
    next_reset_at: str
    used: int
    last_action_at: str


class QuotaStore:
    """持久化每日动作计数，支持按本地时区 + 指定 reset_hour 刷新。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: QuotaState = self._load()

    def snapshot(
        self, *, now_utc: datetime, reset_hour: int, timezone_name: str
    ) -> QuotaSnapshot:
        tz = _safe_zone(timezone_name)
        window_key, next_reset_at_local = self._window_meta(now_utc, reset_hour, tz)
        self._rollover_if_needed(
            window_key=window_key,
            next_reset_at=next_reset_at_local.astimezone(timezone.utc),
        )
        return QuotaSnapshot(
            window_key=self._state["window_key"],
            next_reset_at=_parse_iso(self._state["next_reset_at"]) or now_utc,
            used=int(self._state["used"]),
            last_action_at=_parse_iso(self._state.get("last_action_at")),
        )

    def record_action(
        self, *, now_utc: datetime, reset_hour: int, timezone_name: str
    ) -> None:
        snap = self.snapshot(
            now_utc=now_utc, reset_hour=reset_hour, timezone_name=timezone_name
        )
        self._state["window_key"] = snap.window_key
        self._state["next_reset_at"] = snap.next_reset_at.isoformat()
        self._state["used"] = int(self._state.get("used", 0)) + 1
        self._state["last_action_at"] = now_utc.isoformat()
        self._save()

    def _window_meta(
        self, now_utc: datetime, reset_hour: int, tz: ZoneInfo
    ) -> tuple[str, datetime]:
        local_now = now_utc.astimezone(tz)
        reset_today = local_now.replace(
            hour=reset_hour, minute=0, second=0, microsecond=0
        )
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

    def _load(self) -> QuotaState:
        # 1. 只有文件缺失时初始化新状态
        if not self.path.exists():
            return {
                "version": 1,
                "window_key": "",
                "next_reset_at": "",
                "used": 0,
                "last_action_at": "",
            }

        # 2. 严格读取已有状态，保留 JSON 与 I/O 原始异常
        raw_data: object = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            raise ValueError(
                f"anyaction quota 必须是 JSON 对象：path={self.path}"
            )
        raw = cast(dict[str, object], raw_data)

        # 3. 验证写侧 version=1 的完整 schema
        required = {
            "version",
            "window_key",
            "next_reset_at",
            "used",
            "last_action_at",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(
                f"anyaction quota 缺少字段：path={self.path} fields={missing}"
            )
        version = raw["version"]
        used = raw["used"]
        if type(version) is not int or version != 1:
            self._raise_invalid_field("version", version)
        if type(used) is not int or used < 0:
            self._raise_invalid_field("used", used)
        window_key = self._validate_window_key(raw["window_key"])
        next_reset_at = self._validate_time("next_reset_at", raw["next_reset_at"])
        last_action_at = self._validate_time(
            "last_action_at", raw["last_action_at"], allow_empty=True
        )
        return {
            "version": version,
            "window_key": window_key,
            "next_reset_at": next_reset_at,
            "used": used,
            "last_action_at": last_action_at,
        }

    def _validate_window_key(self, value: object) -> str:
        if not isinstance(value, str):
            self._raise_invalid_field("window_key", value)
        try:
            date_text, hour_text, timezone_name = value.split("@", 2)
            _ = date.fromisoformat(date_text)
            if len(hour_text) != 2 or not hour_text.isdigit():
                raise ValueError
            hour = int(hour_text)
            if not 0 <= hour <= 23:
                raise ValueError
            _ = ZoneInfo(timezone_name)
        except (ValueError, ZoneInfoNotFoundError):
            self._raise_invalid_field("window_key", value)
        return value

    def _validate_time(
        self, field_name: str, value: object, *, allow_empty: bool = False
    ) -> str:
        if not isinstance(value, str) or (not value and not allow_empty):
            self._raise_invalid_field(field_name, value)
        if value:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                self._raise_invalid_field(field_name, value)
            if parsed.tzinfo is None:
                self._raise_invalid_field(field_name, value)
        return value

    def _raise_invalid_field(self, field_name: str, value: object) -> Never:
        raise ValueError(
            f"anyaction quota 字段无效：path={self.path} "
            f"field={field_name} value={value!r}"
        )

    def _save(self) -> None:
        atomic_save_json(self.path, self._state, domain="anyaction.quota")


class AnyActionGate:
    """后台 AnyAction 通用层：硬规则 + 概率门。"""

    def __init__(
        self, *, cfg, quota_store: QuotaStore, rng: _random_module.Random | None = None
    ) -> None:
        self._cfg = cfg
        self._quota = quota_store
        self._rng = rng

    def should_act(
        self, *, now_utc: datetime, last_user_at: datetime | None
    ) -> tuple[bool, dict[str, float | int | str]]:
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
        idle_factor = 1.0 - math.exp(
            -idle_min / max(self._cfg.anyaction_idle_scale_minutes, 1.0)
        )
        p = (
            self._cfg.anyaction_probability_min
            + (
                self._cfg.anyaction_probability_max
                - self._cfg.anyaction_probability_min
            )
            * idle_factor
        )
        p = max(0.0, min(1.0, p))
        draw = (self._rng or _random_module).random()
        return draw < p, {
            "reason": "probability",
            "used_today": snap.used,
            "remaining_today": remaining,
            "idle_minutes": idle_min,
            "p_act": p,
            "draw": draw,
        }

    def record_action(self, *, now_utc: datetime) -> None:
        self._quota.record_action(
            now_utc=now_utc,
            reset_hour=self._cfg.anyaction_reset_hour_local,
            timezone_name=self._cfg.anyaction_timezone,
        )

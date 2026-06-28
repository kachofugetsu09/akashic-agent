from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent.plugins import Plugin
from proactive_v2.frame import ProactiveFrame
from pydantic import BaseModel, Field, field_validator


class DayNightGateConfig(BaseModel):
    enabled: bool = True
    timezone: str = "Asia/Shanghai"
    start: str = "00:00"
    end: str = "06:00"
    pass_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    reason: str = "quiet_hours"

    @field_validator("start", "end")
    @classmethod
    def validate_time(cls, value: str) -> str:
        _ = _parse_hhmm(value)
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            _ = ZoneInfo(value)
        except ZoneInfoNotFoundError as e:
            raise ValueError(f"未知时区: {value}") from e
        return value


class DayNightGateModule:
    slot = "proactive.gate.daynight"
    phase = "proactive.gate"

    def __init__(self, config: DayNightGateConfig) -> None:
        self._config = config
        self._zone = ZoneInfo(config.timezone)
        self._start = _parse_hhmm(config.start)
        self._end = _parse_hhmm(config.end)

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        if not self._config.enabled:
            return frame
        local_now = frame.input.started_at.astimezone(self._zone)
        if not _in_window(local_now.time(), self._start, self._end):
            return frame
        current = frame.slots.get("proactive:gate:pass_probability")
        pass_probability = self._config.pass_probability
        if current is not None:
            pass_probability = min(float(current), pass_probability)
        frame.slots["proactive:gate:pass_probability"] = pass_probability
        frame.slots["proactive:gate:reason"] = self._config.reason
        frame.slots["proactive:effect:daynight_gate"] = {
            "provider_name": "daynight_gate",
            "pass_probability": pass_probability,
            "reason": self._config.reason,
            "timezone": self._config.timezone,
            "start": self._config.start,
            "end": self._config.end,
        }
        return frame


class DayNightGatePlugin(Plugin):
    name = "daynight_gate"
    ConfigModel = DayNightGateConfig

    async def initialize(self) -> None:
        return None

    async def terminate(self) -> None:
        return None

    def proactive_modules(self) -> list[object]:
        config = self.context.config
        if not isinstance(config, DayNightGateConfig):
            config = DayNightGateConfig()
        return [DayNightGateModule(config)]


def _parse_hhmm(value: str) -> time:
    hour_text, minute_text = value.split(":", 1)
    return time(hour=int(hour_text), minute=int(minute_text))


def _in_window(value: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= value < end
    return value >= start or value < end

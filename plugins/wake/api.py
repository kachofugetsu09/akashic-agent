from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, field_validator

from agent.plugin_composition import EmitEventKey
from agent.plugin_contracts.proactive import (
    DRIFT_DELIVERY as DRIFT_DELIVERY,
    DRIFT_WAKE as DRIFT_WAKE,
    EVENTMAIL_DELIVERY as EVENTMAIL_DELIVERY,
    EVENTMAIL_WAKE as EVENTMAIL_WAKE,
    ContentWakeServices as ContentWakeServices,
    DeliveryServices as DeliveryServices,
    DriftWakeServices as DriftWakeServices,
)


class DeliveryTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str
    recipient: str
    session_id: str

    @field_validator("channel", "recipient", "session_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("Wake delivery target 必须非空且无首尾空白")
        return value


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delivery: DeliveryTarget | None = None
    timezone: str = "Asia/Shanghai"
    investigation_tools: tuple[str, ...] = ("recall_memory", "web_fetch")

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("Wake timezone 必须非空且无首尾空白")
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Wake timezone 无效: {value}") from error
        return value


EVENTMAIL_CHANGED = EmitEventKey[None]("eventmail.changed")

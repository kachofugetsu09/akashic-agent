from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, field_validator

from agent.plugin_composition import EmitEventKey
from plugins.models.contract import ModelChoice, ModelUse
from plugins.eventmail.contract import EVENTMAIL_DELIVERY_V2 as EVENTMAIL_DELIVERY, EVENTMAIL_WAKE_V2 as EVENTMAIL_WAKE, ContentWakeServicesV2 as ContentWakeServices, EventMailDeliveryServicesV2 as DeliveryServices
from plugins.drift.contract import DRIFT_DELIVERY_V2 as DRIFT_DELIVERY, DRIFT_WAKE_V2 as DRIFT_WAKE, DriftWakeServicesV2 as DriftWakeServices


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


# 主动消息默认沿用接收对话里选过的模型（没选过就是默认模型）；也可固定跟随默认或指定模型。
WAKE_MODEL = ModelUse("wake", "主动联系", follows=("session", "default"))


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = True
    delivery: DeliveryTarget | None = None
    timezone: str = "Asia/Shanghai"
    investigation_tools: tuple[str, ...] = ("recall_memory", "web_fetch")
    model: ModelChoice = ModelChoice(follow="session")

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: ModelChoice) -> ModelChoice:
        return WAKE_MODEL.check(value)

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

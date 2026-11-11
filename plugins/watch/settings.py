"""值班的配置与设置表单；不配置主对话就不运行。"""
from __future__ import annotations

import re
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from plugins.models.contract import ModelChoice, ModelUse
from plugins.plugin_settings.contract import SettingsField, SettingsForm

WAKE_MODEL = ModelUse("watch", "值班叫醒后的主 Agent")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # 主对话的会话 ID，形如 akashic:xxxx 或 telegram:123；主 Agent 想说话时发到这里。
    target: str = ""
    interval_minutes: int = Field(default=10, ge=1, le=240)
    # 值班员用的模型 ID，按顺序退避；留空用默认模型。
    watch_models: tuple[str, ...] = ()
    model: ModelChoice = ModelChoice(follow="default")
    daily_wakes: int = Field(default=8, ge=0, le=100)
    # 待看列表有新条目、而主 Agent 已经这么久没醒过时，叫它过一眼；0 表示不这样做。
    digest_hours: int = Field(default=4, ge=0, le=72)
    review_at: str = "05:00"
    timezone: str = "Asia/Shanghai"
    memory_file: str = "memory/MEMORY.md"

    @field_validator("target")
    @classmethod
    def _target(cls, value: str) -> str:
        value = value.strip()
        if value and (":" not in value or value.startswith(":") or value.endswith(":")):
            raise ValueError("主对话需要写成 渠道:会话 的形式，例如 akashic:1234abcd")
        return value

    @field_validator("review_at")
    @classmethod
    def _clock(cls, value: str) -> str:
        value = value.strip()
        if value != "off" and re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", value) is None:
            raise ValueError("复盘时间写成 HH:MM，不复盘写 off")
        return value

    @field_validator("timezone")
    @classmethod
    def _zone(cls, value: str) -> str:
        _ = ZoneInfo(value)
        return value

    @field_validator("model")
    @classmethod
    def _model(cls, value: ModelChoice) -> ModelChoice:
        return WAKE_MODEL.check(value)

    @field_validator("watch_models")
    @classmethod
    def _lines(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.strip() for item in value if item.strip())


FORM = SettingsForm(
    title="值班",
    description="便宜的模型看着事件流，只在该叫的时候叫醒主 Agent；主 Agent 在后台醒来后自己决定做什么、说不说话。",
    fields=(
        SettingsField("target", "主对话", hint="主 Agent 想说话时发到的会话 ID，例如 akashic:1234abcd。留空则值班不运行。"),
        SettingsField("interval_minutes", "值班间隔（分钟）", kind="number", hint="每隔这么久把新事件攒成一批看一次。"),
        SettingsField("watch_models", "值班模型", kind="lines", hint="每行一个模型 ID，按顺序退避：前一个失败或输出不合格就换下一个。留空用默认模型。"),
        SettingsField("model", "主 Agent 模型", kind="model", use=WAKE_MODEL),
        SettingsField("daily_wakes", "每天最多叫醒几次", kind="number", hint="超出的不丢，放进待看列表。"),
        SettingsField("digest_hours", "待看列表多久看一次（小时）", kind="number",
                      hint="待看列表有新条目、主 Agent 又这么久没醒过时叫它过一眼；0 表示只在被叫醒和复盘时看。"),
        SettingsField("review_at", "每日复盘时间", hint="HH:MM；写 off 表示不复盘。"),
        SettingsField("timezone", "时区"),
    ),
)

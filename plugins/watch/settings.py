"""值班的配置与设置表单；不配置主对话就不运行。"""
from __future__ import annotations

from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from plugins.plugin_settings.contract import SettingsField, SettingsForm

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # 主对话的会话 ID，形如 akashic:xxxx 或 telegram:123；值班员从这里读最近的对话。
    target: str = ""
    interval_minutes: int = Field(default=10, ge=1, le=240)
    timezone: str = "Asia/Shanghai"

    @field_validator("target")
    @classmethod
    def _target(cls, value: str) -> str:
        value = value.strip()
        if value and (":" not in value or value.startswith(":") or value.endswith(":")):
            raise ValueError("主对话需要写成 渠道:会话 的形式，例如 akashic:1234abcd")
        return value

    @field_validator("timezone")
    @classmethod
    def _zone(cls, value: str) -> str:
        _ = ZoneInfo(value)
        return value


FORM = SettingsForm(
    title="值班",
    description="把各个来源的事件收成同一种形状，按天记在工作区 watch/events/ 目录。",
    fields=(
        SettingsField("target", "主对话", hint="用户平时聊天的会话 ID，例如 akashic:1234abcd。留空则值班不运行。"),
        SettingsField("interval_minutes", "值班间隔（分钟）", kind="number", hint="每隔这么久收一次新事件。"),
        SettingsField("timezone", "时区"),
    ),
)

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
    # 值班员用的模型 ID，按顺序退避；留空用默认模型。
    watch_models: tuple[str, ...] = ()
    timezone: str = "Asia/Shanghai"
    memory_file: str = "memory/MEMORY.md"

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

    @field_validator("watch_models")
    @classmethod
    def _lines(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.strip() for item in value if item.strip())


FORM = SettingsForm(
    title="值班",
    description="便宜的模型看着事件流，给每条事件一个去向，记录在工作区 watch/ 目录。",
    fields=(
        SettingsField("target", "主对话", hint="用户平时聊天的会话 ID，例如 akashic:1234abcd。留空则值班不运行。"),
        SettingsField("interval_minutes", "值班间隔（分钟）", kind="number", hint="每隔这么久把新事件攒成一批看一次。"),
        SettingsField("watch_models", "值班模型", kind="lines", hint="每行一个模型 ID，按顺序退避：前一个失败或输出不合格就换下一个。留空用默认模型。"),
        SettingsField("timezone", "时区"),
    ),
)

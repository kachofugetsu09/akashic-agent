from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QQGroupConfig(BaseModel):
    """Own the legacy NapCat group filter values inside the QQ plugin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str = Field(min_length=1)
    allow_from: tuple[str, ...] = ()
    require_at: bool = True


class QQChannelConfig(BaseModel):
    """Validate the redacted legacy NapCat channel projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    bot_uin: str = ""
    allow_from: tuple[str, ...] = ()
    groups: tuple[QQGroupConfig, ...] = ()
    websocket_open_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_enabled_identity(self) -> QQChannelConfig:
        if self.enabled and not self.bot_uin.strip():
            raise ValueError("启用 QQ channel 需要 bot_uin")
        seen: set[str] = set()
        for group in self.groups:
            if group.group_id in seen:
                raise ValueError(f"QQ 群配置重复: {group.group_id}")
            seen.add(group.group_id)
        return self

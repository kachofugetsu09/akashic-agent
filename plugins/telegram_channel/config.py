from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.plugin_composition import CredentialRef


class TelegramChannelConfig(BaseModel):
    """Validate the redacted Telegram channel configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    token: CredentialRef | None = None
    allow_from: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_enabled_credentials(self) -> TelegramChannelConfig:
        if self.enabled and self.token is None:
            raise ValueError("启用 Telegram channel 需要 token")
        return self

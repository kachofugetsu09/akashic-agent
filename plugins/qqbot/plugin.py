from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from pydantic import AliasChoices, BaseModel, Field, field_validator

from agent.plugins import Plugin
from .channel import QQBotChannel

if TYPE_CHECKING:
    from infra.channels.contract import Channel

_UNRESOLVED_ENV_RE = re.compile(r"^\$\{\w+\}$")


class QQBotGroupConfigModel(BaseModel):
    group_openid: str = Field(
        default="",
        validation_alias=AliasChoices("group_openid", "groupOpenid"),
    )
    allow_from: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("allow_from", "allowFrom"),
    )
    require_at: bool = Field(
        default=True,
        validation_alias=AliasChoices("require_at", "requireAt"),
    )
    allow_proactive: bool = Field(
        default=False,
        validation_alias=AliasChoices("allow_proactive", "allowProactive"),
    )


class QQBotConfigModel(BaseModel):
    app_id: str = Field(
        default="",
        validation_alias=AliasChoices("app_id", "appId"),
    )
    client_secret: str = Field(
        default="",
        validation_alias=AliasChoices("client_secret", "clientSecret"),
    )
    allow_from: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("allow_from", "allowFrom"),
    )
    groups: list[QQBotGroupConfigModel] = Field(default_factory=list)

    @field_validator("app_id", "client_secret", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str:
        text = str(value or "").strip()
        if _UNRESOLVED_ENV_RE.fullmatch(text):
            return ""
        return text


class QQBotPlugin(Plugin):
    name = "qqbot"
    desc = "官方 QQBot 渠道"
    ConfigModel = QQBotConfigModel

    def channels(self) -> list["Channel"]:
        config = cast(QQBotConfigModel | None, self.context.config)
        if config is None or not config.app_id or not config.client_secret:
            return []
        return [
            QQBotChannel(
                app_id=config.app_id,
                client_secret=config.client_secret,
                allow_from=config.allow_from,
                groups=config.groups,
            )
        ]

from __future__ import annotations

import re
from typing import cast

from pydantic import AliasChoices, BaseModel, Field, field_validator

_UNRESOLVED_ENV_RE = re.compile(r"^\$\{\w+\}$")
_DEFAULT_DOMAIN = "https://open.feishu.cn"


# 飞书插件配置，来自主 config.toml 的 [plugins.feishu]，支持 ${ENV} 插值。
class FeishuConfigModel(BaseModel):
    app_id: str = Field(
        default="",
        validation_alias=AliasChoices("app_id", "appId"),
    )
    app_secret: str = Field(
        default="",
        validation_alias=AliasChoices("app_secret", "appSecret"),
    )
    allow_from: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("allow_from", "allowFrom"),
    )
    domain: str = Field(default=_DEFAULT_DOMAIN)

    @field_validator("app_id", "app_secret", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str:
        text = str(value or "").strip()
        if _UNRESOLVED_ENV_RE.fullmatch(text):
            return ""
        return text

    @field_validator("domain", mode="before")
    @classmethod
    def _normalize_domain(cls, value: object) -> str:
        text = str(value or "").strip()
        return (text or _DEFAULT_DOMAIN).rstrip("/")

    @field_validator("allow_from", mode="before")
    @classmethod
    def _normalize_allow_from(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in cast(list[object], value):
            text = str(item).strip()
            if text:
                result.append(text)
        return result

"""固定配置的 Telegram 出站；不创建 Bot 收件实例。"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Self
from urllib.parse import urlsplit

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.plugin_composition import CREDENTIALS, Context, CredentialRef
from agent.plugin_composition.artifacts import ARTIFACT_READ
from agent.plugin_composition.messages import MESSAGE_CATALOG
from plugins.delivery.senders import DELIVERY_SENDERS

from .sender import TelegramSender

api_version = 3
name = "telegram_sender"
version = "1.0.0"
desc = "用固定凭据发送 Telegram 正文和附件"
inject = (DELIVERY_SENDERS, CREDENTIALS, MESSAGE_CATALOG, ARTIFACT_READ)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    token: CredentialRef | None = None
    channel: str = Field(default="telegram", pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    api_base: str = "https://api.telegram.org"
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check(self) -> Self:
        url = urlsplit(self.api_base)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("api_base 必须是无凭据的 HTTP(S) URL")
        if self.enabled and self.token is None:
            raise ValueError("启用 Telegram Sender 需要 token")
        return self


async def apply(ctx: Context, config: Config) -> None:
    if not config.enabled:
        return
    token_ref = config.token
    assert token_ref is not None

    @asynccontextmanager
    async def open_sender() -> AsyncGenerator[TelegramSender]:
        async with ctx.require(CREDENTIALS).open(ctx, {"token": token_ref}) as credentials:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout_seconds), trust_env=True) as client:
                yield TelegramSender(client, config.api_base, credentials.credential(token_ref),
                                     ctx.require(MESSAGE_CATALOG), ctx.require(ARTIFACT_READ))

    _ = await ctx.require(DELIVERY_SENDERS).register(ctx, name=config.channel, idempotent=False, open=open_sender)

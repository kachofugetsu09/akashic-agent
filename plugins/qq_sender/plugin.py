"""固定 OneBot WebSocket 配置的 QQ 出站。"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidHandshake

from agent.plugin_composition import CREDENTIALS, Context, CredentialRef
from agent.plugin_composition.artifacts import ARTIFACT_READ
from agent.plugin_composition.messages import MESSAGE_CATALOG
from plugins.delivery.senders import DELIVERY_SENDERS

from .sender import QQSender

api_version = 3
name = "qq_sender"
version = "1.0.0"
desc = "通过固定 OneBot 连接发送 QQ 正文和附件"
inject = (DELIVERY_SENDERS, CREDENTIALS, MESSAGE_CATALOG, ARTIFACT_READ)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    endpoint: str | None = None
    token: CredentialRef | None = None
    channel: str = Field(default="qq", pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check(self) -> Self:
        if self.enabled and self.endpoint is None:
            raise ValueError("启用 QQ Sender 需要 OneBot WS endpoint")
        if self.endpoint is not None:
            url = urlsplit(self.endpoint)
            if (url.scheme not in {"ws", "wss"} or not url.hostname or url.username or url.password
                    or url.query or url.fragment or url.path.rstrip("/") == "/event"):
                raise ValueError("endpoint 必须是无凭据的 OneBot WS API URL")
        return self


async def apply(ctx: Context, config: Config) -> None:
    if not config.enabled:
        return
    endpoint = config.endpoint
    assert endpoint is not None
    refs = {} if config.token is None else {"token": config.token}
    # WebSocket DEBUG 会打印 Authorization；只隔离本连接的低层日志。
    connection_logger = logging.Logger(__name__)
    connection_logger.addHandler(logging.NullHandler())
    connection_logger.propagate = False

    @asynccontextmanager
    async def open_sender() -> AsyncGenerator[QQSender]:
        async with ctx.require(CREDENTIALS).open(ctx, refs) as credentials:
            headers = {} if config.token is None else {"Authorization": "Bearer " + credentials.credential(config.token)}
            try:
                async with connect(endpoint, additional_headers=headers, open_timeout=config.timeout_seconds,
                                   close_timeout=config.timeout_seconds, proxy=None, logger=connection_logger) as connection:
                    yield QQSender(connection, config.timeout_seconds, ctx.require(MESSAGE_CATALOG), ctx.require(ARTIFACT_READ))
            except InvalidHandshake as error:
                # 错误对象可带服务端回显的请求头；保持失败类型而不传播 secret。
                raise ConnectionError(f"QQ 连接握手失败：{type(error).__name__}") from None

    _ = await ctx.require(DELIVERY_SENDERS).register(ctx, name=config.channel, idempotent=False, open=open_sender)

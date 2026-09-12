from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from session.message import Message

api_version = 3
name = "akashic_sender"
version = "1.0.0"
desc = "确认目标会话已保存的消息，Web 和 Mobile 从日志订阅"


@dataclass(frozen=True, slots=True)
class SendResult:
    """Akashic provider 的本地结果；Delivery 在注册边界重新校验它。"""

    status: Literal["delivered", "rejected", "failed"]
    provider_ids: tuple[str, ...] = ()
    error: str | None = None


class SenderTarget(Protocol):
    idempotent: bool

    async def send(self, key: str, address: str, message: Message) -> SendResult: ...

    async def query(self, key: str, address: str) -> SendResult | None: ...


class SenderRegistry(Protocol):
    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        idempotent: bool,
        open: Callable[[], AbstractAsyncContextManager[SenderTarget]],
    ) -> Effect: ...


DELIVERY_SENDERS = ServiceKey[SenderRegistry]("delivery.senders.v1")
inject = (DELIVERY_SENDERS,)


class Sender:
    """保存到目标 Session 即送达；在线客户端不拥有另一份正文或终态。"""

    idempotent = True

    async def send(self, key: str, address: str, message: Message) -> SendResult:
        # Delivery 只传入已提交的原 Message；这里核对它是否属于收件地址。
        if message.session_id != "akashic:" + address:
            return SendResult(status="rejected", error="消息未保存在 Akashic 收件地址对应的 Session")
        return SendResult(status="delivered", provider_ids=(message.message_id,))

    async def query(self, key: str, address: str) -> SendResult | None:
        # 没有独立网络效果可查询；原 key 的幂等 send 只确认已保存的正文。
        return None


async def apply(ctx: Context, config: object) -> None:
    @asynccontextmanager
    async def open() -> AsyncGenerator[Sender]:
        yield Sender()

    _ = await ctx.require(DELIVERY_SENDERS).register(
        ctx, name="akashic", idempotent=True, open=open,
    )

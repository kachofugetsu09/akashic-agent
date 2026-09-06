from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from agent.plugin_composition import Context
from plugins.delivery.api import Receipt
from plugins.delivery.senders import DELIVERY_SENDERS
from session.message import Message

api_version = 3
name = "akashic_sender"
version = "1.0.0"
desc = "确认目标会话已保存的消息，Web 和 Mobile 从日志订阅"
inject = (DELIVERY_SENDERS,)


class Sender:
    """保存到目标 Session 即送达；在线客户端不拥有另一份正文或终态。"""

    idempotent = True

    async def send(self, key: str, address: str, message: Message) -> Receipt:
        # Delivery 只传入已提交的原 Message；这里核对它是否属于收件地址。
        if message.session_id != "akashic:" + address:
            return Receipt(status="rejected", error="消息未保存在 Akashic 收件地址对应的 Session")
        return Receipt(status="delivered", provider_ids=(message.message_id,))

    async def query(self, key: str, address: str) -> Receipt | None:
        # 没有独立网络效果可查询；原 key 的幂等 send 只确认已保存的正文。
        return None


async def apply(ctx: Context, config: object) -> None:
    @asynccontextmanager
    async def open() -> AsyncGenerator[Sender]:
        yield Sender()

    _ = await ctx.require(DELIVERY_SENDERS).register(
        ctx, name="akashic", idempotent=True, open=open,
    )
